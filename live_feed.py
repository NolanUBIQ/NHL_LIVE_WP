"""
Live game-state feeds for the NHL live pricer.

Everything downstream (live_pricer, live_run, live_ui) only ever sees a
GameState. To plug in a different data source, either

  - write a class with a `poll() -> GameState` method, or
  - create a PushFeed and call `feed.push(state)` from your own callback /
    websocket thread whenever the provider sends an update.

Feeds included:
  NHLApiFeed  - the NHL's public gamecenter play-by-play (api-web.nhle.com)
  ReplayFeed  - replays a finished game's play-by-play on a sped-up clock
  PushFeed    - holds the latest state some other code pushed in

Power plays. The play-by-play gives each penalty's duration but not the time
left on it, so PenaltyBox rebuilds the penalty clock from the events: stacked
penalties (max two serving per team), coincidental penalties cancelling, a PP
goal ending the earliest minor (or the first half of a double minor), majors
running in full. The latest `situationCode` after the last penalty/goal is
used as a cross-check on whether a man advantage exists. During live games the
API's `situation` block, when present, overrides the seconds remaining.

situationCode digits: [away goalie][away skaters][home skaters][home goalie],
e.g. 1551 even, 1451 home PP, 1351 home 5-on-3, 1560 home goalie pulled.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import date


def _enable_ssl_dlls():
    """Conda Pythons only find their OpenSSL DLLs when the env is activated."""
    if os.name != "nt":
        return
    lib_bin = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(lib_bin):
        os.environ["PATH"] = lib_bin + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(lib_bin)


_enable_ssl_dlls()
import urllib.request  # noqa: E402  (after the DLL fix so ssl loads)

API = "https://api-web.nhle.com/v1"
USER_AGENT = "Mozilla/5.0 (NHL live WP pricer)"

REG_SECONDS = 3600
PERIOD_SECONDS = 1200
RS_OT_SECONDS = 300            # regular-season 3-on-3 overtime


def get_json(path: str, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(f"{API}/{path.lstrip('/')}",
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def find_games(day: str | date | None = None) -> list[dict]:
    """Games on a date (default: today per the API) as small dicts."""
    data = get_json(f"score/{day}" if day else "score/now")
    out = []
    for g in data.get("games", []):
        out.append(dict(id=str(g["id"]), away=g["awayTeam"]["abbrev"],
                        home=g["homeTeam"]["abbrev"], state=g.get("gameState", ""),
                        start=g.get("startTimeUTC", ""),
                        away_score=g["awayTeam"].get("score"),
                        home_score=g["homeTeam"].get("score")))
    return out


def find_game_id(team_or_id: str, day: str | None = None) -> str:
    """A 10-digit game id passes through; a team abbrev finds that team's game."""
    s = team_or_id.strip().upper()
    if s.isdigit():
        return s
    for g in find_games(day):
        if s in (g["home"], g["away"]):
            return g["id"]
    raise LookupError(f"no game found for {team_or_id!r} on {day or 'today'}")


# ── Clock helpers ────────────────────────────────────────────────────────────

def period_start(n: int, playoff: bool) -> int:
    if n <= 3 or playoff:
        return (n - 1) * PERIOD_SECONDS
    return REG_SECONDS + (n - 4) * RS_OT_SECONDS


def period_length(n: int, playoff: bool, period_type: str = "REG") -> int:
    if period_type == "SO":
        return 0
    return PERIOD_SECONDS if (n <= 3 or playoff) else RS_OT_SECONDS


def _mmss(s: str) -> int:
    mm, ss = s.split(":")
    return int(mm) * 60 + int(ss)


def fmt_clock(seconds: int) -> str:
    seconds = max(int(seconds), 0)
    return f"{seconds // 60}:{seconds % 60:02d}"


# ── State ────────────────────────────────────────────────────────────────────

@dataclass
class GameEvent:
    elapsed: int
    kind: str                  # goal / penalty / period / final
    team: str | None           # "home" / "away"
    text: str


@dataclass
class GameState:
    game_id: str
    home: str                  # team abbrevs
    away: str
    status: str = "PRE"        # PRE / LIVE / INT / FINAL
    period: int = 1
    period_type: str = "REG"   # REG / OT / SO
    elapsed: int = 0           # game seconds since puck drop (OT continues past 3600)
    clock_running: bool = False
    home_score: int = 0
    away_score: int = 0
    home_skaters: int = 5
    away_skaters: int = 5
    home_goalie_pulled: bool = False
    away_goalie_pulled: bool = False
    pp_team: str | None = None # team with the man advantage
    pp_seconds: int = 0        # until the advantage ends (if no goals)
    pp_kind: str = "minor"     # minor: a PP goal ends it; major: runs in full
    pp_two_man_seconds: int = 0  # part of pp_seconds at a two-man advantage
    pp_estimated: bool = False # PP seen in situationCode but not rebuilt from events
    winner: str | None = None
    playoff: bool = False
    events: list[GameEvent] = field(default_factory=list)
    fetched_at: float = 0.0

    @property
    def d(self) -> int:
        return self.home_score - self.away_score

    def team(self, side: str | None) -> str:
        return {"home": self.home, "away": self.away}.get(side, "")

    @property
    def clock(self) -> str:
        if self.status == "PRE":
            return "PRE"
        if self.status == "FINAL":
            return {"OT": "FINAL/OT", "SO": "FINAL/SO"}.get(self.period_type, "FINAL")
        if self.period_type == "SO":
            return "SO"
        name = "OT" if self.period_type == "OT" and not self.playoff else (
            f"OT{self.period - 3}" if self.period_type == "OT" else f"P{self.period}")
        if self.status == "INT":
            return f"END {name}"
        end = period_start(self.period, self.playoff) + period_length(
            self.period, self.playoff, self.period_type)
        return f"{name} {fmt_clock(end - self.elapsed)}"

    @property
    def situation(self) -> str:
        if self.status in ("PRE", "FINAL"):
            return ""
        if self.status == "INT":
            return "Intermission"
        if self.pp_team:
            sk = (f"{self.home_skaters}v{self.away_skaters}" if self.pp_team == "home"
                  else f"{self.away_skaters}v{self.home_skaters}")
            est = "~" if self.pp_estimated else ""
            kind = " major" if self.pp_kind == "major" else ""
            return f"{self.team(self.pp_team)} PP{kind} {est}{fmt_clock(self.pp_seconds)} ({sk})"
        pulled = [self.team(s) for s, f in (("home", self.home_goalie_pulled),
                                            ("away", self.away_goalie_pulled)) if f]
        if pulled:
            return f"{' & '.join(pulled)} goalie pulled"
        if self.home_skaters == self.away_skaters and self.home_skaters != 5:
            return f"{self.home_skaters}v{self.away_skaters}"
        return "Even strength"

    def key(self) -> tuple:
        """Fields whose change counts as an event (not just the clock ticking)."""
        return (self.status, self.period, self.home_score, self.away_score,
                self.pp_team, self.pp_kind, self.home_goalie_pulled, self.away_goalie_pulled)


def advance_state(st: GameState, seconds: float) -> GameState:
    """Run the clock forward locally between feed updates (capped at period end)."""
    if st.status != "LIVE" or not st.clock_running or seconds <= 0:
        return st
    end = period_start(st.period, st.playoff) + period_length(st.period, st.playoff,
                                                              st.period_type)
    step = int(min(st.elapsed + seconds, end) - st.elapsed)
    if step <= 0:
        return st
    out = replace(st, elapsed=st.elapsed + step)
    if st.pp_team:
        left = st.pp_seconds - step
        if left <= 0:
            out = replace(out, pp_team=None, pp_seconds=0, pp_two_man_seconds=0,
                          home_skaters=max(st.home_skaters, st.away_skaters),
                          away_skaters=max(st.home_skaters, st.away_skaters))
        else:
            out = replace(out, pp_seconds=left,
                          pp_two_man_seconds=max(st.pp_two_man_seconds - step, 0))
    return out


# ── Penalty clock ────────────────────────────────────────────────────────────

@dataclass
class _Pen:
    team: str
    start: int
    end: int
    kind: str                  # minor / double / major


class PenaltyBox:
    MAX_SERVING = 2

    def __init__(self):
        self.active: dict[str, list[_Pen]] = {"home": [], "away": []}
        self.queued: dict[str, list[tuple[int, str]]] = {"home": [], "away": []}

    def counts(self) -> tuple[int, int]:
        return len(self.active["home"]), len(self.active["away"])

    def _remove(self, p: _Pen, at: int):
        self.active[p.team].remove(p)
        if self.queued[p.team]:
            dur, kind = self.queued[p.team].pop(0)
            self.active[p.team].append(_Pen(p.team, at, at + dur, kind))

    def expire(self, t: int):
        while True:
            done = [p for side in self.active.values() for p in side if p.end <= t]
            if not done:
                return
            p = min(done, key=lambda x: x.end)
            self._remove(p, p.end)

    def add(self, t: int, pens: list[tuple[str, int, str]]):
        """Penalties assessed at the same time: (team, seconds, kind)."""
        self.expire(t)
        home = [x for x in pens if x[0] == "home"]
        away = [x for x in pens if x[0] == "away"]
        for h in list(home):                       # coincidental: cancel pairwise
            match = next((a for a in away if a[1] == h[1]), None)
            if match:
                home.remove(h)
                away.remove(match)
        for team, dur, kind in home + away:
            if len(self.active[team]) < self.MAX_SERVING:
                self.active[team].append(_Pen(team, t, t + dur, kind))
            else:
                self.queued[team].append((dur, kind))

    def goal(self, t: int, scorer: str):
        """A goal by the team with more skaters ends the earliest non-major."""
        self.expire(t)
        other = "away" if scorer == "home" else "home"
        if len(self.active[other]) <= len(self.active[scorer]):
            return
        cands = [p for p in self.active[other] if p.kind != "major"]
        if not cands:
            return
        p = min(cands, key=lambda x: x.end)
        if p.kind == "double" and p.end - t > 120:
            p.end, p.kind = t + 120, "minor"       # first half of a double minor
        else:
            self._remove(p, t)

    def project(self, t: int) -> tuple[str | None, int, int, str]:
        """(pp_team, seconds left, two-man seconds, kind) if nothing else happens."""
        box = copy.deepcopy(self)
        box.expire(t)
        nh, na = box.counts()
        if nh == na:
            return None, 0, 0, "minor"
        team = "home" if na > nh else "away"
        sh = "away" if team == "home" else "home"
        kind = "major" if all(p.kind == "major" for p in box.active[sh]) else "minor"
        now, two = t, 0
        while True:
            nh, na = box.counts()
            adv = na - nh
            if adv == 0 or (adv > 0) != (team == "home"):
                break
            nxt = min(p.end for side in box.active.values() for p in side)
            if abs(adv) >= 2:
                two += nxt - now
            box.expire(nxt)
            now = nxt
        return team, now - t, two, kind


def _skaters(nh: int, na: int, period_type: str) -> tuple[int, int]:
    if period_type == "OT_RS":             # 3-on-3: a penalty adds an attacker
        return min(3 + max(na - nh, 0), 5), min(3 + max(nh - na, 0), 5)
    return max(5 - nh, 3), max(5 - na, 3)


def _parse_code(code: str):
    """(away_goalie_in, away_sk, home_sk, home_goalie_in) or None."""
    if not code or len(code) != 4 or not code.isdigit():
        return None
    return code[0] == "1", int(code[1]), int(code[2]), code[3] == "1"


def _penalty_kind(det: dict) -> tuple[int, str] | None:
    """(seconds, kind) for strength-affecting penalties, else None."""
    mins = int(det.get("duration") or 0)
    if det.get("typeCode") in ("MIS", "GMIS") or not 0 < mins < 10:
        return None
    return mins * 60, {2: "minor", 4: "double"}.get(mins, "major")


# ── Play-by-play -> GameState ────────────────────────────────────────────────

def state_from_pbp(pbp: dict, upto: int | None = None,
                   use_situation_codes: bool = True) -> GameState:
    """
    Game state from a gamecenter play-by-play payload.

    upto=None : live use; the clock comes from the payload's `clock` block.
    upto=t    : the state at game second t (replays), using plays up to t.
    use_situation_codes : cross-check the rebuilt PP against situationCode
        (off only to test the penalty clock on its own)
    """
    playoff = pbp.get("gameType") == 3
    home_id = pbp["homeTeam"]["id"]
    st = GameState(game_id=str(pbp["id"]), home=pbp["homeTeam"]["abbrev"],
                   away=pbp["awayTeam"]["abbrev"], playoff=playoff)

    def side(team_id):
        return None if team_id is None else ("home" if team_id == home_id else "away")

    def play_time(p):
        pd_ = p["periodDescriptor"]
        n, ptype = pd_["number"], pd_.get("periodType", "REG")
        if ptype == "SO":
            return period_start(n, playoff)
        return period_start(n, playoff) + _mmss(p.get("timeInPeriod", "00:00"))

    box = PenaltyBox()
    plays = pbp.get("plays", [])
    period, ptype, last_t = 1, "REG", 0
    last_code, code_after_change = None, False
    hs = as_ = 0
    i = 0
    while i < len(plays):
        p = plays[i]
        t = play_time(p)
        if upto is not None and t > upto:
            break
        pd_ = p["periodDescriptor"]
        period, ptype, last_t = pd_["number"], pd_.get("periodType", "REG"), t
        kind, det = p.get("typeDescKey"), p.get("details", {}) or {}
        team = side(det.get("eventOwnerTeamId"))
        clock = fmt_clock(_mmss(p.get("timeInPeriod", "00:00")))
        label = f"P{period}" if ptype == "REG" else ptype

        if kind == "penalty":
            group, j = [p], i + 1
            while (j < len(plays) and plays[j].get("typeDescKey") == "penalty"
                   and play_time(plays[j]) == t):
                group.append(plays[j])
                j += 1
            pens = []
            for q in group:
                qd = q.get("details", {}) or {}
                qt = side(qd.get("eventOwnerTeamId"))
                pk = _penalty_kind(qd)
                desc = (qd.get("descKey") or "penalty").replace("-", " ")
                st.events.append(GameEvent(t, "penalty", qt,
                                           f"{label} {clock}  {st.team(qt)} {qd.get('duration', '?')} min {desc}"))
                if pk and qt and ptype != "SO":
                    pens.append((qt, pk[0], pk[1]))
            if pens:
                box.add(t, pens)
                code_after_change = False
            i = j
            continue

        if kind == "goal" and ptype != "SO" and team:
            box.expire(t)
            nh, na = box.counts()
            mine, theirs = (nh, na) if team == "home" else (na, nh)
            code = _parse_code(p.get("situationCode", ""))
            tag = ""
            if mine < theirs:
                tag = "PPG"
            elif mine > theirs:
                tag = "SHG"
            elif code and not (code[0] if team == "home" else code[3]):
                tag = "EN"
            box.goal(t, team)
            hs = int(det.get("homeScore", hs))
            as_ = int(det.get("awayScore", as_))
            st.events.append(GameEvent(t, "goal", team,
                                       f"{label} {clock}  {st.team(team)} goal"
                                       f"{'  (' + tag + ')' if tag else ''}   {st.away} {as_}-{hs} {st.home}"))
            code_after_change = False
        elif kind == "period-end" and ptype != "SO":
            st.events.append(GameEvent(t, "period", None, f"End of {label}   {st.away} {as_}-{hs} {st.home}"))
        elif kind != "goal" and p.get("situationCode"):
            last_code, code_after_change = p["situationCode"], True
        i += 1

    st.home_score, st.away_score = hs, as_
    game_state = pbp.get("gameState", "")
    final_payload = game_state in ("FINAL", "OFF")

    if upto is None:
        clock = pbp.get("clock") or {}
        pdesc = pbp.get("periodDescriptor") or {}
        period = pdesc.get("number", period)
        ptype = pdesc.get("periodType", ptype)
        if game_state in ("FUT", "PRE") or not plays:
            st.status, elapsed = "PRE", 0
        elif final_payload:
            st.status, elapsed = "FINAL", last_t
        else:
            plen = period_length(period, playoff, ptype)
            if clock.get("inIntermission"):
                st.status, elapsed = "INT", period_start(period, playoff) + plen
            else:
                st.status = "LIVE"
                elapsed = period_start(period, playoff) + plen - int(clock.get("secondsRemaining", plen))
                st.clock_running = bool(clock.get("running"))
    else:
        end_t = play_time(plays[-1]) if plays else 0
        if final_payload and upto >= end_t:
            st.status, elapsed = "FINAL", end_t
        else:
            st.status, st.clock_running = "LIVE", True
            elapsed = max(upto, 0)
            if elapsed < REG_SECONDS:
                period, ptype = elapsed // PERIOD_SECONDS + 1, "REG"
            elif period < 4:
                period, ptype = 4, "OT"
    st.period, st.period_type, st.elapsed = period, ptype, int(elapsed)

    if st.status == "FINAL":
        st.home_score = int(pbp["homeTeam"].get("score", hs))
        st.away_score = int(pbp["awayTeam"].get("score", as_))
        st.winner = "home" if st.home_score > st.away_score else "away"
        st.period_type = (pbp.get("gameOutcome") or {}).get("lastPeriodType", st.period_type)
        st.events.append(GameEvent(st.elapsed, "final", st.winner,
                                   f"{st.clock}   {st.away} {st.away_score}-{st.home_score} {st.home}"))
        return st
    if st.status == "PRE" or ptype == "SO":
        return st

    # power play from the rebuilt penalty clock
    box.expire(st.elapsed)
    nh, na = box.counts()
    st.home_skaters, st.away_skaters = _skaters(
        nh, na, "OT_RS" if ptype == "OT" and not playoff else "REG")
    st.pp_team, st.pp_seconds, st.pp_two_man_seconds, st.pp_kind = box.project(st.elapsed)

    code = _parse_code(last_code or "")
    if code:
        st.away_goalie_pulled, st.home_goalie_pulled = not code[0], not code[3]
        if code_after_change and use_situation_codes:
            # skater edge net of any extra attacker for a pulled goalie
            adv = (code[2] - (not code[3])) - (code[1] - (not code[0]))
            if adv == 0 and st.pp_team:
                st.pp_team, st.pp_seconds, st.pp_two_man_seconds = None, 0, 0
            elif adv != 0 and not st.pp_team:
                st.pp_team = "home" if adv > 0 else "away"
                st.pp_seconds, st.pp_estimated = 60, True
                st.pp_kind = "minor"
            if adv != 0:
                st.home_skaters, st.away_skaters = code[2], code[1]
    return st


def apply_live_situation(st: GameState, pbp: dict) -> GameState:
    """Use the live `situation` block's PP clock when the payload carries one."""
    sit = pbp.get("situation")
    if not isinstance(sit, dict) or not st.pp_team:
        return st
    secs = sit.get("secondsRemaining")
    if secs is None and sit.get("timeRemaining"):
        try:
            secs = _mmss(sit["timeRemaining"])
        except ValueError:
            secs = None
    if secs is not None:
        st.pp_seconds = int(secs)
        st.pp_two_man_seconds = min(st.pp_two_man_seconds, st.pp_seconds)
        st.pp_estimated = False
    return st


# ── Feeds ────────────────────────────────────────────────────────────────────

class NHLApiFeed:
    """Polls the public play-by-play for one game."""

    def __init__(self, game_id: str):
        self.game_id = str(game_id)
        self.pbp: dict | None = None

    def poll(self) -> GameState:
        self.pbp = get_json(f"gamecenter/{self.game_id}/play-by-play")
        st = apply_live_situation(state_from_pbp(self.pbp), self.pbp)
        st.fetched_at = time.time()
        return st

    def state_at(self, elapsed: int) -> GameState:
        """Historical state from the last payload (for chart backfill)."""
        return state_from_pbp(self.pbp, upto=elapsed)


class ReplayFeed:
    """Replays a finished game at `speed` game-seconds per real second."""

    def __init__(self, game_id: str | None = None, pbp: dict | None = None,
                 speed: float = 60.0, start_elapsed: int = 0):
        self.pbp = pbp if pbp is not None else get_json(f"gamecenter/{game_id}/play-by-play")
        self.game_id = str(self.pbp["id"])
        self.speed = speed
        self.start_elapsed = start_elapsed
        self.t0 = time.time()

    def seek(self, elapsed: int):
        self.start_elapsed, self.t0 = elapsed, time.time()

    def now_elapsed(self) -> int:
        return int(self.start_elapsed + (time.time() - self.t0) * self.speed)

    def poll(self) -> GameState:
        st = state_from_pbp(self.pbp, upto=self.now_elapsed())
        st.fetched_at = time.time()
        st.clock_running = False       # the replay clock is driven by poll()
        return st

    def state_at(self, elapsed: int) -> GameState:
        return state_from_pbp(self.pbp, upto=elapsed)


class PushFeed:
    """Latest-state holder for push-style providers (websockets, callbacks)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state: GameState | None = None
        self.pbp = None

    def push(self, state: GameState):
        state.fetched_at = state.fetched_at or time.time()
        with self._lock:
            self._state = state

    def poll(self) -> GameState:
        with self._lock:
            if self._state is None:
                raise LookupError("no state pushed yet")
            return self._state
