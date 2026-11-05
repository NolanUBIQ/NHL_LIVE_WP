"""
Live pricer: GameState -> fair (no-vig) prices from the Skellam live model.

Prices produced for every state:
  - 2-way moneyline incl. OT/SO (home / away win-equivalent, fair American odds)
  - regulation 3-way (home / tie / away)
  - alt puck lines incl. OT/SO
  - live totals (final score incl. the OT/SO winner's goal), see below

Pregame inputs are a home win prob (incl. OT) and an expected total (incl. the
OT/SO goal), from the market or from the pregame model's predictions file;
lam / mu are solved so the model's own puck-drop prices, pulled-goalie goals
and OT goal included, reproduce them (fit_to_market). PP / SH scoring uses the
pregame model's special-teams ratings for the two teams (pregame_st.py), same
as the sim UI.

Game situations:
  - regulation: nhl_ingame_skellam.endgame_margin_dist with the live score,
    clock, PP window (minor / major) and pull-time endgame model: a team
    down 1 or 2 is assumed to pull at 2:00 / 3:20 left (rolled back 30 s
    each time that passes), re-pulls ~25 s after a goal, and the feed's
    goalie-out flag switches to the pulled state as soon as it happens;
    3rd-period score effects raise the trailing team's rate down 1 / 2
  - 5-on-3: the PP multiplier is the time-weighted mix of pp53_mult (two-man
    portion) and pp_mult over the whole advantage
  - OT / shootout: the game is tied, so home wins with p_ot
Totals use nhl_ingame_skellam.endgame_goals_dist (the same endgame chain over
each team's goals; before the endgame, independent Poissons with the PP at its
expected length), so they can differ very slightly from the exact margin
model used for the moneyline and puck lines.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

import nhl_ingame_skellam as m
import pregame_st
from live_feed import REG_SECONDS, GameState

PREDICTIONS_CSV = pregame_st.PREGAME_DIR / "data" / "predictions_today_tomorrow.csv"
DEFAULT_SPREADS = (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)
TOTAL_GRID = 16
# a goalie-out flag only counts as an endgame pull in the last this-many
# seconds of regulation (earlier ones are delayed-penalty pulls)
PULL_MAX_SECONDS = 360
NO_PULLS = m.EndgameParams(pull_seconds=0, pull2_seconds=0)   # endgame off


def parse_price(s: str) -> float:
    """Win prob from a probability (0.58), a percent (58) or American odds (-140)."""
    x = float(str(s).replace("−", "-").strip())
    if x <= -100 or x >= 100:
        return -x / (-x + 100) if x < 0 else 100 / (x + 100)
    if 0 < x < 1:
        return x
    if 1 <= x < 100:
        return x / 100
    raise ValueError(f"can't read a price from {s!r}")


def american(p: float) -> float | None:
    return m.prob_to_american(p) if 0 < p < 1 else None


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class PricerConfig:
    lam: float                         # full-game regulation xG, home
    mu: float                          # away
    p_ot: float | None = None          # home OT/SO win prob; None -> lam/(lam+mu)
    pp_mult: float = 2.3               # 5v4 PP rate vs base
    sh_mult: float = 0.27              # SH rate vs base
    pp53_mult: float = 5.5             # 5v3 PP rate vs base (placeholder)
    st_home: float = 1.0               # PP/SH multipliers relative to lam / mu
    st_away: float = 1.0
    endgame: m.EndgameParams | None = field(default_factory=m.EndgameParams)
    dt: float = 1.0
    spreads: tuple = DEFAULT_SPREADS   # home alt lines
    totals: tuple | None = None        # None -> a ladder around the live total

    @property
    def p_ot_value(self) -> float:
        return self.p_ot if self.p_ot is not None else self.lam / (self.lam + self.mu)


_RATINGS = None


def st_ratings():
    global _RATINGS
    if _RATINGS is None:
        _RATINGS = pregame_st.load_st_ratings()
    return _RATINGS


def build_config(home: str, away: str, home_prob: float, total: float,
                 p_ot: float | None = None, use_st: bool = True,
                 league_gf: float = 3.05, **kw) -> tuple[PricerConfig, str]:
    """
    Config from a de-vigged home win prob (incl. OT) and expected total.
    Returns (config, note). With use_st the PP/SH rates are built off
    league_gf x mults x the teams' pregame special-teams factors.
    """
    lam, mu = fit_to_market(home_prob, total, p_ot, kw.get("endgame", m.EndgameParams()))
    note = f"λ={lam:.3f} μ={mu:.3f}"
    cfg = PricerConfig(lam, mu, p_ot, **kw)
    if use_st:
        try:
            r = st_ratings()
            fh, fa = r.factors(home, away)
            cfg.st_home, cfg.st_away = fh * league_gf / lam, fa * league_gf / mu
            note += f" | ST {home} ×{fh:.2f}, {away} ×{fa:.2f} (ratings {r.as_of})"
        except Exception as e:          # unknown team, missing files
            note += f" | ST ratings off: {e}"
    return cfg, note


def fit_to_market(home_prob: float, total: float, p_ot: float | None,
                  endgame: m.EndgameParams | None, iters: int = 8) -> tuple[float, float]:
    """
    (lam, mu) such that the live model's puck-drop win prob incl. OT and
    expected final total (incl. pulled-goalie goals and the OT/SO goal)
    match the market. lam + mu ends up below the market total.
    """
    eg = endgame or NO_PULLS
    k = np.arange(TOTAL_GRID + 1)
    tot_grid = k[:, None] + k[None, :] + (k[:, None] == k[None, :])   # +1 OT goal on ties
    base_total, prob = total, home_prob
    for _ in range(iters):
        lam, mu = m.fit_lambdas(prob, base_total, p_ot)
        po = p_ot if p_ot is not None else lam / (lam + mu)
        reg = m.endgame_margin_dist(lam, mu, 1.0, 0, eg)
        wp = sum(v for d, v in reg.items() if d > 0) + reg.get(0, 0.0) * po
        exp_total = float((m.endgame_goals_dist(lam, mu, 1.0, 0, eg) * tot_grid).sum())
        if abs(wp - home_prob) < 1e-4 and abs(exp_total - total) < 1e-3:
            break
        base_total *= total / exp_total
        prob = min(max(prob + home_prob - wp, 0.03), 0.97)
    return lam, mu


def pregame_line(home: str, away: str, day: date | None = None) -> tuple[float, float, str]:
    """
    (home win prob incl. OT, expected total, row date) from the pregame model's
    predictions_today_tomorrow.csv (last row for the matchup, i.e. the latest
    goalie update). The total is the Poisson mean that best fits the model's
    over 5.5 / 6.5 / 7.5 probabilities.
    """
    r = st_ratings()
    home_name, away_name = r.teams[r.index(home)], r.teams[r.index(away)]
    rows = []
    with open(PREDICTIONS_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["home_team"] == home_name and row["away_team"] == away_name:
                rows.append(row)
    if day is not None:
        rows = [x for x in rows if pregame_st._parse_date(x["date"]) == day]
    if not rows:
        raise LookupError(f"no pregame prediction for {away} @ {home}"
                          f"{' on ' + str(day) if day else ''} in {PREDICTIONS_CSV.name}")
    row = rows[-1]
    overs = [(5, float(row["prob_over_55"])), (6, float(row["prob_over_65"])),
             (7, float(row["prob_over_75"]))]
    fit = minimize_scalar(lambda lt: sum((poisson.sf(k, lt) - p) ** 2 for k, p in overs),
                          bounds=(3.0, 10.0), method="bounded")
    return float(row["home_win_prob"]), float(fit.x), row["date"]


# ── Prices ───────────────────────────────────────────────────────────────────

@dataclass
class Prices:
    game_id: str
    home: str
    away: str
    status: str
    clock: str
    elapsed: int
    home_score: int
    away_score: int
    situation: str
    home_win: float                    # win-equivalent incl. OT/SO
    away_win: float
    home_ml: float | None              # fair American odds
    away_ml: float | None
    reg_home: float
    reg_tie: float
    reg_away: float
    exp_total: float
    spreads: list[dict]
    totals: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def _fold_ot(reg: dict[int, float], p_ot: float) -> dict[int, float]:
    out: dict[int, float] = {}
    for k, v in reg.items():
        if k == 0:
            out[1] = out.get(1, 0.0) + v * p_ot
            out[-1] = out.get(-1, 0.0) + v * (1 - p_ot)
        else:
            out[k] = out.get(k, 0.0) + v
    return out


class LivePricer:
    def __init__(self, cfg: PricerConfig):
        self.cfg = cfg

    # -- model pieces ---------------------------------------------------------
    def _pp(self, st: GameState, T: float):
        """(pp_team, seconds, kind, effective PP mult) or None."""
        if not st.pp_team or st.pp_seconds <= 0 or T <= 0:
            return None
        c = self.cfg
        s = min(st.pp_seconds, T)
        two = min(st.pp_two_man_seconds, s)
        mult = (two * c.pp53_mult + (s - two) * c.pp_mult) / s
        return st.pp_team, s, st.pp_kind, mult

    def pulled(self, st: GameState) -> bool:
        """The trailing team (down 1 or 2) has its goalie out late in regulation."""
        if self.cfg.endgame is None or abs(st.d) not in (1, 2):
            return False
        if REG_SECONDS - st.elapsed > PULL_MAX_SECONDS or st.period != 3:
            return False
        return st.home_goalie_pulled if st.d < 0 else st.away_goalie_pulled

    def _model_args(self, st: GameState):
        """(t, endgame params, pulled, PP kwargs) for the library calls."""
        c = self.cfg
        T = max(REG_SECONDS - st.elapsed, 0)
        kw = {}
        pp = self._pp(st, T)
        if pp:
            team, s, kind, mult = pp
            kw = dict(pp_team=team, pp_seconds=s, penalty=kind, pp_mult=mult,
                      sh_mult=c.sh_mult, dt=c.dt, st_home=c.st_home, st_away=c.st_away)
        return T / REG_SECONDS, c.endgame or NO_PULLS, self.pulled(st), kw

    def _regulation(self, st: GameState) -> dict[int, float]:
        c = self.cfg
        t, eg, pulled, kw = self._model_args(st)
        return m.endgame_margin_dist(c.lam, c.mu, t, st.d, eg, pulled=pulled, **kw)

    def _total_dist(self, st: GameState) -> dict[int, float]:
        """Final total goals (incl. the OT/SO goal) -> prob."""
        c = self.cfg
        cur = st.home_score + st.away_score
        t, eg, pulled, kw = self._model_args(st)
        joint = m.endgame_goals_dist(c.lam, c.mu, t, st.d, eg, pulled=pulled,
                                     max_goals=TOTAL_GRID, **kw)
        out: dict[int, float] = {}
        for h, a in zip(*np.nonzero(joint > 1e-12)):
            total = cur + h + a
            if st.d + h - a == 0:      # OT/SO winner's goal counts in the final score
                total += 1
            out[int(total)] = out.get(int(total), 0.0) + float(joint[h, a])
        return out

    # -- public -----------------------------------------------------------------
    def price(self, st: GameState) -> Prices:
        c = self.cfg
        p_ot = c.p_ot_value
        cur = st.home_score + st.away_score

        if st.status == "FINAL":
            went_long = st.period_type in ("OT", "SO") or st.period >= 4
            reg = {0: 1.0} if went_long else {st.d: 1.0}
            final = {st.d: 1.0}
            totals = {cur: 1.0}
            win_equiv = 1.0 if st.d > 0 else 0.0
        elif st.period >= 4 or st.period_type in ("OT", "SO"):
            reg = {0: 1.0}
            final = {1: p_ot, -1: 1 - p_ot}
            totals = {cur + 1: 1.0}
            win_equiv = p_ot
        else:
            reg = self._regulation(st)
            final = _fold_ot(reg, p_ot)
            totals = self._total_dist(st)
            win_equiv = None

        r_home = sum(v for k, v in reg.items() if k > 0)
        r_tie = reg.get(0, 0.0)
        r_away = sum(v for k, v in reg.items() if k < 0)
        if win_equiv is None:
            win_equiv = r_home + r_tie * p_ot

        spreads = []
        for L in c.spreads:
            lp = m.price_line(final, L)
            spreads.append(dict(line=L, home=lp.home_cover, away=lp.away_cover, push=lp.push,
                                home_odds=american(lp.home_fair), away_odds=american(lp.away_fair)))

        exp_total = sum(k * v for k, v in totals.items())
        lines = c.totals or self._total_ladder(cur, exp_total)
        tot_rows = []
        for L in lines:
            over = sum(v for k, v in totals.items() if k > L)
            under = sum(v for k, v in totals.items() if k < L)
            push = 1 - over - under
            ou = over + under
            tot_rows.append(dict(line=L, over=over, under=under, push=push,
                                 over_odds=american(over / ou) if ou > 0 else None,
                                 under_odds=american(under / ou) if ou > 0 else None))

        return Prices(st.game_id, st.home, st.away, st.status, st.clock, st.elapsed,
                      st.home_score, st.away_score, st.situation,
                      win_equiv, 1 - win_equiv, american(win_equiv), american(1 - win_equiv),
                      r_home, r_tie, r_away, exp_total, spreads, tot_rows)

    @staticmethod
    def _total_ladder(cur: int, exp_total: float, n: int = 5) -> list[float]:
        mid = math.floor(exp_total) + 0.5
        lines = [mid + i for i in range(-(n // 2), n - n // 2)]
        lo = cur + 0.5
        if lines[0] < lo:
            lines = [lo + i for i in range(n)]
        return lines
