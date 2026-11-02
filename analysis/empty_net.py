"""
Empty-net / extra-attacker analysis from NHL play-by-play
(run analysis/fetch_pbp.py first). Fits the EndgameParams rates.

Pull episodes: in the 3rd period, a trailing team's goalie is out (situationCode
goalie digit 0). An episode starts at the first goalie-out event for the
trailing team at a given margin and ends at the next goal (either team) or the
end of regulation. Re-pulls after a d-zone faceoff stay in the same episode.
Pull time = midpoint between the last goalie-in event and the first
goalie-out event (the pull itself isn't logged).

Delayed-penalty pulls are dropped: the leading team is penalised and the
goalie is back in at the next event.

Rates: pulled exposure by margin and strength, from consecutive-event
intervals inside kept episodes (half weight when the state flips in between).
"""
import glob, json, os
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def secs(t):
    m, s = t.split(":")
    return int(m) * 60 + int(s)


def load():
    for path in sorted(glob.glob(os.path.join(HERE, "pbp_*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def code_info(code):
    """(away_out, home_out, away_skaters, home_skaters) or None."""
    if not code or len(code) != 4 or not code.isdigit():
        return None
    return code[0] == "0", code[3] == "0", int(code[1]), int(code[2])


episodes = []
expo = defaultdict(float)      # (margin, strength) -> pulled seconds
goals = Counter()              # (margin, strength, 'lead'/'trail') -> goals while pulled
n_games = games_with_en = dropped_delayed = 0
by_season = Counter(); en_by_season = Counter()

for g in load():
    n_games += 1
    season = str(g["id"])[:4]
    by_season[season] += 1
    hid = g["home_id"]
    h = a = 0
    had_en = False
    active = None
    prev = None                # (left, trailing goalie out?) for the previous P3 event
    for p in g["plays"]:
        per = (p.get("periodDescriptor") or {}).get("number")
        typ = p.get("typeDescKey")
        d = p.get("details") or {}
        info = code_info(p.get("situationCode"))
        if typ == "goal" and per in (1, 2, 3, 4) and info:
            scorer_home = d.get("eventOwnerTeamId") == hid
            if info[0] if scorer_home else info[1]:
                had_en = True
        if per != 3 or not p.get("timeInPeriod"):
            if typ == "goal" and per in (1, 2):
                h, a = d.get("homeScore", h), d.get("awayScore", a)
            continue
        left = 1200 - secs(p["timeInPeriod"])
        trailing = "home" if h < a else "away" if a < h else None
        margin = min(abs(h - a), 3)
        t_out, strength = None, None
        if info and trailing:
            away_out, home_out, as_, hs_ = info
            t_out = home_out if trailing == "home" else away_out
            t_sk, l_sk = (hs_, as_) if trailing == "home" else (as_, hs_)
            strength = "6v5" if (t_sk, l_sk) == (6, 5) else f"{t_sk}v{l_sk}"

        # delayed penalty: leading team penalised, goalie back at the next event
        if active and active.get("pen_pending") and typ != "penalty":
            if t_out is False:
                dropped_delayed += 1
                active = None
            else:
                active["pen_pending"] = False

        if active and prev is not None and t_out is not None and prev[1] is not None:
            w = (float(prev[1]) + float(t_out)) / 2
            if w > 0:
                active["expo"][(margin, strength)] += w * max(prev[0] - left, 0)

        if typ == "goal":
            scorer = "home" if d.get("eventOwnerTeamId") == hid else "away"
            if active:
                side = "trail" if scorer == active["team"] else "lead"
                if t_out:
                    active["goals"][(margin, strength, side)] += 1
                active.update(outcome=side, goal_left=left, en=bool(t_out) and side == "lead")
                episodes.append(active)
                active = None
            h, a = d.get("homeScore", h), d.get("awayScore", a)
            prev = (left, None)
            continue

        if (typ == "penalty" and active
                and d.get("eventOwnerTeamId") != (hid if active["team"] == "home" else g["away_id"])):
            active["pen_pending"] = True

        if (trailing and t_out and active is None and left >= 3
                and typ not in ("period-end", "game-end")):
            pull_left = (prev[0] + left) / 2 if prev is not None and prev[1] is False else left
            active = dict(season=season, team=trailing, margin=margin, pull_left=pull_left,
                          outcome="none", goal_left=None, en=False,
                          expo=defaultdict(float), goals=Counter())
        prev = (left, t_out if trailing else None)
    if active:
        episodes.append(active)
    if had_en:
        games_with_en += 1
        en_by_season[season] += 1

for e in episodes:
    for k, v in e["expo"].items():
        expo[k] += v
    goals.update(e["goals"])

print(f"games: {n_games}  {dict(by_season)}   delayed-penalty pulls dropped: {dropped_delayed}")
print(f"games with an empty-net goal (regulation + OT): {games_with_en / n_games:.1%}  "
      + "  ".join(f"{s}-{int(s[2:]) + 1}: {en_by_season[s] / by_season[s]:.1%}" for s in sorted(by_season)))

for margin in (1, 2):
    E = [e for e in episodes if e["margin"] == margin]
    pl = np.array([e["pull_left"] for e in E])
    out = Counter(e["outcome"] for e in E)
    en = sum(e["en"] for e in E)
    print(f"\nDown {margin}: {len(E)} pulls in the 3rd ({len(E) / n_games:.2f} per game)")
    print(f"  pull time left: mean {pl.mean():.0f}s  median {np.median(pl):.0f}s  sd {pl.std():.0f}s  "
          f"p10 {np.percentile(pl, 10):.0f}s  p90 {np.percentile(pl, 90):.0f}s")
    print(f"  next goal: leading team {out['lead'] / len(E):.1%} (into empty net {en / len(E):.1%})"
          f"   trailing team {out['trail'] / len(E):.1%}   none {out['none'] / len(E):.1%}")
    print("  by pull time -> next goal: leading / trailing / none")
    bins = [(0, 60), (60, 90), (90, 120), (120, 150), (150, 180), (180, 240), (240, 1200)]
    for lo, hi in bins:
        B = [e for e in E if lo < e["pull_left"] <= hi]
        if not B:
            continue
        o = Counter(e["outcome"] for e in B)
        mid = np.mean([e["pull_left"] for e in B])
        print(f"    {lo // 60}:{lo % 60:02d}-{hi // 60}:{hi % 60:02d} (avg {mid:4.0f}s, n={len(B):4d}): "
              f"{o['lead'] / len(B):5.1%} / {o['trail'] / len(B):5.1%} / {o['none'] / len(B):5.1%}")

print("\nGoals per 150 s of pulled time:")
for margin in (1, 2):
    for strength in sorted({s for (m, s) in expo if m == margin}, key=lambda s: -expo[(margin, s)]):
        T = expo[(margin, strength)]
        if T < 900:
            continue
        gl, gt = goals[(margin, strength, "lead")], goals[(margin, strength, "trail")]
        print(f"  down {margin}, {strength}: {T / 3600:5.1f} h   leading team (EN) {gl:4d} = "
              f"{gl / T * 150:.3f} ± {np.sqrt(gl) / T * 150:.3f}   trailing team {gt:4d} = "
              f"{gt / T * 150:.3f} ± {np.sqrt(gt) / T * 150:.3f}")

json.dump([{k: v for k, v in e.items() if k not in ("expo", "goals")} for e in episodes],
          open(os.path.join(HERE, "episodes.json"), "w"))
