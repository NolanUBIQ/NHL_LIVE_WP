"""
Download regular-season play-by-play (trimmed) to analysis/data/pbp_<season>.jsonl.

  python analysis/fetch_pbp.py            # 2024-25 and 2025-26
  python analysis/fetch_pbp.py 2026       # 2026-27 (resumes; skips games already saved)
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import live_feed as lf
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)
KEEP = ("typeDescKey", "timeInPeriod", "situationCode", "details", "periodDescriptor")

def fetch(gid):
    for attempt in range(4):
        try:
            g = lf.get_json(f"gamecenter/{gid}/play-by-play", timeout=20)
            break
        except Exception as e:
            if "404" in str(e): return None
            time.sleep(2 + attempt * 3)
    else:
        return {"id": gid, "error": True}
    if g.get("gameType") != 2 or g.get("gameState") not in ("OFF", "FINAL"):
        return None
    plays = [{k: p.get(k) for k in KEEP} for p in g.get("plays", [])]
    return {"id": gid, "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"],
            "home_id": g["homeTeam"]["id"], "away_id": g["awayTeam"]["id"],
            "home_score": g["homeTeam"]["score"], "away_score": g["awayTeam"]["score"],
            "date": g.get("gameDate"), "plays": plays}

SEASONS = [int(s) for s in sys.argv[1:]] or [2024, 2025]

for season in SEASONS:
    path = os.path.join(OUT, f"pbp_{season}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {json.loads(l)["id"] for l in f}
    ids = [int(f"{season}02{n:04d}") for n in range(1, 1313)]
    ids = [i for i in ids if i not in done]
    n_ok = 0
    with ThreadPoolExecutor(6) as ex, open(path, "a", encoding="utf-8") as f:
        for r in ex.map(fetch, ids):
            if r and not r.get("error"):
                f.write(json.dumps(r) + "\n"); n_ok += 1
            elif r:
                print("failed", r["id"], flush=True)
    print(season, "fetched", n_ok, "new games", flush=True)
