# NHL Live WP

In-game NHL win probability and fair live prices from a Skellam goal-difference
model with power-play and pulled-goalie endgame adjustments.

| File | What it does |
|---|---|
| `nhl_ingame_skellam.py` | Core model: win / tie / loss, win-equivalent, power plays, endgame, alt puck lines, market fit |
| `pregame_st.py` | Loads special-teams ratings from the pregame model and turns a matchup into PP/SH rate multipliers |
| `nhl_sim_ui.py` | Tkinter simulator: play out a simulated game second by second and watch the prices move |
| `live_feed.py` | Live game state: NHL API feed, replay feed for finished games, push feed for other providers |
| `live_pricer.py` | Game state to fair prices: moneyline incl. OT, regulation 3-way, alt puck lines, live totals |
| `live_run.py` | Headless live pricer with pluggable outputs (console, JSONL, your own callback) |
| `live_ui.py` | Live pricing dashboard |

## Run

```
python live_ui.py
python live_ui.py --date 2026-10-20 --game 2026020148 --price -150 --total 6.3
python live_run.py --game BUF --price -140 --total 6.5
python live_run.py --replay 2026020148 --speed 60 --price -150 --total 6.3
python nhl_sim_ui.py
```

Requires numpy, scipy and matplotlib (tkinter ships with Python).

## Connecting other feeds and outputs

- **Feed:** any object with `poll() -> GameState` works; for push-style
  providers, call `PushFeed.push(state)` from your callback.
- **Output:** any callable `sink(state, prices)` can be passed to `LiveRunner`;
  `prices.to_dict()` is JSON-ready.

## Pregame inputs

The pregame line is a home win probability incl. OT and an expected total
(from the market, or from the pregame model's predictions file). Special-teams
ratings are read from the pregame model's folder (`PREGAME_DIR` in
`pregame_st.py`); without it, PP/SH rates fall back to league-level multipliers.
