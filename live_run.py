"""
Headless live pricer: poll a feed, reprice every tick, push prices to sinks.

  python live_run.py --game BUF --price -140 --total 6.5
  python live_run.py --game 2026020152 --pregame          # line from the pregame model
  python live_run.py --replay 2026020148 --speed 60 --price -150 --total 6.3

Between feed updates the game clock is run forward locally (when the feed says
it's running), so prices keep moving every tick even though the NHL API is only
polled every few seconds.

Connecting outputs: a sink is any callable `sink(state, prices)`. Built in:
ConsoleSink (readable lines) and JsonlSink (one JSON object per tick). To push
prices somewhere else, e.g. your betting-edge service:

    from live_run import LiveRunner, ConsoleSink
    def send(state, prices):
        requests.post(MY_URL, json=prices.to_dict())
    LiveRunner(feed, pricer, sinks=[ConsoleSink(), send]).run()

Connecting inputs: any object with `poll() -> GameState` works as a feed (see
live_feed.py, incl. PushFeed for push-style providers).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import live_feed as lf
import live_pricer as lp


def _odds(x) -> str:
    return "  —  " if x is None else f"{x:+.0f}"


class ConsoleSink:
    """Prints on every event and otherwise every `every` real seconds."""

    def __init__(self, every: float = 15.0):
        self.every, self._last_t, self._last_key = every, 0.0, None

    def __call__(self, st: lf.GameState, p: lp.Prices):
        now = time.time()
        if st.key() == self._last_key and now - self._last_t < self.every:
            return
        self._last_t, self._last_key = now, st.key()
        spread = next((s for s in p.spreads if s["line"] == -1.5), None)
        main = min(p.totals, key=lambda r: abs(r["over"] - 0.5)) if p.totals else None
        parts = [f"{p.clock:>9}  {p.away} {p.away_score}-{p.home_score} {p.home}",
                 f"{p.situation:<26}",
                 f"ML {p.home} {p.home_win:6.1%} {_odds(p.home_ml)} / {p.away} {_odds(p.away_ml)}",
                 f"3way {p.reg_home:.2f}/{p.reg_tie:.2f}/{p.reg_away:.2f}"]
        if spread:
            parts.append(f"{p.home} -1.5 {_odds(spread['home_odds'])}")
        if main:
            parts.append(f"o/u {main['line']:g} {_odds(main['over_odds'])}/{_odds(main['under_odds'])}")
        print(" | ".join(parts), flush=True)


class JsonlSink:
    """Appends every priced tick as one JSON line."""

    def __init__(self, path: str):
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, st: lf.GameState, p: lp.Prices):
        row = p.to_dict()
        row["ts"] = time.time()
        self.fh.write(json.dumps(row) + "\n")
        self.fh.flush()


class LiveRunner:
    def __init__(self, feed, pricer: lp.LivePricer, sinks=(), poll: float = 5.0,
                 tick: float = 1.0, local_clock: bool = True):
        self.feed, self.pricer, self.sinks = feed, pricer, list(sinks)
        self.poll, self.tick, self.local_clock = poll, tick, local_clock
        self.base: lf.GameState | None = None
        self._fetched = 0.0

    def step(self) -> lf.GameState | None:
        """One tick: refresh from the feed if due, price, notify sinks."""
        now = time.time()
        if self.base is None or now - self._fetched >= self.poll:
            try:
                self.base, self._fetched = self.feed.poll(), now
            except Exception as e:      # network blip: keep pricing the last state
                print(f"[feed] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                if self.base is None:
                    return None
        st = self.base
        if self.local_clock:
            st = lf.advance_state(st, now - self._fetched)
        prices = self.pricer.price(st)
        for sink in self.sinks:
            try:
                sink(st, prices)
            except Exception as e:
                print(f"[sink] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return st

    def run(self):
        while True:
            st = self.step()
            if st is not None and st.status == "FINAL":
                return st
            time.sleep(self.tick)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--game", help="game id or team abbrev (today's game)")
    src.add_argument("--replay", help="game id of a finished game to replay")
    ap.add_argument("--speed", type=float, default=60, help="replay game-seconds per second")
    ap.add_argument("--start", type=float, default=0, help="replay start, game minute")
    ap.add_argument("--price", help="home price incl. OT: American odds, prob or percent")
    ap.add_argument("--total", type=float, help="expected total goals")
    ap.add_argument("--pregame", action="store_true", help="take price/total from the pregame model")
    ap.add_argument("--p-ot", type=float, help="home OT/SO win %% (default λ/(λ+μ))")
    ap.add_argument("--no-st", action="store_true", help="ignore pregame special-teams ratings")
    ap.add_argument("--league-gf", type=float, default=3.05)
    ap.add_argument("--pp-mult", type=float, default=2.3)
    ap.add_argument("--sh-mult", type=float, default=0.27)
    ap.add_argument("--pp53-mult", type=float, default=5.5)
    ap.add_argument("--no-endgame", action="store_true")
    ap.add_argument("--spreads", default=",".join(str(x) for x in lp.DEFAULT_SPREADS))
    ap.add_argument("--poll", type=float, default=5.0, help="feed poll seconds")
    ap.add_argument("--tick", type=float, default=1.0, help="reprice seconds")
    ap.add_argument("--every", type=float, default=15.0, help="console print seconds")
    ap.add_argument("--jsonl", help="append priced ticks to this file")
    a = ap.parse_args(argv)

    if a.replay:
        feed = lf.ReplayFeed(a.replay, speed=a.speed, start_elapsed=int(a.start * 60))
        poll, local = a.tick, False
    else:
        feed = lf.NHLApiFeed(lf.find_game_id(a.game))
        poll, local = a.poll, True
    st = feed.poll()
    print(f"Game {st.game_id}: {st.away} @ {st.home}  ({st.clock})")

    if a.pregame:
        prob, total, d = lp.pregame_line(st.home, st.away)
        print(f"Pregame model ({d}): {st.home} {prob:.3f}, total {total:.2f}")
    elif a.price and a.total:
        prob, total = lp.parse_price(a.price), a.total
    else:
        ap.error("give --price and --total, or --pregame")
    cfg, note = lp.build_config(
        st.home, st.away, prob, total,
        p_ot=None if a.p_ot is None else a.p_ot / 100, use_st=not a.no_st,
        league_gf=a.league_gf, pp_mult=a.pp_mult, sh_mult=a.sh_mult,
        pp53_mult=a.pp53_mult, endgame=None if a.no_endgame else lp.m.EndgameParams(),
        spreads=tuple(float(x) for x in a.spreads.split(",") if x.strip()))
    print(note)

    sinks = [ConsoleSink(a.every)]
    if a.jsonl:
        sinks.append(JsonlSink(a.jsonl))
    try:
        LiveRunner(feed, lp.LivePricer(cfg), sinks, poll=poll, tick=a.tick,
                   local_clock=local).run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
