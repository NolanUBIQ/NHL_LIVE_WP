"""
NHL live pricing dashboard.

Pick a game (today's slate from the NHL API) and connect, or replay a finished
game by id. The feed is polled on a background thread; between polls the game
clock runs locally so prices update every tick. Prices come from live_pricer
(moneyline incl. OT, regulation 3-way, alt puck lines, live totals), using the
pregame line you enter (or the pregame model's) and the pregame special-teams
ratings for the two teams.

The WP chart is backfilled from the play-by-play when you connect, and rebuilt
whenever you change a parameter. Hover it for the time / score / WP readout.

Run: python live_ui.py
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk

import numpy as np

import live_feed as lf
import live_pricer as lp
import nhl_ingame_skellam as m
from nhl_sim_ui import (ACCENT, AWAY_C, BAD, BG, FONT, HOME_C, MONO, MUTED,
                        PANEL, PANEL_ALT, WHITE, YELLOW, FigureCanvasTkAgg, Figure,
                        fmt_odds)

TICK_MS = 500
HISTORY_STEP = 30          # chart backfill spacing (game seconds)

PARAMS = [
    ("Pregame line", [
        ("price", "Home price (odds / prob)", "-110"),
        ("total", "Expected total", "6.0"),
        ("p_ot", "Home OT/SO win %", "50"),
    ]),
    ("Power play", [
        ("pp_mult", "5v4 PP rate mult", "2.30"),
        ("pp53_mult", "5v3 PP rate mult", "5.5"),
        ("sh_mult", "SH rate mult", "0.27"),
        ("league_gf", "League GF / team / 60", "3.05"),
    ]),
    ("Endgame (pulled goalie)", [
        ("eg_lead1", "Down 1: leader (EN) xG / 2:30", "0.85"),
        ("eg_trail1", "Down 1: trailer xG / 2:30", "0.55"),
        ("eg_lead1f", "Down 1, last 1:00: leader xG / 2:30", "0.85"),
        ("eg_trail1f", "Down 1, last 1:00: trailer xG / 2:30", "0.55"),
        ("eg_lead2", "Down 2: leader (EN) xG / 2:30", "0.67"),
        ("eg_trail2", "Down 2: trailer xG / 2:30", "0.55"),
        ("pull_at", "Assumed pull, down 1 (s left)", "120"),
        ("pull2_at", "Assumed pull, down 2 (s left)", "225"),
        ("roll", "Then every (s)", "30"),
    ]),
]


def local_time(utc: str) -> str:
    try:
        dt = datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%I:%M %p").lstrip("0")
    except ValueError:
        return ""


class LiveApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NHL Live WP — live pricer")
        self.geometry("1560x960")
        self.minsize(1300, 925)
        self.configure(bg=BG)
        self._style()

        self.vars: dict[str, tk.StringVar] = {}
        self.entries: dict[str, ttk.Entry] = {}
        self.last_good: dict[str, str] = {}
        self.p_ot_auto = tk.BooleanVar(value=True)
        self.endgame_on = tk.BooleanVar(value=True)
        self.st_on = tk.BooleanVar(value=True)
        self.game_choice = tk.StringVar(value="")
        self.replay_id = tk.StringVar(value="")
        self.replay_speed = tk.StringVar(value="30")
        self.poll_secs = tk.StringVar(value="5")
        self.ladder_var = tk.StringVar(value=", ".join(f"{x:+g}" for x in lp.DEFAULT_SPREADS))
        self.fit_info = tk.StringVar(value="")
        self.feed_info = tk.StringVar(value="Not connected")
        self.status = tk.StringVar(value="")
        self.slate_day = date.today()
        self.day_label = tk.StringVar(value="")

        self.games: list[dict] = []
        self.feed = None
        self.feed_gen = 0
        self.q: queue.Queue = queue.Queue()
        self.base: lf.GameState | None = None
        self.base_at = 0.0
        self.live_clock = False
        self.pricer: lp.LivePricer | None = None
        self.prices: lp.Prices | None = None
        self._cfg_key = None
        self._param_job = None
        self.reset_history()

        self._build()
        self.load_games()
        self.after(TICK_MS, self._tick)

    # -- styling / layout --------------------------------------------------------
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=WHITE, font=(FONT, 10))
        s.configure("Panel.TFrame", background=PANEL)
        s.configure("TLabel", background=PANEL, foreground=WHITE)
        s.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=(FONT, 9))
        s.configure("Head.TLabel", background=PANEL, foreground=YELLOW, font=(FONT, 10, "bold"))
        s.configure("Bg.TLabel", background=BG, foreground=MUTED)
        s.configure("TEntry", fieldbackground=PANEL_ALT, foreground=WHITE,
                    insertcolor=WHITE, bordercolor=ACCENT)
        s.configure("Bad.TEntry", fieldbackground=BAD, foreground=WHITE)
        s.configure("TButton", background=ACCENT, foreground=WHITE, borderwidth=0, padding=(8, 4))
        s.map("TButton", background=[("active", "#3d4c85"), ("disabled", PANEL_ALT)])
        s.configure("Play.TButton", background="#166534", font=(FONT, 10, "bold"))
        s.map("Play.TButton", background=[("active", "#15803d")])
        s.configure("TCheckbutton", background=PANEL, foreground=WHITE)
        s.map("TCheckbutton", background=[("active", PANEL)])
        s.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=WHITE,
                    rowheight=22, font=(MONO, 10), borderwidth=0)
        s.configure("Treeview.Heading", background=PANEL_ALT, foreground=MUTED,
                    font=(FONT, 9, "bold"), borderwidth=0)
        s.map("Treeview", background=[("selected", ACCENT)])

    def _build(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        left = ttk.Frame(self, style="Panel.TFrame", padding=12)
        left.grid(row=0, column=0, sticky="ns", padx=(10, 5), pady=10)
        self._build_left(left)
        center = ttk.Frame(self, style="Panel.TFrame", padding=12)
        center.grid(row=0, column=1, sticky="nsew", padx=5, pady=10)
        self._build_center(center)
        right = ttk.Frame(self, style="Panel.TFrame", padding=12)
        right.grid(row=0, column=2, sticky="ns", padx=(5, 10), pady=10)
        self._build_right(right)
        ttk.Label(self, textvariable=self.status, style="Bg.TLabel",
                  padding=(12, 0, 12, 6)).grid(row=1, column=0, columnspan=3, sticky="w")

    def _head(self, f, r, text):
        ttk.Label(f, text=text, style="Head.TLabel").grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(8, 2))
        return r + 1

    def _build_left(self, f):
        r = self._head(f, 0, "Game")
        day_row = tk.Frame(f, bg=PANEL)
        day_row.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Button(day_row, text="◀", width=3, command=lambda: self.shift_day(-1)).pack(side="left")
        tk.Label(day_row, textvariable=self.day_label, bg=PANEL, fg=WHITE,
                 font=(FONT, 10, "bold"), width=16).pack(side="left", padx=4)
        ttk.Button(day_row, text="▶", width=3, command=lambda: self.shift_day(1)).pack(side="left")
        ttk.Button(day_row, text="Today", command=lambda: self.shift_day(None)).pack(side="left", padx=(6, 0))
        r += 1
        self.game_box = ttk.Combobox(f, textvariable=self.game_choice, width=34, state="readonly")
        self.game_box.grid(row=r, column=0, columnspan=2, sticky="ew")
        r += 1
        row = tk.Frame(f, bg=PANEL)
        row.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(row, text="Connect live", style="Play.TButton",
                   command=self.connect_live).pack(side="left")
        ttk.Button(row, text="↻ Slate", command=self.load_games).pack(side="left", padx=4)
        ttk.Button(row, text="Stop", command=self.stop_feed).pack(side="left")
        r += 1
        for label, var in [("Poll seconds", self.poll_secs), ("Replay game ID", self.replay_id),
                           ("Replay speed (×)", self.replay_speed)]:
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=2)
            ttk.Entry(f, textvariable=var, width=12).grid(row=r, column=1, sticky="e")
            r += 1
        ttk.Button(f, text="Replay finished game", command=self.connect_replay).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        r += 1

        for title, specs in PARAMS:
            r = self._head(f, r, title)
            if title.startswith("Pregame"):
                ttk.Button(f, text="Load pregame model line", command=self.load_pregame).grid(
                    row=r, column=0, columnspan=2, sticky="ew", pady=(0, 2))
                r += 1
            if title.startswith("Endgame"):
                ttk.Checkbutton(f, text="Enabled", variable=self.endgame_on,
                                command=self.params_changed).grid(row=r, column=0, sticky="w")
                r += 1
            for key, label, default in specs:
                var = tk.StringVar(value=default)
                var.trace_add("write", lambda *_: self.params_changed())
                ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=1)
                e = ttk.Entry(f, textvariable=var, width=9)
                e.grid(row=r, column=1, sticky="e", padx=(10, 0))
                self.vars[key], self.entries[key], self.last_good[key] = var, e, default
                r += 1
                if key == "p_ot":
                    ttk.Checkbutton(f, text="Auto: λ / (λ + μ)", variable=self.p_ot_auto,
                                    command=self.params_changed).grid(row=r, column=0, sticky="w")
                    r += 1
            if title.startswith("Pregame"):
                ttk.Label(f, textvariable=self.fit_info, style="Muted.TLabel", justify="left",
                          wraplength=270).grid(row=r, column=0, columnspan=2, sticky="w")
                r += 1
            if title.startswith("Power"):
                ttk.Checkbutton(f, text="Use pregame team PP/PK ratings", variable=self.st_on,
                                command=self.params_changed).grid(
                    row=r, column=0, columnspan=2, sticky="w")
                r += 1

        r = self._head(f, r, "Puck-line ladder (home)")
        ttk.Entry(f, textvariable=self.ladder_var, width=30).grid(
            row=r, column=0, columnspan=2, sticky="ew")
        self.ladder_var.trace_add("write", lambda *_: self.params_changed())

    def _build_center(self, f):
        f.columnconfigure(0, weight=1)
        f.rowconfigure(3, weight=1)
        sb = tk.Frame(f, bg=PANEL)
        sb.grid(row=0, column=0, sticky="ew")
        for c in (0, 2):
            sb.columnconfigure(c, weight=1)
        self.lbl_away = tk.Label(sb, text="AWAY  –", bg=PANEL, fg=AWAY_C, font=(FONT, 30, "bold"))
        self.lbl_away.grid(row=0, column=0, sticky="e", padx=20)
        mid = tk.Frame(sb, bg=PANEL)
        mid.grid(row=0, column=1)
        self.lbl_clock = tk.Label(mid, text="—", bg=PANEL, fg=WHITE, font=(MONO, 22, "bold"))
        self.lbl_clock.pack()
        self.lbl_sit = tk.Label(mid, text="", bg=PANEL, fg=MUTED, font=(FONT, 10))
        self.lbl_sit.pack()
        self.lbl_home = tk.Label(sb, text="–  HOME", bg=PANEL, fg=HOME_C, font=(FONT, 30, "bold"))
        self.lbl_home.grid(row=0, column=2, sticky="w", padx=20)

        self.bar = tk.Canvas(f, height=34, bg=PANEL_ALT, highlightthickness=0)
        self.bar.grid(row=1, column=0, sticky="ew", pady=(12, 4))

        nums = tk.Frame(f, bg=PANEL)
        nums.grid(row=2, column=0, sticky="ew", pady=(4, 8))
        self.num_labels = {}
        cells = [("away", "Away ML (incl. OT)", AWAY_C), ("reg", "Regulation  H / T / A", WHITE),
                 ("total", "Expected final total", WHITE), ("home", "Home ML (incl. OT)", HOME_C)]
        for i, (key, title, color) in enumerate(cells):
            nums.columnconfigure(i, weight=1)
            cell = tk.Frame(nums, bg=PANEL)
            cell.grid(row=0, column=i, sticky="ew")
            tk.Label(cell, text=title, bg=PANEL, fg=MUTED, font=(FONT, 9)).pack()
            big = tk.Label(cell, text="—", bg=PANEL, fg=color, font=(FONT, 20, "bold"))
            big.pack()
            small = tk.Label(cell, text="", bg=PANEL, fg=MUTED, font=(MONO, 10))
            small.pack()
            self.num_labels[key] = (big, small)

        self.fig = Figure(figsize=(8, 4), dpi=100, facecolor=PANEL)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.12)
        self.canvas = FigureCanvasTkAgg(self.fig, master=f)
        self.canvas.get_tk_widget().grid(row=3, column=0, sticky="nsew")
        self._cursor_x = None
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("figure_leave_event", self._on_leave)

        ttk.Label(f, textvariable=self.feed_info, style="Muted.TLabel").grid(
            row=4, column=0, sticky="w", pady=(8, 0))

    def _tree(self, f, row, cols):
        tree = ttk.Treeview(f, columns=[c[0] for c in cols], show="headings", height=7)
        for key, title, w in cols:
            tree.heading(key, text=title)
            tree.column(key, width=w, anchor="e")
        tree.grid(row=row, column=0, sticky="ew", pady=(4, 10))
        return tree

    def _build_right(self, f):
        f.rowconfigure(5, weight=1)
        ttk.Label(f, text="Alt puck lines (fair, incl. OT/SO)", style="Head.TLabel").grid(
            row=0, column=0, sticky="w")
        self.tree_lines = self._tree(f, 1, [("line", "Home", 55), ("p", "P(cover)", 70),
                                            ("h", "Home", 70), ("a", "Away", 70), ("push", "Push", 55)])
        ttk.Label(f, text="Totals (fair, final score incl. OT/SO goal)", style="Head.TLabel").grid(
            row=2, column=0, sticky="w")
        self.tree_tot = self._tree(f, 3, [("line", "Line", 55), ("p", "P(over)", 70),
                                          ("o", "Over", 70), ("u", "Under", 70), ("push", "Push", 55)])
        ttk.Label(f, text="Event log", style="Head.TLabel").grid(row=4, column=0, sticky="w")
        self.log_box = tk.Listbox(f, bg=PANEL_ALT, fg=WHITE, font=(MONO, 9), borderwidth=0,
                                  highlightthickness=0, width=44, selectbackground=ACCENT,
                                  activestyle="none")
        self.log_box.grid(row=5, column=0, sticky="nsew", pady=(4, 0))

    # -- games / feeds -------------------------------------------------------------
    def shift_day(self, step: int | None):
        self.slate_day = date.today() if step is None else self.slate_day + timedelta(days=step)
        self.game_choice.set("")
        self.load_games()

    def load_games(self, select_id: str | None = None):
        day = self.slate_day
        rel = {0: "Today", -1: "Yesterday", 1: "Tomorrow"}.get((day - date.today()).days)
        self.day_label.set(rel + f" {day:%m/%d}" if rel else f"{day:%a %m/%d}")

        def work():
            try:
                self.q.put(("games", (day, lf.find_games(day.isoformat()), select_id), None))
            except Exception as e:
                self.q.put(("games", None, e))
        threading.Thread(target=work, daemon=True).start()

    def _set_games(self, day, games, select_id=None):
        if day != self.slate_day:            # a slower, older request
            return
        self.games = games
        labels = [f"{g['away']} @ {g['home']}   {local_time(g['start'])}   {g['state']}   [{g['id']}]"
                  for g in games]
        self.game_box.configure(values=labels)
        ids = [g["id"] for g in games]
        if select_id in ids:
            self.game_box.current(ids.index(select_id))
        elif labels and not self.game_choice.get():
            live = [i for i, g in enumerate(games) if g["state"] in ("LIVE", "CRIT")]
            self.game_box.current(live[0] if live else 0)
        self.status.set(f"{len(games)} games on {day:%a %m/%d}. Finished games: 'Connect' shows the "
                        f"full game, 'Replay' plays it back.")

    def _selected_id(self) -> str | None:
        s = self.game_choice.get()
        return s[s.rfind("[") + 1:s.rfind("]")] if "[" in s else None

    def connect_live(self):
        gid = self._selected_id()
        if not gid:
            self.status.set("Pick a game first (↻ Slate reloads today's games)")
            return
        self._start_feed(lf.NHLApiFeed(gid), live_clock=True)

    def connect_replay(self):
        gid = self.replay_id.get().strip() or self._selected_id()
        try:
            speed = float(self.replay_speed.get())
        except ValueError:
            speed = 30.0
        self.status.set(f"Loading play-by-play for {gid}…")

        def work():
            try:
                self.q.put(("replay", lf.ReplayFeed(gid, speed=speed), None))
            except Exception as e:
                self.q.put(("replay", None, e))
        threading.Thread(target=work, daemon=True).start()

    def stop_feed(self):
        self.feed_gen += 1
        self.feed = None
        self.feed_info.set("Stopped")

    def _start_feed(self, feed, live_clock: bool):
        self.feed_gen += 1
        gen = self.feed_gen
        self.feed, self.live_clock = feed, live_clock
        self.base, self.pricer, self.prices, self._cfg_key = None, None, None, None
        self.reset_history()
        self.log_box.delete(0, "end")
        try:
            poll = max(float(self.poll_secs.get()), 1.0) if live_clock else 0.5
        except ValueError:
            poll = 5.0

        def loop():
            while self.feed_gen == gen:
                try:
                    self.q.put(("state", (gen, feed.poll()), None))
                except Exception as e:
                    self.q.put(("state", (gen, None), e))
                time.sleep(poll)
        threading.Thread(target=loop, daemon=True).start()
        self.feed_info.set(f"Connecting to {feed.game_id}…")

    # -- parameters ----------------------------------------------------------------
    def _num(self, key, positive=False):
        try:
            v = float(self.vars[key].get())
            ok = np.isfinite(v) and v >= 0 and (v > 0 or not positive)
        except ValueError:
            ok = False
        self.entries[key].configure(style="TEntry" if ok else "Bad.TEntry")
        if ok:
            self.last_good[key] = self.vars[key].get()
        return float(self.last_good[key])

    def params_changed(self):
        if self._param_job:
            self.after_cancel(self._param_job)
        self._param_job = self.after(250, self._apply_params)

    def _apply_params(self):
        self._param_job = None
        self._cfg_key = None          # force a rebuild
        if self.base:
            self.build_pricer(self.base)
            self.rebuild_history()

    def build_pricer(self, st: lf.GameState):
        try:
            prob = lp.parse_price(self.vars["price"].get())
            self.entries["price"].configure(style="TEntry")
        except ValueError:
            self.entries["price"].configure(style="Bad.TEntry")
            prob = lp.parse_price(self.last_good["price"])
        else:
            self.last_good["price"] = self.vars["price"].get()
        total = self._num("total", positive=True)
        p_ot = None if self.p_ot_auto.get() else min(self._num("p_ot") / 100, 1.0)
        self.entries["p_ot"].state(["disabled"] if self.p_ot_auto.get() else ["!disabled"])
        eg = None
        if self.endgame_on.get():
            lead1, trail1 = self._num("eg_lead1"), self._num("eg_trail1")
            eg = m.EndgameParams(lead1, trail1, trail1, lead1,
                                 pull_seconds=self._num("pull_at"),
                                 roll_seconds=self._num("roll", positive=True),
                                 e2_lead=self._num("eg_lead2"), e2_trail=self._num("eg_trail2"),
                                 pull2_seconds=self._num("pull2_at"),
                                 final_lead=self._num("eg_lead1f"),
                                 final_trail=self._num("eg_trail1f"))
        try:
            ladder = tuple(float(x) for x in self.ladder_var.get().replace(";", ",").split(",")
                           if x.strip())
        except ValueError:
            ladder = lp.DEFAULT_SPREADS
        try:
            cfg, note = lp.build_config(
                st.home, st.away, prob, total, p_ot=p_ot, use_st=self.st_on.get(),
                league_gf=self._num("league_gf", positive=True),
                pp_mult=self._num("pp_mult"), sh_mult=self._num("sh_mult"),
                pp53_mult=self._num("pp53_mult"), endgame=eg, spreads=ladder or lp.DEFAULT_SPREADS)
        except Exception as e:
            self.fit_info.set(f"Fit failed: {e}")
            return
        self.pricer = lp.LivePricer(cfg)
        self.fit_info.set(note.replace(" | ", "\n"))
        self._cfg_key = (st.game_id,)

    def load_pregame(self):
        if not self.base:
            self.status.set("Connect to a game first, so the teams are known")
            return
        try:
            prob, total, d = lp.pregame_line(self.base.home, self.base.away)
        except Exception as e:
            self.status.set(f"Pregame line unavailable: {e}")
            return
        self.vars["price"].set(f"{prob:.4f}")
        self.vars["total"].set(f"{total:.2f}")
        self.status.set(f"Pregame model line ({d}): {self.base.home} {prob:.1%}, total {total:.2f}")

    # -- history (chart) -------------------------------------------------------------
    def reset_history(self):
        self.path_t: list[float] = []
        self.path_wp: list[float] = []
        self.path_sit: list[str] = []
        self._last_key = None

    def _record(self, st: lf.GameState, we: float):
        self.path_t.append(st.elapsed / 60)
        self.path_wp.append(we)
        self.path_sit.append(st.situation)

    def rebuild_history(self):
        """Re-price the game so far from the play-by-play (connect / param change)."""
        if not (self.pricer and self.base and self.feed is not None
                and getattr(self.feed, "pbp", None) and hasattr(self.feed, "state_at")):
            return
        now = self.current_state().elapsed
        times = set(range(0, now, HISTORY_STEP))
        for e in self.base.events:
            if e.kind in ("goal", "penalty") and e.elapsed <= now:
                times.update((max(e.elapsed - 1, 0), e.elapsed))
        self.reset_history()
        for t in sorted(times):
            st = self.feed.state_at(t)
            if st.status == "FINAL":
                break
            self._record(st, self.pricer.price(st).home_win)

    # -- main loop ---------------------------------------------------------------------
    def current_state(self) -> lf.GameState:
        if self.live_clock:
            return lf.advance_state(self.base, time.time() - self.base_at)
        return self.base

    def _tick(self):
        try:
            self._drain()
            if self.base and self.pricer:
                st = self.current_state()
                prev = self.prices
                self.prices = self.pricer.price(st)
                if st.key() != self._last_key and prev and self.path_t:
                    # event: hold the old WP up to now so the chart shows a step
                    self._record(st, prev.home_win)
                self._last_key = st.key()
                if (not self.path_t or st.elapsed / 60 > self.path_t[-1]
                        or self.prices.home_win != self.path_wp[-1]):
                    self._record(st, self.prices.home_win)
                self.render(st)
        finally:
            self.after(TICK_MS, self._tick)

    def _drain(self):
        latest = None
        while True:
            try:
                kind, payload, err = self.q.get_nowait()
            except queue.Empty:
                break
            if kind == "games":
                if err:
                    self.status.set(f"Couldn't load games: {err}")
                else:
                    self._set_games(*payload)
            elif kind == "replay":
                if err:
                    self.status.set(f"Replay failed: {err}")
                else:
                    self._start_feed(payload, live_clock=False)
            elif kind == "state":
                gen, st = payload
                if gen != self.feed_gen:
                    continue
                if err:
                    self.feed_info.set(f"Feed error {datetime.now():%H:%M:%S}: {err}  (pricing last state)")
                else:
                    latest = st
        if latest is None:
            return
        first = self.base is None
        self.base, self.base_at = latest, latest.fetched_at or time.time()
        if first:
            self.build_pricer(latest)
            self.rebuild_history()
        src = "replay" if isinstance(self.feed, lf.ReplayFeed) else "NHL API"
        self.feed_info.set(f"{src} · game {latest.game_id} · updated {datetime.now():%H:%M:%S}"
                           f" · {latest.status}{'' if latest.clock_running or not self.live_clock else ' (clock stopped)'}")
        if len(latest.events) != self.log_box.size():
            self.log_box.delete(0, "end")
            for e in latest.events:
                self.log_box.insert("end", e.text)
            self.log_box.see("end")
        if latest.status == "FINAL" and isinstance(self.feed, lf.ReplayFeed):
            self.feed_gen += 1          # replay done: stop polling

    # -- drawing -----------------------------------------------------------------------
    def render(self, st: lf.GameState):
        p = self.prices
        self.lbl_away.config(text=f"{st.away}  {st.away_score}")
        self.lbl_home.config(text=f"{st.home_score}  {st.home}")
        self.lbl_clock.config(text=st.clock)
        sit = st.situation
        color = MUTED
        if st.pp_team:
            color = HOME_C if st.pp_team == "home" else AWAY_C
        elif "pulled" in sit:
            color = YELLOW
        self.lbl_sit.config(text=sit, fg=color)

        we = p.home_win
        self._num_set("home", f"{we:.1%}", fmt_odds(we))
        self._num_set("away", f"{1 - we:.1%}", fmt_odds(1 - we))
        self._num_set("reg", f"{p.reg_home:.0%} / {p.reg_tie:.0%} / {p.reg_away:.0%}",
                      f"{fmt_odds(p.reg_home)} / {fmt_odds(p.reg_tie)} / {fmt_odds(p.reg_away)}")
        self._num_set("total", f"{p.exp_total:.2f}", f"now {st.home_score + st.away_score}")
        self._draw_bar(we, st)

        self.tree_lines.delete(*self.tree_lines.get_children())
        for s in p.spreads:
            self.tree_lines.insert("", "end", values=(
                f"{s['line']:+.1f}", f"{s['home']:.3f}", fmt_odds(s["home"] / max(s["home"] + s["away"], 1e-12)),
                fmt_odds(s["away"] / max(s["home"] + s["away"], 1e-12)),
                f"{s['push']:.3f}" if s["push"] > 1e-9 else ""))
        self.tree_tot.delete(*self.tree_tot.get_children())
        for t in p.totals:
            ou = max(t["over"] + t["under"], 1e-12)
            self.tree_tot.insert("", "end", values=(
                f"{t['line']:g}", f"{t['over']:.3f}", fmt_odds(t["over"] / ou),
                fmt_odds(t["under"] / ou), f"{t['push']:.3f}" if t["push"] > 1e-9 else ""))
        self._draw_chart(st)

    def _num_set(self, key, big, small):
        b, s = self.num_labels[key]
        b.config(text=big)
        s.config(text=small)

    def _draw_bar(self, we, st):
        c = self.bar
        c.delete("all")
        w = max(c.winfo_width(), 10)
        h = int(c["height"])
        split = w * (1 - we)
        c.create_rectangle(0, 0, split, h, fill=AWAY_C, width=0)
        c.create_rectangle(split, 0, w, h, fill=HOME_C, width=0)
        c.create_line(w / 2, 0, w / 2, h, fill=BG, dash=(3, 3))
        if 1 - we > 0.08:
            c.create_text(8, h / 2, text=f"{st.away} {1 - we:.1%}", anchor="w",
                          fill=BG, font=(FONT, 11, "bold"))
        if we > 0.08:
            c.create_text(w - 8, h / 2, text=f"{st.home} {we:.1%}", anchor="e",
                          fill=BG, font=(FONT, 11, "bold"))

    def _xmax(self, st) -> float:
        if st.period <= 3 and st.period_type == "REG":
            return 60
        if st.playoff:
            return 60 + 20 * (st.period - 3)
        return 65

    def _draw_chart(self, st: lf.GameState):
        ax = self.ax
        ax.clear()
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color(PANEL_ALT)
        ax.tick_params(colors=MUTED, labelsize=8)
        xmax = self._xmax(st)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1)
        ax.set_xticks([x for x in range(0, int(xmax) + 1, 10)])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Game minute", color=MUTED, fontsize=8)
        ax.axhline(0.5, color=MUTED, lw=0.6, ls=":")
        for x in (20, 40, 60):
            if x < xmax:
                ax.axvline(x, color=PANEL_ALT, lw=1)
        # PP shading from the recorded situations
        start = team = None
        for t, sit in zip(self.path_t + [None], self.path_sit + [""]):
            now = None
            if sit and " PP" in sit:
                now = "home" if sit.startswith(st.home + " ") else "away"
            if now != team:
                if team and start is not None and t is not None:
                    ax.axvspan(start, t, color=HOME_C if team == "home" else AWAY_C, alpha=0.10, lw=0)
                start, team = t, now
        if self.path_t:
            t, wp = np.array(self.path_t), np.array(self.path_wp)
            ax.plot(t, wp, color=WHITE, lw=1.6)
            ax.fill_between(t, 0.5, wp, where=wp >= 0.5, color=HOME_C, alpha=0.25, interpolate=True, lw=0)
            ax.fill_between(t, 0.5, wp, where=wp < 0.5, color=AWAY_C, alpha=0.25, interpolate=True, lw=0)
            for e in self.base.events:
                if e.kind == "goal" and e.elapsed / 60 <= t[-1]:
                    y = self.wp_at(e.elapsed / 60)
                    ax.scatter([e.elapsed / 60], [y], s=36, zorder=5,
                               color=HOME_C if e.team == "home" else AWAY_C,
                               edgecolors=WHITE, linewidths=0.8)
        self._cur_line = ax.axvline(0, color=YELLOW, lw=0.9, zorder=6, visible=False)
        self._cur_dot, = ax.plot([], [], "o", color=YELLOW, ms=6, zorder=7,
                                 markeredgecolor=BG, visible=False)
        self._cur_text = ax.text(0, 0.97, "", transform=ax.get_xaxis_transform(), va="top",
                                 color=WHITE, fontsize=9, family=MONO, zorder=8, visible=False,
                                 bbox=dict(boxstyle="round,pad=0.4", fc=PANEL_ALT, ec=ACCENT, lw=1))
        self._update_cursor()
        self.canvas.draw_idle()

    # -- hover cursor ------------------------------------------------------------------
    def _idx_at(self, x: float) -> int:
        return int(np.searchsorted(self.path_t, x, side="right")) - 1

    def wp_at(self, x: float) -> float:
        t, wp = self.path_t, self.path_wp
        i = self._idx_at(x)
        if i < 0:
            return wp[0]
        if i >= len(t) - 1:
            return wp[-1]
        span = t[i + 1] - t[i]
        return wp[i] + (wp[i + 1] - wp[i]) * (x - t[i]) / span if span > 0 else wp[i]

    def _update_cursor(self):
        artists = (self._cur_line, self._cur_dot, self._cur_text)
        if self._cursor_x is None or not self.path_t or not self.base:
            for a in artists:
                a.set_visible(False)
            return
        st = self.base
        x = min(max(self._cursor_x, self.path_t[0]), self.path_t[-1])
        wp = self.wp_at(x)
        sec = int(round(x * 60))
        if sec >= lf.REG_SECONDS and st.period_type != "REG":
            clock = "OT" if sec < 3900 or st.playoff else "SO"
        else:
            per = min(sec // 1200 + 1, 3)
            clock = f"P{per} {lf.fmt_clock(per * 1200 - sec)}"
        hg = sum(1 for e in st.events if e.kind == "goal" and e.team == "home" and e.elapsed <= sec)
        ag = sum(1 for e in st.events if e.kind == "goal" and e.team == "away" and e.elapsed <= sec)
        sit = self.path_sit[max(self._idx_at(x), 0)]
        extra = f"\n{sit}" if sit and sit != "Even strength" else ""
        self._cur_line.set_xdata([x, x])
        self._cur_dot.set_data([x], [wp])
        self._cur_text.set_text(f"{clock}   {st.away} {ag}-{hg} {st.home}\n"
                                f"{st.home} {wp:.1%}  ·  {st.away} {1 - wp:.1%}{extra}")
        right = x > self.ax.get_xlim()[1] * 0.7
        self._cur_text.set_x(x - 0.6 if right else x + 0.6)
        self._cur_text.set_ha("right" if right else "left")
        for a in artists:
            a.set_visible(True)

    def _on_motion(self, event):
        new = event.xdata if event.inaxes is self.ax else None
        if new is None and self._cursor_x is None:
            return
        self._cursor_x = new
        if self.path_t:
            self._update_cursor()
            self.canvas.draw_idle()

    def _on_leave(self, _event):
        if self._cursor_x is not None:
            self._cursor_x = None
            if self.path_t:
                self._update_cursor()
                self.canvas.draw_idle()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="NHL live pricing dashboard")
    ap.add_argument("--date", help="slate date YYYY-MM-DD (default today)")
    ap.add_argument("--game", help="game id to connect to on launch")
    ap.add_argument("--replay", action="store_true", help="replay --game instead of connecting")
    ap.add_argument("--speed", default="30", help="replay speed (game seconds per second)")
    ap.add_argument("--price", help="home price incl. OT")
    ap.add_argument("--total", help="expected total")
    a = ap.parse_args(argv)

    app = LiveApp()
    if a.date:
        app.slate_day = datetime.strptime(a.date, "%Y-%m-%d").date()
        app.load_games(select_id=a.game)
    if a.price:
        app.vars["price"].set(a.price)
    if a.total:
        app.vars["total"].set(a.total)
    if a.game:
        if a.replay:
            app.replay_id.set(a.game)
            app.replay_speed.set(a.speed)
            app.after(300, app.connect_replay)
        else:
            app.after(300, lambda: app._start_feed(lf.NHLApiFeed(a.game), live_clock=True))
    app.mainloop()


if __name__ == "__main__":
    main()
