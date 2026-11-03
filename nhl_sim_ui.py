"""
NHL live win-probability simulator UI.

Tweak the Skellam model's parameters (team xG, OT share, power-play multipliers,
pulled-goalie endgame), then play out a simulated game second by second and
watch the model's win / tie / loss, win-equivalent and alt puck lines move.

The simulated game uses the same assumptions as the model:
  - goals arrive as Poisson processes at lam/3600 and mu/3600 per second
  - penalties arrive at a set rate per team; during a PP the PP team scores at
    base * pp_mult and the SH team at base * sh_mult; a PP goal ends a minor
  - with the endgame on, a team down one or two pulls its goalie at its own
    random pull time (or by hand); while pulled, scoring switches to the
    endgame rates (fitted from NHL play-by-play, see analysis/empty_net.py;
    down 1 has its own rates for the last minute) until a goal changes the
    margin. The model doesn't know the pull time:
    down 1 it assumes 2:00 left, then 1:30, 1:00, ... as each passes without
    a pull (down 2: 3:45, 3:15, ...), and prices a pulled goalie as soon as
    it happens
  - a regulation tie goes to OT/SO, which home wins with probability p_ot
  - optionally, PP/SH scoring rates come from the pregame model's special-teams
    ratings for the chosen teams (see pregame_st.py)
You can also inject goals and penalties by hand at the current game time.

Run: python nhl_sim_ui.py
Requires: numpy, scipy, matplotlib (tkinter ships with Python)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import nhl_ingame_skellam as m
import pregame_st

REG = m.REG_SECONDS
FRAME_MS = 60
SAMPLE_S = 30          # chart WP sample spacing (game seconds) between events

# ── Palette ──────────────────────────────────────────────────────────────────
BG = "#131320"
PANEL = "#1c1c30"
PANEL_ALT = "#22223a"
ACCENT = "#2e3a66"
HOME_C = "#60a5fa"
AWAY_C = "#f87171"
YELLOW = "#facc15"
GREEN = "#4ade80"
WHITE = "#f1f5f9"
MUTED = "#8b93a7"
BAD = "#7f1d1d"
FONT = "Segoe UI"
MONO = "Consolas"


# ── Parameters ───────────────────────────────────────────────────────────────

@dataclass
class Params:
    lam: float
    mu: float
    p_ot: float
    pp_mult: float
    sh_mult: float
    pen_rate: float          # minor/major penalties per team per 60
    major_pct: float         # share of penalties that are majors (0-1)
    dt: float                # model step for minor PPs (seconds)
    endgame: m.EndgameParams | None
    st_home: float = 1.0     # PP/SH scoring-rate multipliers (pregame ST ratings)
    st_away: float = 1.0
    sim_pull_mean: float = 120.0   # simulated pull time down 1, seconds left
    sim_pull_sd: float = 40.0
    sim_pull2_mean: float = 225.0  # simulated pull time down 2
    sim_pull2_sd: float = 70.0


# (key, label, default) grouped by section
SECTIONS = [
    ("Team strength (full-game xG)", [
        ("lam", "Home xG  λ", "3.20"),
        ("mu", "Away xG  μ", "2.80"),
    ]),
    ("Overtime / shootout", [
        ("p_ot", "Home OT/SO win %", "50"),
    ]),
    ("Power play", [
        ("pp_mult", "PP rate mult", "2.30"),
        ("sh_mult", "SH rate mult", "0.27"),
        ("pen_rate", "Penalties / team / 60", "3.0"),
        ("major_pct", "Major penalty %", "3"),
        ("dt", "Model PP step (s)", "1.0"),
    ]),
    ("Pregame special teams", [
        ("league_gf", "League GF / team / 60", "3.05"),
    ]),
    ("Endgame (pulled goalie)", [
        ("eg_lead1", "Down 1 xG / 2:30: leader (EN) / trailer", "0.85"),
        ("eg_trail1", None, "0.55"),
        ("eg_lead1f", "Down 1, last 1:00: leader / trailer", "0.85"),
        ("eg_trail1f", None, "0.55"),
        ("eg_lead2", "Down 2 xG / 2:30: leader / trailer", "0.67"),
        ("eg_trail2", None, "0.55"),
        ("pull_at", "Model: pull down 1 at (s left)", "120"),
        ("pull2_at", "Model: pull down 2 at (s left)", "225"),
        ("roll", "Model: then every (s)", "30"),
        ("sim_pull_mean", "Sim down 1: pull mean / sd", "120"),
        ("sim_pull_sd", None, "40"),
        ("sim_pull2_mean", "Sim down 2: pull mean / sd", "225"),
        ("sim_pull2_sd", None, "70"),
    ]),
]

DEFAULT_LADDER = "-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5"


def parse_price(s: str) -> float:
    """Home win prob from a probability (0.58), a percent (58) or American odds (-140)."""
    x = float(s.replace("−", "-").strip())
    if x <= -100 or x >= 100:
        return -x / (-x + 100) if x < 0 else 100 / (x + 100)
    if 0 < x < 1:
        return x
    if 1 <= x < 100:
        return x / 100
    raise ValueError(s)


def fmt_odds(p: float) -> str:
    if p <= 0:
        return "—"
    if p >= 1:
        return "lock"
    if p < 0.001:
        return "<0.1%"
    if p > 0.999:
        return ">99.9%"
    return m.fmt_american(m.prob_to_american(p))


def clock_str(elapsed: int) -> str:
    if elapsed >= REG:
        return "END 3rd"
    period = elapsed // 1200 + 1
    left = 1200 - elapsed % 1200
    return f"P{period}  {left // 60:02d}:{left % 60:02d}"


def short_clock(elapsed: int) -> str:
    period = min(elapsed // 1200 + 1, 3)
    left = 1200 * period - elapsed
    return f"P{period} {left // 60:d}:{left % 60:02d}"


# ── Model evaluation ─────────────────────────────────────────────────────────

def regulation_dist(p: Params, elapsed: int, d: int, pp_team, pp_left, pp_kind,
                    pulled: bool = False):
    t = max(REG - elapsed, 0) / REG
    kw = {}
    if pp_team and pp_left > 0:
        kw = dict(pp_team=pp_team, pp_seconds=pp_left, penalty=pp_kind,
                  pp_mult=p.pp_mult, sh_mult=p.sh_mult, dt=p.dt,
                  st_home=p.st_home, st_away=p.st_away)
    if p.endgame is not None:
        return m.endgame_margin_dist(p.lam, p.mu, t, d, p.endgame, pulled=pulled, **kw)
    return m.regulation_margin_dist(p.lam, p.mu, t, d, **kw)


def outcomes(reg: dict[int, float], p_ot: float) -> m.InGameProbs:
    win = sum(v for k, v in reg.items() if k > 0)
    tie = reg.get(0, 0.0)
    loss = sum(v for k, v in reg.items() if k < 0)
    return m.InGameProbs(win, tie, loss, win + tie * p_ot)


def final_from_reg(reg: dict[int, float], p_ot: float) -> dict[int, float]:
    out: dict[int, float] = {}
    for k, v in reg.items():
        if k == 0:
            out[1] = out.get(1, 0.0) + v * p_ot
            out[-1] = out.get(-1, 0.0) + v * (1 - p_ot)
        else:
            out[k] = out.get(k, 0.0) + v
    return out


# ── Game simulator ───────────────────────────────────────────────────────────

class GameSim:
    def __init__(self, seed: int | None):
        self.rng = np.random.default_rng(seed)
        self.elapsed = 0
        self.home = 0
        self.away = 0
        self.pp_team: str | None = None
        self.pp_left = 0
        self.pp_kind = "minor"
        self.final: str | None = None      # "home" / "away" once decided
        self.ot = False
        self.log: list[str] = []
        # goalie pulls: each team's pull time is mean + sd * z (own stream, so
        # a seed's goals/penalties don't depend on the pull settings)
        z = np.random.default_rng(None if seed is None else [seed, 7]).standard_normal(4)
        self.pull_z = {("home", 1): z[0], ("away", 1): z[1],
                       ("home", 2): z[2], ("away", 2): z[3]}
        self.pulled: str | None = None     # team whose goalie is out
        self.pull_spans: list[tuple[int, int, str]] = []
        self._pull_open = 0

    @property
    def d(self) -> int:
        return self.home - self.away

    def state(self):
        return (self.elapsed, self.d, self.pp_team, self.pp_left, self.pp_kind,
                self.pulled is not None)

    def pull_time(self, team: str, margin: int, p: Params) -> float:
        """Seconds left at which `team` pulls its goalie when down `margin` (1 or 2)."""
        mean, sd = ((p.sim_pull_mean, p.sim_pull_sd) if margin == 1
                    else (p.sim_pull2_mean, p.sim_pull2_sd))
        return float(np.clip(mean + sd * self.pull_z[(team, margin)], 10, 1200))

    def set_pulled(self, team: str | None, note: str = ""):
        if team == self.pulled:
            return
        if self.pulled:
            self.pull_spans.append((self._pull_open, self.elapsed, self.pulled))
        self.pulled = team
        if team:
            self._pull_open = self.elapsed
            self.log.append(f"{short_clock(self.elapsed)}  {team.upper()} goalie pulled{note}")

    def pull_goalie(self):
        """Trailing team pulls now (manual)."""
        if abs(self.d) in (1, 2):
            self.set_pulled("away" if self.d > 0 else "home", "  (manual)")

    def _update_pull(self, p: Params) -> bool:
        """Pull the trailing team's goalie once the clock reaches its pull time."""
        if p.endgame is None or abs(self.d) not in (1, 2):
            self.set_pulled(None)
            return False
        trailing = "away" if self.d > 0 else "home"
        if self.pulled or REG - self.elapsed > self.pull_time(trailing, abs(self.d), p):
            return False
        self.set_pulled(trailing)
        return True

    # -- events --------------------------------------------------------------
    def goal(self, team: str, note: str = ""):
        if team == "home":
            self.home += 1
        else:
            self.away += 1
        tag = note
        if self.pp_team:
            if team == self.pp_team:
                tag = tag or "PPG"
                if self.pp_kind == "minor":
                    self.pp_team, self.pp_left = None, 0
            else:
                tag = tag or "SHG"
        self.log.append(f"{short_clock(self.elapsed)}  {team.upper()} goal"
                        f"{'  (' + tag + ')' if tag else ''}   {self.away}-{self.home}")
        self.set_pulled(None)      # tie or two-goal game: goalie back in

    def penalty(self, on_team: str, kind: str = "minor"):
        """`on_team` is the team that gets the power play."""
        self.pp_team = on_team
        self.pp_kind = kind
        self.pp_left = 300 if kind == "major" else 120
        offender = "away" if on_team == "home" else "home"
        self.log.append(f"{short_clock(self.elapsed)}  {offender.upper()} {kind} "
                        f"-> {on_team.upper()} PP")

    def clear_pp(self):
        self.pp_team, self.pp_left = None, 0

    # -- one second of play -------------------------------------------------
    def step(self, p: Params) -> bool:
        """Advance one second. Returns True if an event happened."""
        if self.final:
            return False
        event = self._update_pull(p)
        lam_s, mu_s = p.lam / REG, p.mu / REG
        rh, ra = lam_s, mu_s
        if self.pp_team == "home":
            rh, ra = lam_s * p.pp_mult * p.st_home, mu_s * p.sh_mult * p.st_away
        elif self.pp_team == "away":
            rh, ra = lam_s * p.sh_mult * p.st_home, mu_s * p.pp_mult * p.st_away
        elif self.pulled:
            eg, w = p.endgame, p.endgame.window_seconds
            final = REG - self.elapsed <= eg.final_seconds
            rh, ra = m.pulled_rates(eg, final)[self.d]
            rh, ra = rh / w, ra / w
        rp = 0.0 if self.pp_team else p.pen_rate / REG   # per team

        rates = np.array([rh, ra, rp, rp])
        total = rates.sum()
        if total > 0 and self.rng.random() < 1 - np.exp(-total):
            kind = self.rng.choice(4, p=rates / total)
            event = True
            if kind == 0:
                self.goal("home", "EN" if self._en("home", p) else "")
            elif kind == 1:
                self.goal("away", "EN" if self._en("away", p) else "")
            else:
                major = self.rng.random() < p.major_pct
                # kind 2: away takes a penalty -> home PP; kind 3: the reverse
                self.penalty("home" if kind == 2 else "away",
                             "major" if major else "minor")

        self.elapsed += 1
        if self.pp_team:
            self.pp_left -= 1
            if self.pp_left <= 0:
                self.clear_pp()
        if self.elapsed >= REG:
            self._end_regulation(p)
            event = True
        return event

    def _en(self, team: str, p: Params) -> bool:
        """Goal into the empty net (leading team, goalie pulled, no PP)."""
        return (not self.pp_team and self.pulled is not None
                and self.pulled != team)

    def _end_regulation(self, p: Params):
        self.elapsed = REG
        self.clear_pp()
        self.set_pulled(None)
        if self.d != 0:
            self.final = "home" if self.d > 0 else "away"
            self.log.append(f"FINAL   {self.away}-{self.home}")
            return
        self.ot = True
        self.final = "home" if self.rng.random() < p.p_ot else "away"
        self.goal(self.final, "OT/SO")
        self.log.append(f"FINAL/OT   {self.away}-{self.home}")


# ── App ──────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NHL Live WP — Skellam simulator")
        self.geometry("1500x980")
        self.minsize(1200, 960)
        self.configure(bg=BG)
        self._style()

        self.vars: dict[str, tk.StringVar] = {}
        self.entries: dict[str, ttk.Entry] = {}
        self.last_good: dict[str, float] = {}
        self.p_ot_auto = tk.BooleanVar(value=False)
        self.endgame_on = tk.BooleanVar(value=True)
        self.speed = tk.DoubleVar(value=20)
        self.seed_var = tk.StringVar(value="")
        self.ladder_var = tk.StringVar(value=DEFAULT_LADDER)
        self.market_price = tk.StringVar(value="-140")
        self.market_total = tk.StringVar(value="6.0")
        self.status = tk.StringVar(value="")

        # pregame special-teams ratings
        self.st_on = tk.BooleanVar(value=True)
        self.st_info = tk.StringVar(value="")
        try:
            self.st_ratings = pregame_st.load_st_ratings()
            teams = sorted(self.st_ratings.abbrevs)
        except Exception as e:
            self.st_ratings, teams = None, []
            self.st_on.set(False)
            self.st_info.set(f"Ratings unavailable: {e}")
        self.home_team = tk.StringVar(value=teams[0] if teams else "")
        self.away_team = tk.StringVar(value=teams[1] if len(teams) > 1 else "")
        self.st_teams = teams

        self.running = False
        self._refresh_job = None

        self._build()
        self.new_game()

    # -- styling ---------------------------------------------------------------
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=WHITE, font=(FONT, 10))
        s.configure("Panel.TFrame", background=PANEL)
        s.configure("TLabel", background=PANEL, foreground=WHITE)
        s.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=(FONT, 9))
        s.configure("Head.TLabel", background=PANEL, foreground=YELLOW,
                    font=(FONT, 10, "bold"))
        s.configure("Bg.TLabel", background=BG, foreground=MUTED)
        s.configure("TEntry", fieldbackground=PANEL_ALT, foreground=WHITE,
                    insertcolor=WHITE, bordercolor=ACCENT)
        s.configure("Bad.TEntry", fieldbackground=BAD, foreground=WHITE)
        s.configure("TButton", background=ACCENT, foreground=WHITE, borderwidth=0,
                    padding=(8, 4))
        s.map("TButton", background=[("active", "#3d4c85"), ("disabled", PANEL_ALT)])
        s.configure("Play.TButton", background="#166534", font=(FONT, 10, "bold"))
        s.map("Play.TButton", background=[("active", "#15803d")])
        s.configure("TCheckbutton", background=PANEL, foreground=WHITE)
        s.map("TCheckbutton", background=[("active", PANEL)])
        s.configure("Horizontal.TScale", background=PANEL, troughcolor=PANEL_ALT)
        s.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                    foreground=WHITE, rowheight=24, font=(MONO, 10), borderwidth=0)
        s.configure("Treeview.Heading", background=PANEL_ALT, foreground=MUTED,
                    font=(FONT, 9, "bold"), borderwidth=0)
        s.map("Treeview", background=[("selected", ACCENT)])

    # -- layout ----------------------------------------------------------------
    def _build(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, style="Panel.TFrame", padding=12)
        left.grid(row=0, column=0, sticky="ns", padx=(10, 5), pady=10)
        self._build_params(left)

        center = ttk.Frame(self, style="Panel.TFrame", padding=12)
        center.grid(row=0, column=1, sticky="nsew", padx=5, pady=10)
        self._build_center(center)

        right = ttk.Frame(self, style="Panel.TFrame", padding=12)
        right.grid(row=0, column=2, sticky="ns", padx=(5, 10), pady=10)
        self._build_right(right)

        ttk.Label(self, textvariable=self.status, style="Bg.TLabel",
                  padding=(12, 0, 12, 6)).grid(row=1, column=0, columnspan=3, sticky="w")

    def _build_params(self, f):
        r = 0
        ttk.Label(f, text="Market fit", style="Head.TLabel").grid(
            row=r, column=0, columnspan=2, sticky="w")
        r += 1
        for label, var in [("Home price (odds / prob)", self.market_price),
                           ("Total goals", self.market_total)]:
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=2)
            ttk.Entry(f, textvariable=var, width=9).grid(row=r, column=1, sticky="e")
            r += 1
        ttk.Button(f, text="Fit λ / μ from market", command=self.fit_market).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=(4, 10))
        r += 1

        for title, specs in SECTIONS:
            ttk.Label(f, text=title, style="Head.TLabel").grid(
                row=r, column=0, columnspan=2, sticky="w", pady=(8, 2))
            r += 1
            if title.startswith("Overtime"):
                ttk.Checkbutton(f, text="Auto: λ / (λ + μ)", variable=self.p_ot_auto,
                                command=self.params_changed).grid(
                    row=r, column=0, columnspan=2, sticky="w")
                r += 1
            if title.startswith("Endgame"):
                ttk.Checkbutton(f, text="Enabled", variable=self.endgame_on,
                                command=self.params_changed).grid(
                    row=r, column=0, columnspan=2, sticky="w")
                r += 1
            if title.startswith("Pregame"):
                r = self._build_st(f, r)
            for i, (key, label, default) in enumerate(specs):
                # label None: second entry on the previous row (e.g. mean / sd)
                paired = i + 1 < len(specs) and specs[i + 1][1] is None
                var = tk.StringVar(value=default)
                var.trace_add("write", lambda *_: self.params_changed())
                if label is None:
                    e = ttk.Entry(pair, textvariable=var, width=4)
                    e.pack(side="left", padx=(4, 0))
                else:
                    ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=2)
                    if paired:
                        pair = tk.Frame(f, bg=PANEL)
                        pair.grid(row=r, column=1, sticky="e", padx=(10, 0))
                        e = ttk.Entry(pair, textvariable=var, width=4)
                        e.pack(side="left")
                    else:
                        e = ttk.Entry(f, textvariable=var, width=9)
                        e.grid(row=r, column=1, sticky="e", padx=(10, 0))
                    r += 1
                self.vars[key], self.entries[key] = var, e
                self.last_good[key] = float(default)
            if title.startswith("Pregame"):
                ttk.Label(f, textvariable=self.st_info, style="Muted.TLabel",
                          justify="left").grid(row=r, column=0, columnspan=2, sticky="w")
                r += 1

        ttk.Label(f, text="Alt-line ladder (home)", style="Head.TLabel").grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(10, 2))
        r += 1
        lad = ttk.Entry(f, textvariable=self.ladder_var, width=30)
        lad.grid(row=r, column=0, columnspan=2, sticky="ew")
        self.ladder_var.trace_add("write", lambda *_: self.params_changed())

    def _build_st(self, f, r):
        """Toggle + team pickers for the pregame special-teams ratings."""
        cb = ttk.Checkbutton(f, text="Use team PP/PK ratings", variable=self.st_on,
                             command=self.params_changed)
        cb.grid(row=r, column=0, columnspan=2, sticky="w")
        if self.st_ratings is None:
            cb.state(["disabled"])
        r += 1
        for label, var in [("Home team", self.home_team), ("Away team", self.away_team)]:
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=2)
            box = ttk.Combobox(f, textvariable=var, values=self.st_teams, width=7,
                               state="readonly" if self.st_teams else "disabled")
            box.grid(row=r, column=1, sticky="e", padx=(10, 0))
            box.bind("<<ComboboxSelected>>", lambda _e: self.params_changed())
            r += 1
        return r

    def team_names(self) -> tuple[str, str]:
        """(home, away) labels for the scoreboard."""
        if self.st_on.get() and self.st_ratings is not None:
            return self.home_team.get(), self.away_team.get()
        return "HOME", "AWAY"

    def _build_center(self, f):
        f.columnconfigure(0, weight=1)
        f.rowconfigure(3, weight=1)

        # Scoreboard
        sb = tk.Frame(f, bg=PANEL)
        sb.grid(row=0, column=0, sticky="ew")
        for c in (0, 2):
            sb.columnconfigure(c, weight=1)
        self.lbl_away = tk.Label(sb, text="AWAY  0", bg=PANEL, fg=AWAY_C,
                                 font=(FONT, 30, "bold"))
        self.lbl_away.grid(row=0, column=0, sticky="e", padx=20)
        mid = tk.Frame(sb, bg=PANEL)
        mid.grid(row=0, column=1)
        self.lbl_clock = tk.Label(mid, text="P1  20:00", bg=PANEL, fg=WHITE,
                                  font=(MONO, 22, "bold"))
        self.lbl_clock.pack()
        self.lbl_situation = tk.Label(mid, text="Even strength", bg=PANEL, fg=MUTED,
                                      font=(FONT, 10))
        self.lbl_situation.pack()
        self.lbl_home = tk.Label(sb, text="0  HOME", bg=PANEL, fg=HOME_C,
                                 font=(FONT, 30, "bold"))
        self.lbl_home.grid(row=0, column=2, sticky="w", padx=20)

        # Probability bar
        self.bar = tk.Canvas(f, height=34, bg=PANEL_ALT, highlightthickness=0)
        self.bar.grid(row=1, column=0, sticky="ew", pady=(12, 4))

        # Numbers
        nums = tk.Frame(f, bg=PANEL)
        nums.grid(row=2, column=0, sticky="ew", pady=(4, 8))
        self.num_labels = {}
        cells = [("away_we", "Away win-equiv", AWAY_C), ("reg", "Regulation  W / T / L (home)", WHITE),
                 ("swing", "Next goal  home / away", WHITE), ("home_we", "Home win-equiv", HOME_C)]
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

        # Chart
        self.fig = Figure(figsize=(8, 4), dpi=100, facecolor=PANEL)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.12)
        self.canvas = FigureCanvasTkAgg(self.fig, master=f)
        self.canvas.get_tk_widget().grid(row=3, column=0, sticky="nsew")
        self._cursor_x = None          # game minute under the mouse, or None
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("figure_leave_event", self._on_leave)

        # Controls
        ctl = tk.Frame(f, bg=PANEL)
        ctl.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        self.btn_play = ttk.Button(ctl, text="▶ Play", style="Play.TButton",
                                   command=self.toggle_play, width=9)
        self.btn_play.pack(side="left")
        ttk.Button(ctl, text="+1:00", command=lambda: self.advance(60)).pack(side="left", padx=4)
        ttk.Button(ctl, text="Finish", command=self.finish).pack(side="left")
        ttk.Button(ctl, text="New game", command=self.new_game).pack(side="left", padx=(4, 12))
        tk.Label(ctl, text="Seed", bg=PANEL, fg=MUTED).pack(side="left")
        ttk.Entry(ctl, textvariable=self.seed_var, width=8).pack(side="left", padx=4)
        ttk.Button(ctl, text="Replay seed",
                   command=lambda: self.new_game(replay=True)).pack(side="left", padx=(0, 12))
        tk.Label(ctl, text="Speed (game s / frame)", bg=PANEL, fg=MUTED).pack(side="left")
        ttk.Scale(ctl, from_=1, to=120, variable=self.speed, length=140,
                  command=lambda v: self.lbl_speed.config(text=f"{int(float(v))}")).pack(
            side="left", padx=4)
        self.lbl_speed = tk.Label(ctl, text="20", bg=PANEL, fg=WHITE, width=3)
        self.lbl_speed.pack(side="left")

        man = tk.Frame(f, bg=PANEL)
        man.grid(row=5, column=0, sticky="ew", pady=(6, 0))
        tk.Label(man, text="Inject:", bg=PANEL, fg=MUTED).pack(side="left", padx=(0, 6))
        for text, cmd in [("Home goal", lambda: self.inject_goal("home")),
                          ("Away goal", lambda: self.inject_goal("away")),
                          ("Home PP", lambda: self.inject_pp("home", "minor")),
                          ("Away PP", lambda: self.inject_pp("away", "minor")),
                          ("Home 5-min PP", lambda: self.inject_pp("home", "major")),
                          ("Away 5-min PP", lambda: self.inject_pp("away", "major")),
                          ("End PP", self.inject_clear_pp),
                          ("Pull goalie", self.inject_pull)]:
            ttk.Button(man, text=text, command=cmd).pack(side="left", padx=2)

    def _build_right(self, f):
        f.rowconfigure(3, weight=1)
        ttk.Label(f, text="Alt puck lines (fair, incl. OT/SO)", style="Head.TLabel").grid(
            row=0, column=0, sticky="w")
        cols = ("line", "home", "home_odds", "away_odds", "push")
        self.tree = ttk.Treeview(f, columns=cols, show="headings", height=10)
        for c, title, w in [("line", "Home", 60), ("home", "P(cover)", 75),
                            ("home_odds", "Home", 75), ("away_odds", "Away", 75),
                            ("push", "Push", 55)]:
            self.tree.heading(c, text=title)
            self.tree.column(c, width=w, anchor="e")
        self.tree.grid(row=1, column=0, sticky="ew", pady=(4, 12))

        ttk.Label(f, text="Event log", style="Head.TLabel").grid(row=2, column=0, sticky="w")
        self.log_box = tk.Listbox(f, bg=PANEL_ALT, fg=WHITE, font=(MONO, 10),
                                  borderwidth=0, highlightthickness=0, width=40,
                                  selectbackground=ACCENT, activestyle="none")
        self.log_box.grid(row=3, column=0, sticky="nsew", pady=(4, 0))

    # -- parameters ------------------------------------------------------------
    def read_params(self) -> Params:
        vals = {}
        for key, var in self.vars.items():
            try:
                v = float(var.get())
                ok = np.isfinite(v) and v >= 0
                if key in ("lam", "mu", "roll", "dt", "league_gf"):
                    ok = ok and v > 0
            except ValueError:
                ok = False
            if ok:
                self.last_good[key] = v
            self.entries[key].configure(style="TEntry" if ok else "Bad.TEntry")
            vals[key] = self.last_good[key]

        lam, mu = vals["lam"], vals["mu"]
        if self.p_ot_auto.get():
            p_ot = lam / (lam + mu)
            self.entries["p_ot"].state(["disabled"])
        else:
            p_ot = min(max(vals["p_ot"] / 100, 0.0), 1.0)
            self.entries["p_ot"].state(["!disabled"])
        eg = None
        if self.endgame_on.get():
            lead1, trail1 = vals["eg_lead1"], vals["eg_trail1"]
            eg = m.EndgameParams(lead1, trail1, trail1, lead1,
                                 pull_seconds=vals["pull_at"], roll_seconds=vals["roll"],
                                 e2_lead=vals["eg_lead2"], e2_trail=vals["eg_trail2"],
                                 pull2_seconds=vals["pull2_at"],
                                 final_lead=vals["eg_lead1f"], final_trail=vals["eg_trail1f"])
        st_home, st_away = self.st_multipliers(vals)
        return Params(lam, mu, p_ot, vals["pp_mult"], vals["sh_mult"],
                      vals["pen_rate"], min(vals["major_pct"] / 100, 1.0),
                      vals["dt"], eg, st_home, st_away,
                      vals["sim_pull_mean"], vals["sim_pull_sd"],
                      vals["sim_pull2_mean"], vals["sim_pull2_sd"])

    def st_multipliers(self, vals) -> tuple[float, float]:
        """
        PP/SH rate multipliers from the pregame special-teams ratings.

        With ratings on, each team's PP and SH scoring rate is
            league_gf/60 * pp_mult (or sh_mult) * f_team
        i.e. built off a league-average base rather than the team's own λ/μ,
        so 5v5 strength baked into λ/μ doesn't leak into special teams.
        Returned relative to λ/μ because the model scales base rates.
        """
        if not self.st_on.get() or self.st_ratings is None:
            if self.st_ratings is not None:
                self.st_info.set("Off: PP/SH rates = λ/μ × mults")
            return 1.0, 1.0
        home, away = self.home_team.get(), self.away_team.get()
        fh, fa = self.st_ratings.factors(home, away)
        base = vals["league_gf"]
        pp, sh = base * vals["pp_mult"], base * vals["sh_mult"]
        same = "  (same team twice)" if home == away else ""
        self.st_info.set(
            f"Ratings {self.st_ratings.as_of:%Y-%m-%d} ({self.st_ratings.source}){same}\n"
            f"{home} ×{fh:.2f}: PP {pp * fh:.2f} / SH {sh * fh:.2f} GF/60\n"
            f"{away} ×{fa:.2f}: PP {pp * fa:.2f} / SH {sh * fa:.2f} GF/60")
        return fh * base / vals["lam"], fa * base / vals["mu"]

    def ladder(self) -> list[float]:
        try:
            out = [float(x) for x in self.ladder_var.get().replace(";", ",").split(",")
                   if x.strip()]
            self._last_ladder = out
        except ValueError:
            pass
        return getattr(self, "_last_ladder", [float(x) for x in DEFAULT_LADDER.split(",")])

    def params_changed(self):
        if self._refresh_job:
            self.after_cancel(self._refresh_job)
        self._refresh_job = self.after(150, self._refresh_now)

    def _refresh_now(self):
        self._refresh_job = None
        if not self.running:
            self.refresh(record=False)

    def fit_market(self):
        try:
            prob = parse_price(self.market_price.get())
            total = float(self.market_total.get())
            p_ot = None if self.p_ot_auto.get() else self.read_params().p_ot
            lam, mu = m.fit_lambdas(prob, total, p_ot)
        except Exception as e:      # bad input or brentq bracket failure
            self.status.set(f"Market fit failed: {e}")
            return
        self.vars["lam"].set(f"{lam:.3f}")
        self.vars["mu"].set(f"{mu:.3f}")
        note = "  (fit ignores the endgame and PPs, so puck-drop WP can differ slightly)" \
            if self.endgame_on.get() else ""
        self.status.set(f"Fit: home {prob:.3f}, total {total:g} -> λ={lam:.3f}, μ={mu:.3f}{note}")

    # -- evaluation & drawing -----------------------------------------------------
    def evaluate(self, p: Params, state, lines=False):
        elapsed, d, pp_team, pp_left, pp_kind, pulled = state
        reg = regulation_dist(p, elapsed, d, pp_team, pp_left, pp_kind, pulled)
        probs = outcomes(reg, p.p_ot)
        if not lines:
            return probs, None
        final = final_from_reg(reg, p.p_ot)
        return probs, [m.price_line(final, L) for L in self.ladder()]

    def current_we(self, p: Params) -> float:
        if self.sim.final:
            return 1.0 if self.sim.final == "home" else 0.0
        return self.evaluate(p, self.sim.state())[0].win_equiv

    def record(self, p: Params):
        self.path_t.append(self.sim.elapsed / 60)
        self.path_wp.append(self.current_we(p))

    def next_goal_swing(self, p: Params, base_we: float):
        """Win-equiv change if home / away scores right now."""
        elapsed, d, pp_team, pp_left, pp_kind, _ = self.sim.state()
        out = []
        for team, dd in (("home", 1), ("away", -1)):
            team_pp, left = pp_team, pp_left
            if pp_team == team and pp_kind == "minor":
                team_pp, left = None, 0
            # a goal ends any pull
            we = self.evaluate(p, (elapsed, d + dd, team_pp, left, pp_kind, False))[0].win_equiv
            out.append(we - base_we)
        return out

    def refresh(self, record=True):
        p = self.read_params()
        sim = self.sim

        if sim.final:
            we = 1.0 if sim.final == "home" else 0.0
            probs = m.InGameProbs(float(sim.d > 0 and not sim.ot), float(sim.ot),
                                  float(sim.d < 0 and not sim.ot), we)
            lines = [m.price_line({sim.d: 1.0}, L) for L in self.ladder()]
        else:
            probs, lines = self.evaluate(p, sim.state(), lines=True)
        if record:
            self.path_t.append(sim.elapsed / 60)
            self.path_wp.append(probs.win_equiv)

        # scoreboard
        home_name, away_name = self.team_names()
        self.lbl_away.config(text=f"{away_name}  {sim.away}")
        self.lbl_home.config(text=f"{sim.home}  {home_name}")
        if sim.final:
            self.lbl_clock.config(text="FINAL/OT" if sim.ot else "FINAL")
        else:
            self.lbl_clock.config(text=clock_str(sim.elapsed))
        pulled = f"{sim.pulled.upper()} goalie pulled" if sim.pulled else ""
        if sim.pp_team:
            self.lbl_situation.config(
                text=f"{sim.pp_team.upper()} PP ({sim.pp_kind})  "
                     f"{sim.pp_left // 60}:{sim.pp_left % 60:02d}"
                     + (f"  ·  {pulled}" if pulled else ""),
                fg=HOME_C if sim.pp_team == "home" else AWAY_C)
        elif pulled:
            self.lbl_situation.config(text=pulled, fg=YELLOW)
        elif p.endgame and not sim.final and abs(sim.d) in (1, 2) and sim.elapsed >= 2400:
            tau = int(m.assumed_pull_time(REG - sim.elapsed, p.endgame, sim.d))
            note = f"pull assumed at {tau // 60}:{tau % 60:02d}" if tau > 0 else "no pull assumed"
            self.lbl_situation.config(text=f"Even strength  ·  {note}", fg=MUTED)
        else:
            self.lbl_situation.config(text="Even strength", fg=MUTED)

        # numbers
        we = probs.win_equiv
        self._set_num("home_we", f"{we:.1%}", fmt_odds(we))
        self._set_num("away_we", f"{1 - we:.1%}", fmt_odds(1 - we))
        self._set_num("reg", f"{probs.win:.0%} / {probs.tie:.0%} / {probs.loss:.0%}",
                      f"{fmt_odds(probs.win)} / {fmt_odds(probs.tie)} / {fmt_odds(probs.loss)}")
        if sim.final:
            self._set_num("swing", "—", "")
        else:
            sh, sa = self.next_goal_swing(p, we)
            self._set_num("swing", f"{sh:+.1%} / {sa:+.1%}", f"OT share {p.p_ot:.1%}")

        self._draw_bar(we)
        self._draw_lines(lines)
        self._draw_chart()
        self.log_box.delete(0, "end")
        for line in sim.log:
            self.log_box.insert("end", line)
        self.log_box.see("end")

    def _set_num(self, key, big, small):
        b, s = self.num_labels[key]
        b.config(text=big)
        s.config(text=small)

    def _draw_bar(self, we):
        c = self.bar
        c.delete("all")
        w = max(c.winfo_width(), 10)
        h = int(c["height"])
        split = w * (1 - we)
        c.create_rectangle(0, 0, split, h, fill=AWAY_C, width=0)
        c.create_rectangle(split, 0, w, h, fill=HOME_C, width=0)
        c.create_line(w / 2, 0, w / 2, h, fill=BG, dash=(3, 3))
        if 1 - we > 0.08:
            c.create_text(8, h / 2, text=f"AWAY {1 - we:.1%}", anchor="w",
                          fill=BG, font=(FONT, 11, "bold"))
        if we > 0.08:
            c.create_text(w - 8, h / 2, text=f"HOME {we:.1%}", anchor="e",
                          fill=BG, font=(FONT, 11, "bold"))

    def _draw_lines(self, lines):
        self.tree.delete(*self.tree.get_children())
        for lp in lines:
            push = f"{lp.push:.3f}" if lp.push > 1e-9 else ""
            self.tree.insert("", "end", values=(
                f"{lp.line:+.1f}", f"{lp.home_cover:.3f}",
                fmt_odds(lp.home_fair), fmt_odds(lp.away_fair), push))

    def _draw_chart(self):
        ax = self.ax
        ax.clear()
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color(PANEL_ALT)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlim(0, 61 if self.sim.ot else 60)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Game minute", color=MUTED, fontsize=8)
        ax.axhline(0.5, color=MUTED, lw=0.6, ls=":")
        for x in (20, 40):
            ax.axvline(x, color=PANEL_ALT, lw=1)
        for start, end, team in self.pp_spans():
            ax.axvspan(start / 60, end / 60, color=HOME_C if team == "home" else AWAY_C,
                       alpha=0.10, lw=0)
        sim = self.sim
        pulls = list(sim.pull_spans)
        if sim.pulled:
            pulls.append((sim._pull_open, sim.elapsed, sim.pulled))
        for start, end, _team in pulls:
            ax.axvspan(start / 60, max(end, start + 2) / 60, color=YELLOW, alpha=0.18, lw=0)
        if self.path_t:
            t, wp = np.array(self.path_t), np.array(self.path_wp)
            ax.plot(t, wp, color=WHITE, lw=1.6)
            ax.fill_between(t, 0.5, wp, where=wp >= 0.5, color=HOME_C, alpha=0.25,
                            interpolate=True, lw=0)
            ax.fill_between(t, 0.5, wp, where=wp < 0.5, color=AWAY_C, alpha=0.25,
                            interpolate=True, lw=0)
        for el, team in self.goal_marks:
            y = np.interp(el / 60, self.path_t, self.path_wp) if self.path_t else 0.5
            ax.scatter([el / 60], [y], s=36, zorder=5,
                       color=HOME_C if team == "home" else AWAY_C,
                       edgecolors=WHITE, linewidths=0.8)

        # hover cursor (recreated on every redraw since ax.clear() drops it)
        self._cur_line = ax.axvline(0, color=YELLOW, lw=0.9, zorder=6, visible=False)
        self._cur_dot, = ax.plot([], [], "o", color=YELLOW, ms=6, zorder=7,
                                 markeredgecolor=BG, visible=False)
        self._cur_text = ax.text(0, 0.97, "", transform=ax.get_xaxis_transform(),
                                 va="top", color=WHITE, fontsize=9, family=MONO,
                                 zorder=8, visible=False,
                                 bbox=dict(boxstyle="round,pad=0.4", fc=PANEL_ALT,
                                           ec=ACCENT, lw=1))
        self._update_cursor()
        self.canvas.draw_idle()

    # -- hover cursor ------------------------------------------------------------
    def wp_at(self, x: float) -> float:
        """Home win-equiv at game minute x, following the chart line (post-event at jumps)."""
        t, wp = self.path_t, self.path_wp
        i = int(np.searchsorted(t, x, side="right")) - 1
        if i < 0:
            return wp[0]
        if i >= len(t) - 1:
            return wp[-1]
        span = t[i + 1] - t[i]
        return wp[i] + (wp[i + 1] - wp[i]) * (x - t[i]) / span if span > 0 else wp[i]

    def _update_cursor(self):
        artists = (self._cur_line, self._cur_dot, self._cur_text)
        if self._cursor_x is None or not self.path_t:
            for a in artists:
                a.set_visible(False)
            return
        # only up to the latest recorded point; nothing has happened beyond it
        x = min(max(self._cursor_x, self.path_t[0]), self.path_t[-1])
        wp = self.wp_at(x)
        sec = int(round(x * 60))
        if sec > REG:
            clock = "OT/SO"
        elif sec == REG:
            clock = "END 3rd"
        else:
            clock = short_clock(sec)
        home_g = sum(1 for el, team in self.goal_marks if team == "home" and el <= sec)
        away_g = sum(1 for el, team in self.goal_marks if team == "away" and el <= sec)
        home, away = self.team_names()

        self._cur_line.set_xdata([x, x])
        self._cur_dot.set_data([x], [wp])
        self._cur_text.set_text(f"{clock}   {away} {away_g}-{home_g} {home}\n"
                                f"{home} {wp:.1%}  ·  {away} {1 - wp:.1%}")
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
        self._update_cursor()
        self.canvas.draw_idle()

    def _on_leave(self, _event):
        if self._cursor_x is not None:
            self._cursor_x = None
            self._update_cursor()
            self.canvas.draw_idle()

    def pp_spans(self):
        spans = list(self._pp_spans)
        if self._pp_open:
            spans.append((self._pp_open[0], self.sim.elapsed, self._pp_open[1]))
        return spans

    # -- simulation control -----------------------------------------------------------
    def new_game(self, replay: bool = False):
        """Start a game: a fresh random seed, or the seed in the box when replaying."""
        self.running = False
        self.btn_play.config(text="▶ Play")
        seed = None
        if replay:
            try:
                seed = int(self.seed_var.get().strip())
            except ValueError:
                pass
        if seed is None:
            seed = int(np.random.default_rng().integers(1_000_000))
        self.seed_var.set(str(seed))
        self.sim = GameSim(seed)
        self.path_t, self.path_wp = [], []
        self.goal_marks: list[tuple[int, str]] = []
        self._pp_spans: list[tuple[int, int, str]] = []
        self._pp_open = None
        self.status.set(f"Game seed {seed}. 'Replay seed' reruns the seed in the box "
                        f"(e.g. same game under different parameters).")
        self.refresh()

    def toggle_play(self):
        if self.sim.final:
            return
        self.running = not self.running
        self.btn_play.config(text="❚❚ Pause" if self.running else "▶ Play")
        if self.running:
            self._tick()

    def _tick(self):
        if not self.running:
            return
        self.advance(int(self.speed.get()))
        if self.sim.final:
            self.running = False
            self.btn_play.config(text="▶ Play")
            return
        self.after(FRAME_MS, self._tick)

    def advance(self, seconds: int, redraw=True):
        """Simulate `seconds` of play, recording WP just before and after each event."""
        p = self.read_params()
        sim = self.sim
        for _ in range(seconds):
            if sim.final:
                break
            before = sim.state()
            score_before = (sim.home, sim.away)
            pp_before = sim.pp_team
            if sim.step(p):
                # WP of the pre-event state at the event time, then post-event,
                # so the chart shows the jump as a vertical step
                self.path_t.append(sim.elapsed / 60)
                self.path_wp.append(
                    self.evaluate(p, (sim.elapsed, *before[1:]))[0].win_equiv)
                self._track(score_before, pp_before, sim.elapsed - 1)
                self.record(p)
            else:
                if pp_before and not sim.pp_team:
                    self._track(score_before, pp_before, sim.elapsed)
                if sim.elapsed % SAMPLE_S == 0:
                    self.record(p)
        if redraw:
            self.refresh()

    def _track(self, score_before, pp_before, at):
        """Update goal markers and PP shading after an event."""
        sim = self.sim
        if sim.home > score_before[0]:
            self.goal_marks.append((at, "home"))
        if sim.away > score_before[1]:
            self.goal_marks.append((at, "away"))
        if pp_before != sim.pp_team:
            if self._pp_open:
                self._pp_spans.append((self._pp_open[0], at, self._pp_open[1]))
                self._pp_open = None
            if sim.pp_team:
                self._pp_open = (at, sim.pp_team)

    def finish(self):
        self.running = False
        self.btn_play.config(text="▶ Play")
        self.advance(REG, redraw=False)
        self.refresh()

    # -- manual injection ------------------------------------------------------------
    def _inject(self, fn):
        if self.sim.final:
            return
        p = self.read_params()
        score_before, pp_before = (self.sim.home, self.sim.away), self.sim.pp_team
        self.record(p)
        fn()
        self._track(score_before, pp_before, self.sim.elapsed)
        if self.sim.pp_team and self.sim.pp_team == pp_before:
            # a new penalty replacing an existing PP of the same team
            if self._pp_open:
                self._pp_spans.append((self._pp_open[0], self.sim.elapsed, self._pp_open[1]))
            self._pp_open = (self.sim.elapsed, self.sim.pp_team)
        self.refresh()

    def inject_goal(self, team):
        self._inject(lambda: self.sim.goal(team))

    def inject_pp(self, team, kind):
        self._inject(lambda: self.sim.penalty(team, kind))

    def inject_pull(self):
        """Trailing team pulls its goalie now (down 1 or 2, endgame on)."""
        if self.read_params().endgame is None or abs(self.sim.d) not in (1, 2) or self.sim.pulled:
            self.status.set("Pull goalie: needs a one- or two-goal game, endgame on, goalie still in")
            return
        self._inject(self.sim.pull_goalie)

    def inject_clear_pp(self):
        if self.sim.pp_team:
            self._inject(lambda: (self.sim.log.append(
                f"{short_clock(self.sim.elapsed)}  PP ended"), self.sim.clear_pp()))


if __name__ == "__main__":
    App().mainloop()
