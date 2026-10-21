"""
Live pricer: GameState -> fair (no-vig) prices from the Skellam live model.

Prices produced for every state:
  - 2-way moneyline incl. OT/SO (home / away win-equivalent, fair American odds)
  - regulation 3-way (home / tie / away)
  - alt puck lines incl. OT/SO
  - live totals (final score incl. the OT/SO winner's goal), see below

Pregame inputs are a home win prob (incl. OT) and an expected total, from the
market or from the pregame model's predictions file; lam / mu are backed out
with nhl_ingame_skellam.fit_lambdas. PP / SH scoring uses the pregame model's
special-teams ratings for the two teams (pregame_st.py), same as the sim UI.

Game situations:
  - regulation: nhl_ingame_skellam.regulation_margin_dist with the live score,
    clock, PP window (minor / major) and pulled-goalie endgame model
  - 5-on-3: the PP multiplier is the time-weighted mix of pp53_mult (two-man
    portion) and pp_mult over the whole advantage
  - OT / shootout: the game is tied, so home wins with p_ot
Totals use a joint grid of independent Poissons for each team's remaining
goals (PP window at its expected length, endgame goals and the OT goal moved
on the grid). That approximation is separate from the exact diff model used
for the moneyline and puck lines, so the two can differ very slightly.
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
    lam, mu = m.fit_lambdas(home_prob, total, p_ot)
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

    def _regulation(self, st: GameState) -> dict[int, float]:
        c = self.cfg
        T = max(REG_SECONDS - st.elapsed, 0)
        kw = {}
        pp = self._pp(st, T)
        if pp:
            team, s, kind, mult = pp
            kw = dict(pp_team=team, pp_seconds=s, penalty=kind, pp_mult=mult,
                      sh_mult=c.sh_mult, dt=c.dt, st_home=c.st_home, st_away=c.st_away)
        return m.regulation_margin_dist(c.lam, c.mu, T / REG_SECONDS, st.d,
                                        endgame=c.endgame, **kw)

    def _total_dist(self, st: GameState) -> dict[int, float]:
        """Final total goals (incl. the OT/SO goal) -> prob."""
        c = self.cfg
        cur = st.home_score + st.away_score
        T = max(REG_SECONDS - st.elapsed, 0)
        lam_s, mu_s = c.lam / REG_SECONDS, c.mu / REG_SECONDS
        h_mean, a_mean = lam_s * T, mu_s * T
        pp = self._pp(st, T)
        if pp:
            team, s, kind, mult = pp
            if team == "home":
                a_rate, b_rate = lam_s * mult * c.st_home, mu_s * c.sh_mult * c.st_away
            else:
                a_rate, b_rate = mu_s * mult * c.st_away, lam_s * c.sh_mult * c.st_home
            # a minor ends on a PP goal: use its expected length
            s_eff = s if kind == "major" or a_rate <= 0 else (1 - math.exp(-a_rate * s)) / a_rate
            pp_mean, sh_mean = a_rate * s_eff, b_rate * s_eff
            if team == "home":
                h_mean = pp_mean + lam_s * (T - s_eff)
                a_mean = sh_mean + mu_s * (T - s_eff)
            else:
                a_mean = pp_mean + mu_s * (T - s_eff)
                h_mean = sh_mean + lam_s * (T - s_eff)

        k = np.arange(TOTAL_GRID + 1)
        joint = np.outer(poisson.pmf(k, h_mean), poisson.pmf(k, a_mean))
        joint /= joint.sum()

        eg = c.endgame
        scale = min(T, eg.window_seconds) / eg.window_seconds if eg and T > 0 else 0.0
        tie_l = en_l = tie_t = en_t = 0.0
        if scale > 0:
            tie_l, en_l = m._endgame_split(eg.e_lam_L, eg.e_mu_L, scale)
            tie_t, en_t = m._endgame_split(eg.e_mu_T, eg.e_lam_T, scale)

        out: dict[int, float] = {}

        def add(margin, total, p):
            if margin == 0:            # OT/SO winner's goal counts in the final score
                total += 1
            out[total] = out.get(total, 0.0) + p

        for h in k:
            for a in k:
                p = joint[h, a]
                if p < 1e-12:
                    continue
                margin, total = st.d + h - a, cur + h + a
                if margin == 1 and scale > 0:
                    add(0, total + 1, p * tie_l)
                    add(2, total + 1, p * en_l)
                    p *= 1 - tie_l - en_l
                elif margin == -1 and scale > 0:
                    add(0, total + 1, p * tie_t)
                    add(-2, total + 1, p * en_t)
                    p *= 1 - tie_t - en_t
                add(margin, total, p)
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
