"""
In-game NHL win / tie / win-equivalent probabilities and alt puck-line pricing
via the Skellam distribution.

Setup
-----
Each team's goals over the FULL game are Poisson:
    home ~ Pois(lam),  away ~ Pois(mu)
For the remaining fraction t of regulation, goals still to come are
    Pois(lam * t) and Pois(mu * t)
so the remaining goal differential D = (home goals left) - (away goals left)
is Skellam(lam*t, mu*t), and

    PrWinBy(k, lam*t, mu*t) = P(D = k) = skellam.pmf(k, lam*t, mu*t)

Translating the current score
-----------------------------
Let d = current home score - away score. Final regulation diff = d + D.
    Tie  : D == -d
    Win  : D >= -d + 1
    Loss : D <= -d - 1

This reproduces the formulas:
    Trailing by m (d = -m):  Pr(Tie) = PrWinBy(m)       Pr(Win) = sum_{k>=m+1} PrWinBy(k)
    Leading  by m (d = +m):  Pr(Tie) = PrWinBy(-m)      Pr(Win) = 1 - sum_{k<=-m} PrWinBy(k)

Note: "1 - sum over k = -m-1, -m-2, ..." (sum starting at -m-1) equals Pr(Win) + Pr(Tie),
not Pr(Win). The pure regulation-win sum has to include k = -m.

Win equivalent
--------------
Regulation tie -> OT/SO. Win-equivalent = Pr(Win) + Pr(Tie) * p_ot.
By default p_ot = lam / (lam + mu), the chance the home team scores the next goal
(this ignores 3-on-3 scoring rates and the shootout; pass p_ot=0.5 for a coin flip).

Alt puck lines
--------------
Puck lines settle on the final score INCLUDING OT/SO, and the OT or shootout
winner is credited with exactly one extra goal. So the final margin is the
regulation margin, except that a regulation tie becomes +1 (prob p_ot) or -1.
A home handicap L covers when margin + L > 0 and pushes when margin + L == 0.
Regulation-only (3-way) lines are available with include_ot=False.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import skellam

REG_SECONDS = 3600


@dataclass
class InGameProbs:
    win: float          # home regulation win
    tie: float          # regulation tie (goes to OT)
    loss: float         # home regulation loss
    win_equiv: float    # home win including OT/SO share

    def __str__(self) -> str:
        return (f"Win {self.win:.3f} | Tie {self.tie:.3f} | Loss {self.loss:.3f} "
                f"| Win-equiv {self.win_equiv:.3f}")


def pr_win_by(k, lam_t: float, mu_t: float):
    """P(remaining home goals - remaining away goals == k)."""
    return skellam.pmf(k, lam_t, mu_t)


def in_game_probs(lam: float, mu: float, t: float, d: int,
                  p_ot: float | None = None,
                  endgame: "EndgameParams | None" = None) -> InGameProbs:
    """
    lam, mu : full-game expected goals for home and away
    t       : fraction of regulation REMAINING (1.0 = puck drop, 0.0 = end of 3rd)
    d       : current goal differential, home - away
    p_ot    : home win prob if tied after regulation (default lam/(lam+mu))
    endgame : optional EndgameParams for the pulled-goalie adjustment
    """
    if p_ot is None:
        p_ot = lam / (lam + mu)

    if endgame is not None:
        return _outcomes_from_margins(
            regulation_margin_dist(lam, mu, t, d, endgame=endgame), p_ot)

    lam_t, mu_t = lam * max(t, 0.0), mu * max(t, 0.0)

    if lam_t == 0 and mu_t == 0:          # regulation over: outcome is fixed
        win, tie = float(d > 0), float(d == 0)
    else:
        # Win: D >= -d + 1  ->  survival function at -d
        win = float(skellam.sf(-d, lam_t, mu_t))
        tie = float(pr_win_by(-d, lam_t, mu_t))

    loss = 1.0 - win - tie
    return InGameProbs(win, tie, loss, win + tie * p_ot)


# ---------------------------------------------------------------------------
# Power plays
# ---------------------------------------------------------------------------
# The remaining time T is split into a power-play window (s seconds) followed
# by ordinary play (T - s seconds, at the base all-situations rates).
#
# During the PP:  PP team scores at  base_rate * pp_mult
#                 SH team scores at  base_rate * sh_mult
#
# Minor penalty: a PP goal ENDS the power play. We step through the window
#   second by second. At each step the PP either ends with a goal (that mass
#   is convolved with ordinary play for whatever time is left) or carries on.
#   Shorthanded goals don't end the PP, so they just accumulate.
# Major penalty: the PP runs its full length regardless of goals, so the PP
#   window is just another Skellam, convolved with ordinary play.
#
# Default multipliers are rough league-level figures relative to a team's
# ALL-SITUATIONS goal rate (5v4 GF/60 ~ 2.3x, 4v5 GF/60 ~ 0.27x). Fit them
# per team from PP/PK data for real use. For a 5-on-3, raise pp_mult.

K_GRID = 25
GRID = np.arange(-K_GRID, K_GRID + 1)
_EPS = 1e-12


def _skellam_grid(lam_t: float, mu_t: float) -> np.ndarray:
    """Skellam pmf on GRID, safe when a rate is zero."""
    if lam_t <= 0 and mu_t <= 0:
        v = np.zeros(len(GRID))
        v[K_GRID] = 1.0
        return v
    return skellam.pmf(GRID, max(lam_t, _EPS), max(mu_t, _EPS))


def _point_mass(x: int) -> np.ndarray:
    v = np.zeros(len(GRID))
    if -K_GRID <= x <= K_GRID:
        v[x + K_GRID] = 1.0
    return v


def remaining_diff_dist(lam: float, mu: float, t: float,
                        pp_team: str | None = None, pp_seconds: float = 0.0,
                        penalty: str = "minor",
                        pp_mult: float = 2.3, sh_mult: float = 0.27,
                        dt: float = 1.0,
                        st_home: float = 1.0, st_away: float = 1.0
                        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Distribution of (home goals - away goals) over the rest of regulation,
    optionally with a power play in progress. Returns (support, probs).

    st_home, st_away : per-team special-teams multipliers on each team's
        scoring rate while the PP is on (as the PP or the SH team), e.g. from
        the pregame model's off/def spec ratings (see pregame_st.py).
    """
    T = max(t, 0.0) * REG_SECONDS
    s = min(max(pp_seconds, 0.0), T) if pp_team else 0.0
    lam_s, mu_s = lam / REG_SECONDS, mu / REG_SECONDS

    if s == 0:
        return GRID.copy(), _skellam_grid(lam_s * T, mu_s * T)

    if pp_team == "home":
        a, b, sign = lam_s * pp_mult * st_home, mu_s * sh_mult * st_away, +1
    elif pp_team == "away":
        a, b, sign = mu_s * pp_mult * st_away, lam_s * sh_mult * st_home, -1
    else:
        raise ValueError("pp_team must be 'home', 'away' or None")
    # a = PP team's rate, b = SH team's rate (goals/second).
    # A PP goal moves the home diff by +sign, a SH goal by -sign.

    support = np.arange(-2 * K_GRID, 2 * K_GRID + 1)

    if penalty == "major":
        pp_dist = (_skellam_grid(a * s, b * s) if sign == 1
                   else _skellam_grid(b * s, a * s))
        es_dist = _skellam_grid(lam_s * (T - s), mu_s * (T - s))
        return support, np.convolve(pp_dist, es_dist)

    if penalty != "minor":
        raise ValueError("penalty must be 'minor' or 'major'")

    final = np.zeros(len(support))
    q = np.zeros(K_GRID + 1)      # q[k]: PP still on, SH team has scored k
    q[0] = 1.0
    ks = np.arange(K_GRID + 1)

    n = max(int(np.ceil(s / dt)), 1)
    h = s / n
    for j in range(n):
        p_pp = 1.0 - np.exp(-a * h)        # PP goal in this step
        p_sh = 1.0 - np.exp(-b * h)        # SH goal in this step

        # Mass that ends the PP with a goal now: diff so far = sign*(1 - k)
        end_mass = q * p_pp
        if end_mass.sum() > 0:
            pp_dist = np.zeros(len(GRID))
            for k in ks[end_mass > 0]:
                pp_dist += end_mass[k] * _point_mass(sign * (1 - k))
            left = T - (j + 0.5) * h       # goal assumed mid-step
            final += np.convolve(pp_dist, _skellam_grid(lam_s * left, mu_s * left))

        # PP continues: possibly a SH goal
        stay = q * (1.0 - p_pp)
        q = stay * (1.0 - p_sh)
        q[1:] += stay[:-1] * p_sh

    # PP expired without a PP goal: diff so far = -sign*k
    pp_dist = np.zeros(len(GRID))
    for k in ks[q > 0]:
        pp_dist += q[k] * _point_mass(-sign * k)
    final += np.convolve(pp_dist, _skellam_grid(lam_s * (T - s), mu_s * (T - s)))

    return support, final


def _outcomes_from_dist(dist: np.ndarray, support: np.ndarray, d: int,
                        p_ot: float) -> InGameProbs:
    win = float(dist[support > -d].sum())
    tie = float(dist[support == -d].sum())
    loss = 1.0 - win - tie
    return InGameProbs(win, tie, loss, win + tie * p_ot)


# ---------------------------------------------------------------------------
# Endgame (pulled goalie) adjustment
# ---------------------------------------------------------------------------
# Notation map to the source formula (home = mu, away = nu there):
#   source mu, nu         -> lam, mu here (full-game expected goals)
#   source mu^EL, nu^EL   -> e_lam_L, e_mu_L
#   source mu^ET, nu^ET   -> e_lam_T, e_mu_T
#   source EXP(a, b)      -> exp(-(a + b)), P(no endgame goal)
#
# Competing-Poisson extension. If regulation would otherwise end with a
# one-goal margin, the trailing team pulls its goalie and an endgame period is
# played with its own expected goals:
#   E_lam_L, E_mu_L : home / away endgame xG when HOME LEADS by 1
#   E_lam_T, E_mu_T : home / away endgame xG when HOME TRAILS by 1
# P(at least one endgame goal) = 1 - exp(-(E_lam + E_mu)), and the first goal
# belongs to each team in proportion to its rate. Only the first goal counts:
#
#   Pr(Reg Win)  = PrWin  - PrWinBy(+1) * (1 - exp(-(E_lam_L + E_mu_L))) * E_mu_L / (E_lam_L + E_mu_L)
#   Pr(Reg Loss) = PrLoss - PrWinBy(-1) * (1 - exp(-(E_lam_T + E_mu_T))) * E_lam_T / (E_lam_T + E_mu_T)
#   Pr(Reg Tie)  = PrWinBy(0) + both transferred amounts
#
# where PrWinBy(k) is the base model's chance the FINAL regulation margin is k
# (i.e. the remaining-goals Skellam evaluated at k - d).
#
# For alt lines the leading team's share also matters: an empty-net goal moves
# a +1 margin to +2 (and -1 to -2) without changing win/tie/loss.
#
# In-game, the endgame rates are scaled by the share of the endgame window
# still to play: min(T, window) / window. At the final horn the adjustment is 0.
#
# Defaults are fitted rates (see EndgameParams); this overlay applies them
# over the last window_seconds on top of the base rates.

# Fitted 2026-11-02 from 2024-25 + 2025-26 regular-season play-by-play
# (analysis/empty_net.py): goals per 150 s of 6v5 pulled-goalie time,
# down 1: leading team 0.92 +/- 0.05 (empty net), trailing team 0.31 +/- 0.03;
# down 2: 0.99 +/- 0.05 and 0.30 +/- 0.03. Median pull: ~2:00 left down 1,
# ~2:45 down 2.

@dataclass
class EndgameParams:
    e_lam_L: float = 0.92   # home endgame xG, home up 1 (empty-net goals)
    e_mu_L: float = 0.31    # away endgame xG, home up 1 (extra attacker)
    e_lam_T: float = 0.31   # home endgame xG, home down 1 (extra attacker)
    e_mu_T: float = 0.92    # away endgame xG, home down 1 (empty-net goals)
    window_seconds: float = 150.0   # e_* are xG over this many seconds pulled
    pull_seconds: float = 120.0     # pull-time model: assumed pull, down 1 (s left) ...
    roll_seconds: float = 30.0      # ... rolled back by this much once passed
    # two-goal pulls (pull-time model only)
    e2_lead: float = 0.99   # leading team xG / window, opponent down 2 and pulled
    e2_trail: float = 0.30  # trailing team xG / window
    pull2_seconds: float = 165.0    # assumed pull, down 2 (s left); 0 = never


def _endgame_split(e_lead: float, e_trail: float, scale: float):
    """(P(trailing team scores first), P(leading team scores first)) in the endgame."""
    a, b = e_lead * scale, e_trail * scale
    if a + b <= 0:
        return 0.0, 0.0
    p_any = 1.0 - np.exp(-(a + b))
    return p_any * b / (a + b), p_any * a / (a + b)


def apply_endgame(margins: dict[int, float], T_seconds: float,
                  eg: EndgameParams) -> dict[int, float]:
    """Move one-goal regulation margins per the endgame model."""
    scale = min(max(T_seconds, 0.0), eg.window_seconds) / eg.window_seconds
    if scale == 0:
        return dict(margins)
    out = dict(margins)

    p1 = margins.get(1, 0.0)            # home leads by 1
    tie_l, en_l = _endgame_split(eg.e_lam_L, eg.e_mu_L, scale)
    out[1] = p1 * (1 - tie_l - en_l)
    out[0] = out.get(0, 0.0) + p1 * tie_l
    out[2] = out.get(2, 0.0) + p1 * en_l

    pm1 = margins.get(-1, 0.0)          # home trails by 1
    tie_t, en_t = _endgame_split(eg.e_mu_T, eg.e_lam_T, scale)
    out[-1] = pm1 * (1 - tie_t - en_t)
    out[0] = out.get(0, 0.0) + pm1 * tie_t
    out[-2] = out.get(-2, 0.0) + pm1 * en_t
    return out


def regulation_margin_dist(lam: float, mu: float, t: float, d: int,
                           endgame: EndgameParams | None = None,
                           **pp_kwargs) -> dict[int, float]:
    """FINAL regulation margin distribution (home - away), with optional PP and endgame."""
    support, dist = remaining_diff_dist(lam, mu, t, **pp_kwargs)
    out: dict[int, float] = {}
    for m, p in zip(support + d, dist):
        if p > 0:
            out[int(m)] = out.get(int(m), 0.0) + float(p)
    if endgame is not None:
        out = apply_endgame(out, max(t, 0.0) * REG_SECONDS, endgame)
    return out


# ---------------------------------------------------------------------------
# Pull-time endgame model (used by the simulator)
# ---------------------------------------------------------------------------
# Instead of the overlay above, the goalie pull is an explicit state:
#   - a team down 1 or 2 with its goalie pulled scores / concedes at the
#     endgame rates (per window_seconds), which REPLACE the base rates (a
#     power play still takes precedence);
#   - a goal ends the pull (the new margin may be pulled again right away
#     if its assumed pull time has passed, e.g. down 2 -> down 1 at 1:30);
#   - a goalie not yet pulled is assumed to come out at `assumed_pull_time`:
#     2:00 left down 1 by default; once 2:00 passes without a pull, 1:30;
#     then 1:00, and so on (down 2: 2:45, 2:15, ...). The assumption applies
#     to any deficit, current or future.
# The remaining time is split at the start of the endgame: ordinary Skellam
# (incl. any power play) up to it, then a second-by-second chain over
# (margin, goalie pulled, PP on) to the horn.

def assumed_pull_time(T_seconds: float, eg: EndgameParams, margin: int = 1) -> float:
    """Seconds left at which a goalie not yet pulled is assumed to come out."""
    first = eg.pull_seconds if abs(margin) == 1 else eg.pull2_seconds
    if first <= 0:
        return 0.0
    if T_seconds > first:
        return first
    k = int(np.floor((first - T_seconds) / eg.roll_seconds)) + 1
    return max(first - k * eg.roll_seconds, 0.0)


_EG_PAD = 30      # chain margin grid: d +/- this


def endgame_margin_dist(lam: float, mu: float, t: float, d: int,
                        eg: EndgameParams, pulled: bool = False,
                        pull_at: float | None = None,
                        pp_team: str | None = None, pp_seconds: float = 0.0,
                        penalty: str = "minor",
                        pp_mult: float = 2.3, sh_mult: float = 0.27,
                        dt: float = 1.0,
                        st_home: float = 1.0, st_away: float = 1.0
                        ) -> dict[int, float]:
    """
    FINAL regulation margin distribution under the pull-time endgame model.

    pulled  : the trailing team's goalie is out right now (used at d = +/-1, +/-2)
    pull_at : (down 1, down 2) seconds left at which a not-yet-pulled goalie
              comes out (default assumed_pull_time(T, eg, margin))
    Other arguments as in remaining_diff_dist.
    """
    T = max(t, 0.0) * REG_SECONDS
    if T <= 0:
        return {d: 1.0}
    tau1, tau2 = pull_at if pull_at is not None else (
        assumed_pull_time(T, eg, 1), assumed_pull_time(T, eg, 2))
    tau = max(tau1, tau2)
    pulled = pulled and abs(d) in (1, 2)
    pp_on = pp_team is not None and pp_seconds > 0
    pp_end = T - pp_seconds if pp_on else -1.0     # seconds left when PP expires

    # Chain from now if the goalie is out or the PP runs into the endgame,
    # otherwise from the earliest assumed pull time.
    S0 = T if (pulled or (pp_on and pp_end < min(T, tau))) else min(T, tau)

    n_m = 2 * _EG_PAD + 1
    margins = np.arange(d - _EG_PAD, d + _EG_PAD + 1)
    P = np.zeros((2, 2, n_m))             # [pulled, pp on, margin]
    if S0 < T:
        kw = dict(pp_team=pp_team, pp_seconds=pp_seconds, penalty=penalty,
                  pp_mult=pp_mult, sh_mult=sh_mult, dt=dt,
                  st_home=st_home, st_away=st_away) if pp_on else {}
        support, dist = remaining_diff_dist(lam, mu, (T - S0) / REG_SECONDS, **kw)
        keep = np.abs(support) <= _EG_PAD
        P[0, 0, support[keep] + _EG_PAD] = dist[keep]
    else:
        P[int(pulled), int(pp_on), _EG_PAD] = 1.0
    if S0 <= 0:
        return {int(k): float(v) for k, v in zip(margins, P.sum(axis=(0, 1))) if v > 0}

    lam_s, mu_s = lam / REG_SECONDS, mu / REG_SECONDS
    W = eg.window_seconds
    # rates[pulled, pp] -> (home, away) per margin
    rh = np.full((2, 2, n_m), lam_s)
    ra = np.full((2, 2, n_m), mu_s)
    up1, dn1 = margins == 1, margins == -1
    up2, dn2 = margins == 2, margins == -2
    rh[1, 0, up1], ra[1, 0, up1] = eg.e_lam_L / W, eg.e_mu_L / W
    rh[1, 0, dn1], ra[1, 0, dn1] = eg.e_lam_T / W, eg.e_mu_T / W
    rh[1, 0, up2], ra[1, 0, up2] = eg.e2_lead / W, eg.e2_trail / W
    rh[1, 0, dn2], ra[1, 0, dn2] = eg.e2_trail / W, eg.e2_lead / W
    if pp_on:
        if pp_team == "home":
            rh[:, 1], ra[:, 1] = lam_s * pp_mult * st_home, mu_s * sh_mult * st_away
        else:
            rh[:, 1], ra[:, 1] = lam_s * sh_mult * st_home, mu_s * pp_mult * st_away
    # a PP goal ends a minor
    home_ends_pp = pp_on and pp_team == "home" and penalty == "minor"
    away_ends_pp = pp_on and pp_team == "away" and penalty == "minor"
    one, two = up1 | dn1, up2 | dn2

    n = max(int(np.ceil(S0 - 1e-9)), 1)
    h = S0 / n
    tot = rh + ra
    pg = 1.0 - np.exp(-tot * h)
    with np.errstate(invalid="ignore", divide="ignore"):
        ph = np.where(tot > 0, pg * rh / tot, 0.0)
        pa = np.where(tot > 0, pg * ra / tot, 0.0)
    for j in range(n):
        s = S0 - j * h                    # seconds left at the start of this step
        if pp_on and s <= pp_end + 1e-9:  # PP expires
            P[:, 0] += P[:, 1]
            P[:, 1] = 0.0
        for mask, tau_m in ((one, tau1), (two, tau2)):
            if s <= tau_m + 1e-9:         # not-yet-pulled games at this margin pull now
                P[1, :, mask] += P[0, :, mask]
                P[0, :, mask] = 0.0
        home_g, away_g = P * ph, P * pa
        P = P - home_g - away_g
        # any goal ends the pull; shift margins
        for pp in (0, 1):
            to_h = 0 if (pp == 1 and home_ends_pp) else pp
            to_a = 0 if (pp == 1 and away_ends_pp) else pp
            P[0, to_h, 1:] += home_g[:, pp, :-1].sum(axis=0)
            P[0, to_a, :-1] += away_g[:, pp, 1:].sum(axis=0)

    final = P.sum(axis=(0, 1))
    return {int(k): float(v) for k, v in zip(margins, final) if v > 0}


def _outcomes_from_margins(margins: dict[int, float], p_ot: float) -> InGameProbs:
    win = sum(p for m, p in margins.items() if m > 0)
    tie = margins.get(0, 0.0)
    loss = sum(p for m, p in margins.items() if m < 0)
    return InGameProbs(win, tie, loss, win + tie * p_ot)


def in_game_probs_endgame(lam: float, mu: float, t: float, d: int,
                          eg: EndgameParams = EndgameParams(),
                          p_ot: float | None = None) -> InGameProbs:
    """Your formula written out directly (no power play), for transparency."""
    if p_ot is None:
        p_ot = lam / (lam + mu)
    base = in_game_probs(lam, mu, t, d, p_ot)
    T = max(t, 0.0) * REG_SECONDS
    if T == 0:
        return base
    lam_t, mu_t = lam * t, mu * t
    win_by_plus1 = pr_win_by(1 - d, lam_t, mu_t)     # final reg margin = +1
    win_by_minus1 = pr_win_by(-1 - d, lam_t, mu_t)   # final reg margin = -1

    s = min(T, eg.window_seconds) / eg.window_seconds
    eLL, eML = eg.e_lam_L * s, eg.e_mu_L * s
    eLT, eMT = eg.e_lam_T * s, eg.e_mu_T * s
    to_tie_from_win = win_by_plus1 * (1 - np.exp(-(eLL + eML))) * eML / (eLL + eML)
    to_tie_from_loss = win_by_minus1 * (1 - np.exp(-(eLT + eMT))) * eLT / (eLT + eMT)

    win = base.win - to_tie_from_win
    loss = base.loss - to_tie_from_loss
    tie = base.tie + to_tie_from_win + to_tie_from_loss
    return InGameProbs(win, tie, loss, win + tie * p_ot)


def in_game_probs_pp(lam: float, mu: float, t: float, d: int,
                     pp_team: str, pp_seconds: float,
                     penalty: str = "minor",
                     pp_mult: float = 2.3, sh_mult: float = 0.27,
                     p_ot: float | None = None,
                     dt: float = 1.0,
                     endgame: EndgameParams | None = None,
                     st_home: float = 1.0, st_away: float = 1.0) -> InGameProbs:
    """
    Win / tie / loss / win-equivalent for the HOME team with a power play in progress.

    pp_team    : "home" or "away" (team with the man advantage)
    pp_seconds : seconds left on the penalty
    penalty    : "minor" (ends on a PP goal) or "major" (runs full length)
    pp_mult, sh_mult : scoring-rate multipliers for the PP and SH teams
    endgame    : optional EndgameParams for the pulled-goalie adjustment
    st_home, st_away : per-team special-teams rate multipliers during the PP
    """
    if p_ot is None:
        p_ot = lam / (lam + mu)
    if max(t, 0.0) * REG_SECONDS == 0 or pp_seconds <= 0:
        return in_game_probs(lam, mu, t, d, p_ot, endgame)
    margins = regulation_margin_dist(lam, mu, t, d, endgame=endgame,
                                     pp_team=pp_team, pp_seconds=pp_seconds,
                                     penalty=penalty, pp_mult=pp_mult,
                                     sh_mult=sh_mult, dt=dt,
                                     st_home=st_home, st_away=st_away)
    return _outcomes_from_margins(margins, p_ot)


# ---------------------------------------------------------------------------
# Final margin and alt puck lines
# ---------------------------------------------------------------------------

def final_margin_dist(lam: float, mu: float, t: float, d: int,
                      include_ot: bool = True, p_ot: float | None = None,
                      endgame: EndgameParams | None = None,
                      **pp_kwargs) -> dict[int, float]:
    """
    Distribution of the FINAL home margin, given the current state.
    include_ot=True: a regulation tie becomes +1 (prob p_ot) or -1, which is
    how puck lines settle. include_ot=False: regulation margin only.
    endgame: optional EndgameParams (pulled-goalie adjustment).
    pp_kwargs are passed to remaining_diff_dist (pp_team, pp_seconds, penalty, ...).
    """
    if p_ot is None:
        p_ot = lam / (lam + mu)
    reg = regulation_margin_dist(lam, mu, t, d, endgame=endgame, **pp_kwargs)
    out: dict[int, float] = {}
    for m, p in reg.items():
        if p <= 0:
            continue
        if include_ot and m == 0:
            out[1] = out.get(1, 0.0) + p * p_ot
            out[-1] = out.get(-1, 0.0) + p * (1 - p_ot)
        else:
            out[m] = out.get(m, 0.0) + p
    return out


def prob_to_american(p: float) -> float:
    if p <= 0 or p >= 1:
        return float("nan")
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p


def fmt_american(odds: float) -> str:
    if np.isnan(odds):
        return "n/a"
    return f"{odds:+.0f}"


@dataclass
class LinePrice:
    line: float          # home handicap (e.g. -1.5 means home -1.5)
    home_cover: float    # P(home covers)
    push: float          # P(push), only non-zero for whole-number lines
    away_cover: float    # P(away covers the opposite side)

    @property
    def home_fair(self) -> float:
        """Home cover prob with pushes removed (what fair odds price)."""
        return self.home_cover / (1 - self.push) if self.push < 1 else float("nan")

    @property
    def away_fair(self) -> float:
        return self.away_cover / (1 - self.push) if self.push < 1 else float("nan")

    def __str__(self) -> str:
        away_line = -self.line
        push = f"  push {self.push:.3f}" if self.push > 1e-12 else ""
        return (f"Home {self.line:+.1f}: {self.home_cover:.3f} "
                f"({fmt_american(prob_to_american(self.home_fair)):>6})  |  "
                f"Away {away_line:+.1f}: {self.away_cover:.3f} "
                f"({fmt_american(prob_to_american(self.away_fair)):>6}){push}")


def price_line(margin_dist: dict[int, float], line: float) -> LinePrice:
    """Price a home handicap `line` against a final-margin distribution."""
    home = sum(p for m, p in margin_dist.items() if m + line > 0)
    push = sum(p for m, p in margin_dist.items() if m + line == 0)
    away = sum(p for m, p in margin_dist.items() if m + line < 0)
    return LinePrice(line, home, push, away)


def alt_lines(lam: float, mu: float, t: float, d: int,
              lines=(-3.5, -2.5, -1.5, 1.5, 2.5, 3.5),
              include_ot: bool = True, p_ot: float | None = None,
              endgame: EndgameParams | None = None,
              **pp_kwargs) -> list[LinePrice]:
    """
    Fair prices for a ladder of home handicaps from the current game state.
    Lines are from the HOME side: -1.5 = home -1.5 (away +1.5 is the other side).
    Works pre-game (t=1, d=0), live, with a power play (pp_kwargs) and with
    the endgame adjustment (endgame=EndgameParams()).
    """
    dist = final_margin_dist(lam, mu, t, d, include_ot, p_ot, endgame, **pp_kwargs)
    return [price_line(dist, L) for L in lines]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def t_from_clock(period: int, seconds_left_in_period: float) -> float:
    """Convert period (1-3) and clock to fraction of regulation remaining."""
    elapsed = (period - 1) * 1200 + (1200 - seconds_left_in_period)
    return max(REG_SECONDS - elapsed, 0) / REG_SECONDS


def fit_lambdas(p_home_win_equiv: float, total: float,
                p_ot: float | None = 0.5) -> tuple[float, float]:
    """
    Back out (lam, mu) from a de-vigged 2-way home win prob (incl. OT/SO)
    and an expected total goals figure. Solves for the split of `total`.
    """
    def err(share):
        lam, mu = total * share, total * (1 - share)
        return in_game_probs(lam, mu, 1.0, 0, p_ot).win_equiv - p_home_win_equiv

    share = brentq(err, 0.05, 0.95)
    return total * share, total * (1 - share)


def wp_path(lam: float, mu: float, goals: list[tuple[float, int]],
            step_s: int = 30, p_ot: float | None = None):
    """
    Win-equivalent path over regulation.
    goals: list of (elapsed_seconds, +1 home goal / -1 away goal).
    Returns arrays (elapsed_seconds, home_win_equiv).
    """
    times = np.arange(0, REG_SECONDS + 1, step_s)
    goals = sorted(goals)
    out = []
    for s in times:
        d = sum(g for (gs, g) in goals if gs <= s)
        out.append(in_game_probs(lam, mu, 1 - s / REG_SECONDS, d, p_ot).win_equiv)
    return times, np.array(out)


if __name__ == "__main__":
    lam, mu = 3.2, 2.8  # home, away full-game expected goals

    print("Puck drop:            ", in_game_probs(lam, mu, 1.0, 0))
    print("Home down 1, 2nd half:", in_game_probs(lam, mu, 0.5, -1))
    print("Home up 1, 5:00 left: ", in_game_probs(lam, mu, t_from_clock(3, 300), 1))
    print("Home up 2, 5:00 left: ", in_game_probs(lam, mu, t_from_clock(3, 300), 2))
    print("Tied, end of reg:     ", in_game_probs(lam, mu, 0.0, 0))

    # Check the formula translation explicitly (leading by m)
    m, t = 1, 0.25
    lt, mt = lam * t, mu * t
    ks = np.arange(-40, 41)
    pmf = pr_win_by(ks, lt, mt)
    win_correct = 1 - pmf[ks <= -m].sum()
    win_as_written = 1 - pmf[ks <= -m - 1].sum()
    p = in_game_probs(lam, mu, t, m)
    print(f"\nLeading by {m}, t={t}: win={p.win:.4f}, tie={p.tie:.4f}")
    print(f"  1 - sum_(k<=-m)   = {win_correct:.4f}  (regulation win)")
    print(f"  1 - sum_(k<=-m-1) = {win_as_written:.4f}  (= win + tie)")

    print("\nFrom market: 58% home, total 6.0 ->", fit_lambdas(0.58, 6.0))

    # --- Power plays ---------------------------------------------------
    print("\nPower plays (tied, start of 2nd period, t = 2/3):")
    t2 = t_from_clock(2, 1200)
    print("  No PP:              ", in_game_probs(lam, mu, t2, 0))
    print("  Home 2:00 minor PP: ", in_game_probs_pp(lam, mu, t2, 0, "home", 120))
    print("  Away 2:00 minor PP: ", in_game_probs_pp(lam, mu, t2, 0, "away", 120))
    print("  Home 5:00 major PP: ", in_game_probs_pp(lam, mu, t2, 0, "home", 300, "major"))

    print("\nHome down 1, 3:00 left, home PP with 2:00 left on minor:")
    t3 = t_from_clock(3, 180)
    print("  No PP:  ", in_game_probs(lam, mu, t3, -1))
    print("  With PP:", in_game_probs_pp(lam, mu, t3, -1, "home", 120))

    # --- Alt puck lines ------------------------------------------------
    ladder = (-3.5, -2.5, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.5, 3.5)

    print("\nAlt lines, pre-game (fair odds, incl. OT/SO):")
    for lp in alt_lines(lam, mu, 1.0, 0, ladder):
        print("  ", lp)

    print("\nAlt lines, home up 1 midway through 2nd:")
    for lp in alt_lines(lam, mu, t_from_clock(2, 600), 1):
        print("  ", lp)

    print("\nSame state, away on a 2:00 minor PP:")
    for lp in alt_lines(lam, mu, t_from_clock(2, 600), 1,
                        pp_team="away", pp_seconds=120):
        print("  ", lp)

    print("\nRegulation-only (3-way) lines, pre-game:")
    for lp in alt_lines(lam, mu, 1.0, 0, (-1.5, -0.5, 0.5, 1.5), include_ot=False):
        print("  ", lp)

    # --- Endgame adjustment ---------------------------------------------
    eg = EndgameParams()
    print("\nEndgame adjustment (W / T / L / win-equiv):")
    for label, tt, dd in [("Pre-game", 1.0, 0),
                          ("Home up 1, 10:00 left", t_from_clock(3, 600), 1),
                          ("Home up 1, 2:00 left", t_from_clock(3, 120), 1),
                          ("Home down 1, 2:00 left", t_from_clock(3, 120), -1),
                          ("Home up 1, 0:30 left", t_from_clock(3, 30), 1)]:
        print(f"  {label:<24} base {in_game_probs(lam, mu, tt, dd)}")
        print(f"  {'':<24} adj  {in_game_probs(lam, mu, tt, dd, endgame=eg)}")

    print("\nAlt lines pre-game, base vs endgame-adjusted (home cover):")
    base_l = alt_lines(lam, mu, 1.0, 0, (-2.5, -1.5, 1.5, 2.5))
    adj_l = alt_lines(lam, mu, 1.0, 0, (-2.5, -1.5, 1.5, 2.5), endgame=eg)
    for b, a in zip(base_l, adj_l):
        print(f"   Home {b.line:+.1f}: {b.home_cover:.3f} -> {a.home_cover:.3f} "
              f"({fmt_american(prob_to_american(a.home_fair))})")

    print("\nHome up 1, 2:00 left, alt lines with endgame:")
    for lp in alt_lines(lam, mu, t_from_clock(3, 120), 1, (-1.5, -2.5), endgame=eg):
        print("  ", lp)

    # --- Sanity checks -------------------------------------------------
    base = in_game_probs(lam, mu, 0.5, 1)
    zero = in_game_probs_pp(lam, mu, 0.5, 1, "home", 0)
    neutral = in_game_probs_pp(lam, mu, 0.5, 1, "home", 120, "major",
                               pp_mult=1.0, sh_mult=1.0)
    coarse = in_game_probs_pp(lam, mu, t2, 0, "home", 120, dt=1.0).win_equiv
    fine = in_game_probs_pp(lam, mu, t2, 0, "home", 120, dt=0.1).win_equiv

    st = t_from_clock(2, 600)
    ml = in_game_probs_pp(lam, mu, st, 1, "away", 120).win_equiv
    half = price_line(final_margin_dist(lam, mu, st, 1, pp_team="away",
                                        pp_seconds=120), -0.5).home_cover
    dist = final_margin_dist(lam, mu, st, 1)
    end_up1 = price_line(final_margin_dist(lam, mu, 0.0, 1), -1.5).home_cover

    print("\nChecks:")
    print(f"  pp_seconds=0 matches base:          {abs(base.win_equiv - zero.win_equiv) < 1e-12}")
    print(f"  multipliers=1 matches base:         {abs(base.win_equiv - neutral.win_equiv) < 1e-9}")
    print(f"  dt=1s vs dt=0.1s difference:        {abs(coarse - fine):.2e}")
    print(f"  home -0.5 == win-equiv (with PP):   {abs(ml - half) < 1e-9}")
    print(f"  margin distribution sums to 1:      {abs(sum(dist.values()) - 1) < 1e-9}")
    print(f"  up 1 at final horn, -1.5 = 0:       {end_up1 == 0.0}")

    states = [(1.0, 0), (0.5, 1), (t_from_clock(3, 120), 1),
              (t_from_clock(3, 120), -1), (t_from_clock(3, 45), -2)]
    match = all(
        abs(in_game_probs_endgame(lam, mu, tt, dd, eg).win
            - in_game_probs(lam, mu, tt, dd, endgame=eg).win) < 1e-9
        and abs(in_game_probs_endgame(lam, mu, tt, dd, eg).tie
                - in_game_probs(lam, mu, tt, dd, endgame=eg).tie) < 1e-9
        for tt, dd in states)
    adj = in_game_probs(lam, mu, 0.5, 1, endgame=eg)
    zero_eg = EndgameParams(0, 0, 0, 0)
    print(f"  formula == distribution method:     {match}")
    print(f"  endgame W+T+L == 1:                 {abs(adj.win + adj.tie + adj.loss - 1) < 1e-9}")
    print(f"  zero endgame rates == base:         "
          f"{abs(in_game_probs(lam, mu, 0.5, 1, endgame=zero_eg).win - base.win) < 1e-9}")
    print(f"  no effect at final horn:            "
          f"{in_game_probs(lam, mu, 0.0, 1, endgame=eg).win == 1.0}")

    # --- Pull-time endgame model ------------------------------------------
    print("\nPull-time endgame model, home up 1 (W / T / L / win-equiv):")
    for label, T, pulled in [("5:00 left", 300, False), ("2:30 left, in net", 150, False),
                             ("2:30 left, pulled", 150, True), ("1:00 left, in net", 60, False),
                             ("1:00 left, pulled", 60, True)]:
        dist = endgame_margin_dist(lam, mu, T / REG_SECONDS, 1, eg, pulled=pulled)
        print(f"  {label:<20} {_outcomes_from_margins(dist, 0.5)}   "
              f"(pull assumed at {assumed_pull_time(T, eg):.0f}s)")
    no_pull = EndgameParams(pull_seconds=0, pull2_seconds=0)
    eg_dists = [endgame_margin_dist(lam, mu, 0.1, 1, eg),
                endgame_margin_dist(lam, mu, 0.05, -1, eg, pulled=True, pp_team="home",
                                    pp_seconds=120)]
    print(f"  pull-time model sums to 1:          "
          f"{all(abs(sum(x.values()) - 1) < 1e-9 for x in eg_dists)}")
    print(f"  never pulling == base model:        "
          f"{max(abs(endgame_margin_dist(lam, mu, 0.1, 1, no_pull).get(k, 0) - v) for k, v in regulation_margin_dist(lam, mu, 0.1, 1).items()) < 1e-9}")
    print(f"  down-1 pulls at 121/120/91/90/30s:  "
          f"{[assumed_pull_time(x, eg) for x in (121, 120, 91, 90, 30)]}")
    print(f"  down-2 pulls at 200/165/136/135s:   "
          f"{[assumed_pull_time(x, eg, 2) for x in (200, 165, 136, 135)]}")
