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
from scipy.stats import poisson, skellam

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
# Double minor: as a minor over the first half, except that a PP goal there
#   starts a fresh 2:00 minor instead of ending the PP. pp_seconds is the
#   whole double's time left and may run past the end of regulation.
#
# Default multipliers are rough league-level figures relative to a team's
# ALL-SITUATIONS goal rate (5v4 GF/60 ~ 2.3x, 4v5 GF/60 ~ 0.27x). Fit them
# per team from PP/PK data for real use. For a 5-on-3, raise pp_mult.

K_GRID = 25
GRID = np.arange(-K_GRID, K_GRID + 1)
SUPPORT = np.arange(-2 * K_GRID, 2 * K_GRID + 1)   # remaining_diff_dist's output
_EPS = 1e-12

# Penalty kinds, by what a power-play goal does to the advantage:
#   minor  : ends it
#   double : double minor; a PP goal in the first half ends only the first
#            minor and the second starts at once with a full 2:00 (pp_seconds
#            <= DOUBLE_HALF is the second half, i.e. an ordinary minor)
#   major  : nothing, it runs in full
PENALTIES = ("minor", "double", "major")
DOUBLE_HALF = 120.0


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

    support = SUPPORT.copy()

    if penalty == "double" and pp_seconds <= DOUBLE_HALF:
        penalty = "minor"                  # second half: an ordinary minor
    if penalty not in PENALTIES:
        raise ValueError(f"penalty must be one of {PENALTIES}")

    if penalty == "major":
        pp_dist = (_skellam_grid(a * s, b * s) if sign == 1
                   else _skellam_grid(b * s, a * s))
        es_dist = _skellam_grid(lam_s * (T - s), mu_s * (T - s))
        return support, np.convolve(pp_dist, es_dist)

    if penalty == "minor":
        return support, _pp_window(
            a, b, sign, s, dt,
            lambda u: _skellam_grid(lam_s * (T - u), mu_s * (T - u)),
            _skellam_grid(lam_s * (T - s), mu_s * (T - s)))

    # Double minor, first half still running. A PP goal ends only the first
    # minor; the second starts at once with a full 2:00 (a fresh minor). If
    # the first half runs out, the second minor follows at its full 2:00.
    w = min(pp_seconds - DOUBLE_HALF, T)   # first-half time left in regulation
    minor = dict(pp_team=pp_team, pp_seconds=DOUBLE_HALF, penalty="minor",
                 pp_mult=pp_mult, sh_mult=sh_mult, dt=dt,
                 st_home=st_home, st_away=st_away)
    # a fresh 2:00 minor with `left` s of regulation to go = (the minor over
    # its own 2:00) + (even strength after), since Skellams add
    m120 = remaining_diff_dist(lam, mu, DOUBLE_HALF / REG_SECONDS, **minor)[1]
    short: dict[float, np.ndarray] = {}

    def after_ppg(u: float) -> np.ndarray:
        left = T - u
        if left >= DOUBLE_HALF:
            es = left - DOUBLE_HALF
            return np.convolve(m120, _skellam_grid(lam_s * es, mu_s * es))
        key = round(left, 6)
        if key not in short:
            short[key] = remaining_diff_dist(lam, mu, left / REG_SECONDS, **minor)[1]
        return short[key]

    after_first = (remaining_diff_dist(lam, mu, (T - w) / REG_SECONDS, **minor)[1]
                   if T - w > 0 else _point_mass(0))
    return support, _pp_window(a, b, sign, w, dt, after_ppg, after_first)


def _fit(v: np.ndarray, n: int) -> np.ndarray:
    """Centre-crop or zero-pad an odd-length pmf (centred on 0) to length n."""
    if len(v) == n:
        return v
    if len(v) > n:
        c = (len(v) - n) // 2
        return v[c:c + n]
    out = np.zeros(n)
    c = (n - len(v)) // 2
    out[c:c + len(v)] = v
    return out


def _pp_window(a: float, b: float, sign: int, s: float, dt: float,
               after_ppg, after_window: np.ndarray) -> np.ndarray:
    """
    Remaining home-diff distribution (on SUPPORT) through a PP window of s
    seconds that a PP goal ends, stepped every dt.

    a, b         : PP / SH team scoring rates (goals per second)
    sign         : +1 if home has the PP, -1 if away
    after_ppg(u) : diff distribution for the rest of regulation after a PP
                   goal u seconds into the window (odd length, centred on 0)
    after_window : same, for the window running out without a PP goal
    Shorthanded goals don't end the PP, so they just accumulate.
    """
    n_out = len(SUPPORT)
    final = np.zeros(n_out)
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
            # goal assumed mid-step
            final += _fit(np.convolve(pp_dist, after_ppg((j + 0.5) * h)), n_out)

        # PP continues: possibly a SH goal
        stay = q * (1.0 - p_pp)
        q = stay * (1.0 - p_sh)
        q[1:] += stay[:-1] * p_sh

    # PP expired without a PP goal: diff so far = -sign*k
    pp_dist = np.zeros(len(GRID))
    for k in ks[q > 0]:
        pp_dist += q[k] * _point_mass(-sign * k)
    final += _fit(np.convolve(pp_dist, after_window), n_out)
    return final


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
# down 2: 0.99 +/- 0.05 and 0.30 +/- 0.03.
# Refit 2026-11-05 on 2024-25 .. 2026-27 (to Nov 4):
#   down 2: 1.03 / 0.31 per 150 s of 6v5 (224 / 68 goals in 9.1 h); median
#     first pull 3:19 left (pull2_seconds 200).
#   down 1: 1.05 / 0.37, from games with the goalie out at a fixed time
#     (e.g. 1:30 left: a goal follows 57.7% +/- 2.2%, leader first 42.9%,
#     trailer first 14.8%). Exposure-based rates (0.86 / 0.37) run lower
#     because the leader's rate rises late (0.53 with 2-3 min left, ~1.0 in
#     the last 1:30). The same rates are used in the last minute.
#   re-pulls: after a goal the goalie comes out again a median ~25 s later;
#     a team that falls to down 2 in the last minute re-pulls only 22% of the
#     time (repull_delay 25, repull2_cutoff 60).
#   score effects, 3rd period, all strengths, goalie in: trailing team scores
#     2.59 / 60 down 1 and 3.02 down 2 vs 2.39 in a tied 3rd (x1.08, x1.26),
#     leading team ~unchanged; no clear goal effect in P1-P2 (sfx_seconds 1200).
#     Down 1 left at x1.0 by choice (2026-11-05); only down 2 is boosted.
#   pulled with a PP on (2026-11-12, down 1 + 2 pooled, 2024-25 .. 2026-27):
#     6v4 (trailing team's PP): leader EN 0.51 +/- 0.09, trailer 0.54 +/- 0.09
#     (2.9 h); 5v5 (trailing team shorthanded): leader EN 1.89 +/- 0.46,
#     trailer 0.11 +/- 0.11 (0.4 h), shrunk to 1.5 / 0.15. The plain PP
#     rates used before gave the leader ~0.03 at 6v4. Pull times on a PP
#     match 6v5 down 1 (median 2:00); down 2 ~25 s earlier, n = 66, not modelled.
#   With these, lead-1 / lead-2 states from 40:00 to 6:00 left are priced
#   within ~1.3 pts on the moneyline and ~1.5 pts on -1.5 / -2.5 on average.

@dataclass
class EndgameParams:
    e_lam_L: float = 1.05   # home endgame xG, home up 1 (empty-net goals)
    e_mu_L: float = 0.37    # away endgame xG, home up 1 (extra attacker)
    e_lam_T: float = 0.37   # home endgame xG, home down 1 (extra attacker)
    e_mu_T: float = 1.05    # away endgame xG, home down 1 (empty-net goals)
    window_seconds: float = 150.0   # e_* are xG over this many seconds pulled
    pull_seconds: float = 120.0     # pull-time model: assumed pull, down 1 (s left) ...
    roll_seconds: float = 30.0      # ... rolled back by this much once passed
    # two-goal pulls (pull-time model only)
    e2_lead: float = 1.03   # leading team xG / window, opponent down 2 and pulled
    e2_trail: float = 0.31  # trailing team xG / window
    pull2_seconds: float = 200.0    # assumed pull, down 2 (s left); 0 = never
    # final stretch, down 1 (pull-time model only): per-window rates once
    # final_seconds or less are left; final_seconds = 0 turns it off
    final_seconds: float = 60.0
    final_lead: float = 1.05
    final_trail: float = 0.37
    # score effects (pull-time model only): with sfx_seconds or less left and
    # no pull / PP, base rates are scaled by margin; sfx_seconds = 0 turns it off
    sfx_seconds: float = 1200.0
    sfx_lead1: float = 1.0
    sfx_trail1: float = 1.0
    sfx_lead2: float = 1.0
    sfx_trail2: float = 1.26
    # re-pulls (pull-time model only): after a goal the goalie is back in for
    # the faceoff and comes out again after ~repull_delay s (exponential; 0 =
    # at once); down 2 with repull2_cutoff s or less left, no re-pull
    repull_delay: float = 25.0
    repull2_cutoff: float = 60.0
    # pulled with a power play on (pull-time model only), xG / window, down 1
    # and 2 pooled; None for a pair = plain PP rates (the old behaviour)
    #   6v4: the trailing team is on the PP and pulls
    pp_pull_lead: float | None = 0.51   # leading (shorthanded) team, empty net
    pp_pull_trail: float | None = 0.54  # trailing team
    #   5v5: the trailing team is shorthanded and pulls anyway
    sh_pull_lead: float | None = 1.5    # leading (PP) team, empty net
    sh_pull_trail: float | None = 0.15  # trailing team


def pulled_pp_rates(eg: EndgameParams, pp_team: str) -> dict[int, tuple[float, float]]:
    """
    Home margin -> (home, away) xG per window_seconds with the trailing goalie
    pulled while `pp_team` has a power play: 6v4 if the trailing team has
    it, 5v5 into an empty net if the leading team has it. Margins whose pair
    is None are left out (they keep the PP rates).
    """
    out = {}
    for margin in (1, -1, 2, -2):
        trailing = "home" if margin < 0 else "away"
        lead, trail = ((eg.pp_pull_lead, eg.pp_pull_trail) if pp_team == trailing
                       else (eg.sh_pull_lead, eg.sh_pull_trail))
        if lead is None or trail is None:
            continue
        out[margin] = (lead, trail) if margin > 0 else (trail, lead)
    return out


def score_effects(eg: EndgameParams) -> dict[int, tuple[float, float]]:
    """Home margin -> (home, away) base-rate multipliers inside the score-effects window."""
    return {1: (eg.sfx_lead1, eg.sfx_trail1), -1: (eg.sfx_trail1, eg.sfx_lead1),
            2: (eg.sfx_lead2, eg.sfx_trail2), -2: (eg.sfx_trail2, eg.sfx_lead2)}


def _sfx_on(eg: EndgameParams) -> bool:
    return eg.sfx_seconds > 0 and any(x != 1.0 for x in (
        eg.sfx_lead1, eg.sfx_trail1, eg.sfx_lead2, eg.sfx_trail2))


def pulled_rates(eg: EndgameParams, final: bool = False) -> dict[int, tuple[float, float]]:
    """Home margin -> (home, away) xG per window_seconds with the trailing goalie pulled."""
    lead, trail = (eg.final_lead, eg.final_trail) if final else (eg.e_lam_L, eg.e_mu_L)
    tlead, ttrail = (eg.final_lead, eg.final_trail) if final else (eg.e_mu_T, eg.e_lam_T)
    return {1: (lead, trail), -1: (ttrail, tlead),
            2: (eg.e2_lead, eg.e2_trail), -2: (eg.e2_trail, eg.e2_lead)}


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
#     endgame rates (per window_seconds), which REPLACE the base rates; with
#     a power play on, the 6v4 / 5v5 pulled-PP rates replace the PP rates
#     (pulled_pp_rates);
#   - in the last final_seconds (1:00) down 1, both teams use the final_*
#     rates instead;
#   - a goal ends the pull (the new margin may be pulled again right away
#     if its assumed pull time has passed, e.g. down 2 -> down 1 at 1:30);
#   - a goalie not yet pulled is assumed to come out at `assumed_pull_time`:
#     2:00 left down 1 by default; once 2:00 passes without a pull, 1:30;
#     then 1:00, and so on (down 2: 3:20, 2:50, ...). The assumption applies
#     to any deficit, current or future;
#   - after a goal the goalie is back in for the faceoff and re-pulls after
#     ~repull_delay s; down 2 in the last repull2_cutoff s it stays in;
#   - score effects: in the last sfx_seconds (the 3rd period), with the
#     goalie in and no PP, base rates at +/-1 and +/-2 are scaled by the
#     sfx_* multipliers (the chain then starts at the 3rd period).
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


def _step_probs(rh: np.ndarray, ra: np.ndarray, h: float):
    """Per-step (home goal, away goal) probabilities from rates over h seconds."""
    tot = rh + ra
    pg = 1.0 - np.exp(-tot * h)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.where(tot > 0, pg * rh / tot, 0.0),
                np.where(tot > 0, pg * ra / tot, 0.0))


def _goalie_moves(P: np.ndarray, s: float, h: float, masks, tau1: float, tau2: float,
                  eg: EndgameParams) -> None:
    """
    Goalie pulls at the start of a chain step (in place). P[0] = in net,
    P[1] = pulled, P[2] = in net after a goal (back for the faceoff); masks =
    (down-1 cells, down-2 cells, other cells) over P's trailing axes.
    Once a margin's assumed pull time has passed, in-net games pull at once
    and post-goal games re-pull at rate 1 / repull_delay (at once if 0); a
    team that falls to down 2 with repull2_cutoff or less left stays in net.
    """
    q = 1.0 - np.exp(-h / eg.repull_delay) if eg.repull_delay > 0 else 1.0
    one, two, other = masks
    for mask, tau_m, cut in ((one, tau1, 0.0), (two, tau2, eg.repull2_cutoff)):
        if s <= tau_m + 1e-9:
            P[1][:, mask] += P[0][:, mask]
            P[0][:, mask] = 0.0
            if s > cut + 1e-9:
                mv = P[2][:, mask] * q
                P[1][:, mask] += mv
                P[2][:, mask] -= mv
        else:
            P[0][:, mask] += P[2][:, mask]
            P[2][:, mask] = 0.0
    P[0][:, other] += P[2][:, other]
    P[2][:, other] = 0.0


def _route_goals(P2: np.ndarray, G_h: np.ndarray, G_a: np.ndarray,
                 pp_team: str | None, penalty: str, reset: int | None) -> None:
    """
    Add one chain step's goals to P2 = P[2] (goalie back in for the faceoff),
    in place. G_h / G_a: home / away goal mass, already moved to the new
    score, over (PP state, ...). PP states: 0 = none, 1 = the PP in progress,
    2.. = a double minor's fresh second minor, by steps left. A PP goal ends
    a minor (state 0) or, with `reset` set (first half of a double minor),
    starts the fresh minor at index `reset`; a major and shorthanded goals
    leave the PP as it is.
    """
    if pp_team is None:
        P2 += G_h + G_a
        return
    G_pp, G_sh = (G_h, G_a) if pp_team == "home" else (G_a, G_h)
    P2 += G_sh
    if penalty == "major":
        P2 += G_pp
    elif reset is None:
        P2[0] += G_pp.sum(axis=0)
    else:
        P2[0] += G_pp[0] + G_pp[2:].sum(axis=0)
        P2[reset] += G_pp[1]


def _tick_reset(P: np.ndarray) -> None:
    """Count the fresh-minor clock (PP states 2..) down one step, in place."""
    P[:, 0] += P[:, 2]
    P[:, 2:-1] = P[:, 3:].copy()
    P[:, -1] = 0.0


def _chain_setup(T: float, eg: EndgameParams, d: int, pulled: bool, pull_at,
                 pp_team, pp_seconds: float, penalty: str):
    """Shared start of the two pull-time chains."""
    if penalty not in PENALTIES:
        raise ValueError(f"penalty must be one of {PENALTIES}")
    tau1, tau2 = pull_at if pull_at is not None else (
        assumed_pull_time(T, eg, 1), assumed_pull_time(T, eg, 2))
    sfx = _sfx_on(eg)
    tau = max(tau1, tau2, eg.sfx_seconds if sfx else 0.0)
    pulled = pulled and abs(d) in (1, 2)
    pp_on = pp_team is not None and pp_seconds > 0
    pp_end = T - pp_seconds if pp_on else -1.0     # seconds left when PP expires
    # Chain from now if the goalie is out or the PP runs into the endgame,
    # otherwise from the earliest assumed pull time (or score-effects start).
    S0 = T if (pulled or (pp_on and pp_end < min(T, tau))) else min(T, tau)
    n = max(int(np.ceil(S0 - 1e-9)), 1)
    h = S0 / n if S0 > 0 else 1.0
    # a double minor still in its first half when the chain starts: clock its
    # possible fresh second minor in steps (PP states 2 .. n_reset + 1)
    n_reset = (int(round(DOUBLE_HALF / h)) if pp_on and S0 >= T and penalty == "double"
               and pp_seconds > DOUBLE_HALF else 0)
    return tau1, tau2, sfx, pulled, pp_on, pp_end, S0, n, h, n_reset


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
    penalty : "minor" / "double" / "major" (see PENALTIES); pp_seconds is the
              PP's whole time left and may run past the end of regulation
    Other arguments as in remaining_diff_dist.
    """
    T = max(t, 0.0) * REG_SECONDS
    if T <= 0:
        return {d: 1.0}
    tau1, tau2, sfx, pulled, pp_on, pp_end, S0, n, h, n_reset = _chain_setup(
        T, eg, d, pulled, pull_at, pp_team, pp_seconds, penalty)

    n_m = 2 * _EG_PAD + 1
    margins = np.arange(d - _EG_PAD, d + _EG_PAD + 1)
    n_pp = 2 + n_reset
    P = np.zeros((3, n_pp, n_m))          # [goalie state, PP state, margin]
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
    # rates[goalie state, PP state] -> (home, away) per margin
    rh = np.full((3, n_pp, n_m), lam_s)
    ra = np.full((3, n_pp, n_m), mu_s)
    up1, dn1 = margins == 1, margins == -1
    up2, dn2 = margins == 2, margins == -2
    if pp_on:
        if pp_team == "home":
            rh[:, 1:], ra[:, 1:] = lam_s * pp_mult * st_home, mu_s * sh_mult * st_away
        else:
            rh[:, 1:], ra[:, 1:] = lam_s * sh_mult * st_home, mu_s * pp_mult * st_away
    one, two = up1 | dn1, up2 | dn2
    masks = (one, two, ~(one | two))

    steps = {}                            # (ph, pa) by (score-effects window, final stretch)
    for in_fx in (False, True):
        for final in (False, True):
            rh_f, ra_f = rh.copy(), ra.copy()
            if in_fx and sfx:
                for margin, (fh, fa) in score_effects(eg).items():
                    rh_f[0::2, 0, margins == margin] *= fh
                    ra_f[0::2, 0, margins == margin] *= fa
            for margin, (eh, ea) in pulled_rates(eg, final).items():
                rh_f[1, 0, margins == margin], ra_f[1, 0, margins == margin] = eh / W, ea / W
            if pp_on:                     # pulled during the PP: 6v4 / 5v5 rates
                for margin, (eh, ea) in pulled_pp_rates(eg, pp_team).items():
                    rh_f[1, 1:, margins == margin] = eh / W
                    ra_f[1, 1:, margins == margin] = ea / W
            steps[in_fx, final] = _step_probs(rh_f, ra_f, h)
    for j in range(n):
        s = S0 - j * h                    # seconds left at the start of this step
        if pp_on and s <= pp_end + 1e-9:  # PP expires (a fresh minor ends before it)
            P[:, 0] += P[:, 1:].sum(axis=1)
            P[:, 1:] = 0.0
            if n_reset:                   # drop the fresh-minor clock
                P = P[:, :2].copy()
                steps = {k: (a[:, :2], b[:, :2]) for k, (a, b) in steps.items()}
                n_reset = 0
        ph, pa = steps[s <= eg.sfx_seconds + 1e-9, s <= eg.final_seconds + 1e-9]
        _goalie_moves(P, s, h, masks, tau1, tau2, eg)
        home_g, away_g = P * ph, P * pa
        P = P - home_g - away_g
        # any goal ends the pull (goalie back in for the faceoff); shift margins
        G_h = np.zeros_like(P[2])
        G_a = np.zeros_like(P[2])
        G_h[:, 1:] = home_g[:, :, :-1].sum(axis=0)
        G_a[:, :-1] = away_g[:, :, 1:].sum(axis=0)
        reset = (n_reset + 1 if n_reset and s - 0.5 * h - pp_end > DOUBLE_HALF
                 else None)               # goal assumed mid-step
        _route_goals(P[2], G_h, G_a, pp_team if pp_on else None, penalty, reset)
        if n_reset:
            _tick_reset(P)

    final = P.sum(axis=(0, 1))
    return {int(k): float(v) for k, v in zip(margins, final) if v > 0}


def endgame_goals_dist(lam: float, mu: float, t: float, d: int,
                       eg: EndgameParams, pulled: bool = False,
                       pull_at: tuple[float, float] | None = None,
                       pp_team: str | None = None, pp_seconds: float = 0.0,
                       penalty: str = "minor",
                       pp_mult: float = 2.3, sh_mult: float = 0.27,
                       st_home: float = 1.0, st_away: float = 1.0,
                       max_goals: int = 16, **_ignored) -> np.ndarray:
    """
    Joint distribution of the (home, away) goals still to come in regulation,
    P[h, a], under the pull-time endgame model (for live totals).

    Same endgame chain as endgame_margin_dist, but over each team's goals so
    empty-net and extra-attacker goals land in the total. Before the endgame
    the two teams' goals are independent Poissons, with a power play counted
    at its expected length (a minor ends on a PP goal, a double minor's
    first-half PP goal starts a fresh 2:00), so the implied margin
    distribution differs very slightly from endgame_margin_dist's exact one.
    """
    G = max_goals
    k = np.arange(G + 1)
    T = max(t, 0.0) * REG_SECONDS
    if T <= 0:
        out = np.zeros((G + 1, G + 1))
        out[0, 0] = 1.0
        return out
    tau1, tau2, sfx, pulled, pp_on, pp_end, S0, n, h, n_reset = _chain_setup(
        T, eg, d, pulled, pull_at, pp_team, pp_seconds, penalty)

    lam_s, mu_s = lam / REG_SECONDS, mu / REG_SECONDS
    if pp_team == "home":
        pp_h, pp_a = lam_s * pp_mult * st_home, mu_s * sh_mult * st_away
    else:
        pp_h, pp_a = lam_s * sh_mult * st_home, mu_s * pp_mult * st_away

    n_pp = 2 + n_reset
    P = np.zeros((3, n_pp, G + 1, G + 1))   # [goalie state, PP state, home goals, away goals]
    if S0 < T:
        L = T - S0                       # pre-endgame stretch; any PP ends inside it
        h_mean, a_mean = lam_s * L, mu_s * L
        if pp_on:
            s = min(pp_seconds, L)
            a_rate = pp_h if pp_team == "home" else pp_a          # PP team's rate
            if penalty == "major" or a_rate <= 0:
                s_eff = s
            elif penalty == "double" and s > DOUBLE_HALF:
                # first half to the first PP goal (or its end), then a fresh minor
                s_eff = ((1 - np.exp(-a_rate * (s - DOUBLE_HALF)))
                         + (1 - np.exp(-a_rate * DOUBLE_HALF))) / a_rate
            else:
                s_eff = (1 - np.exp(-a_rate * s)) / a_rate
            h_mean = pp_h * s_eff + lam_s * (L - s_eff)
            a_mean = pp_a * s_eff + mu_s * (L - s_eff)
        P0 = np.outer(poisson.pmf(k, h_mean), poisson.pmf(k, a_mean))
        P[0, 0] = P0 / P0.sum()
    else:
        P[int(pulled), int(pp_on), 0, 0] = 1.0
    if S0 <= 0:
        return P.sum(axis=(0, 1))

    W = eg.window_seconds
    M = d + k[:, None] - k[None, :]      # margin in each cell
    rh = np.full((3, n_pp, G + 1, G + 1), lam_s)
    ra = np.full((3, n_pp, G + 1, G + 1), mu_s)
    if pp_on:
        rh[:, 1:], ra[:, 1:] = pp_h, pp_a
    one, two = np.abs(M) == 1, np.abs(M) == 2
    masks = (one, two, ~(one | two))

    steps = {}
    for in_fx in (False, True):
        for final in (False, True):
            rh_f, ra_f = rh.copy(), ra.copy()
            if in_fx and sfx:
                for margin, (fh, fa) in score_effects(eg).items():
                    mask = M == margin
                    for g in (0, 2):
                        rh_f[g, 0][mask] *= fh
                        ra_f[g, 0][mask] *= fa
            for margin, (eh, ea) in pulled_rates(eg, final).items():
                mask = M == margin
                rh_f[1, 0][mask], ra_f[1, 0][mask] = eh / W, ea / W
            if pp_on:
                for margin, (eh, ea) in pulled_pp_rates(eg, pp_team).items():
                    mask = M == margin
                    rh_f[1, 1:][:, mask], ra_f[1, 1:][:, mask] = eh / W, ea / W
            steps[in_fx, final] = _step_probs(rh_f, ra_f, h)
    for j in range(n):
        s = S0 - j * h
        if pp_on and s <= pp_end + 1e-9:
            P[:, 0] += P[:, 1:].sum(axis=1)
            P[:, 1:] = 0.0
            if n_reset:
                P = P[:, :2].copy()
                steps = {k_: (a[:, :2], b[:, :2]) for k_, (a, b) in steps.items()}
                n_reset = 0
        ph, pa = steps[s <= eg.sfx_seconds + 1e-9, s <= eg.final_seconds + 1e-9]
        _goalie_moves(P, s, h, masks, tau1, tau2, eg)
        home_g, away_g = P * ph, P * pa
        P = P - home_g - away_g
        G_h = np.zeros_like(P[2])
        G_a = np.zeros_like(P[2])
        G_h[:, 1:, :] = home_g[:, :, :-1, :].sum(axis=0)
        G_a[:, :, 1:] = away_g[:, :, :, :-1].sum(axis=0)
        reset = (n_reset + 1 if n_reset and s - 0.5 * h - pp_end > DOUBLE_HALF
                 else None)
        _route_goals(P[2], G_h, G_a, pp_team if pp_on else None, penalty, reset)
        if n_reset:
            _tick_reset(P)
    return P.sum(axis=(0, 1))


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
    no_pull = EndgameParams(pull_seconds=0, pull2_seconds=0, sfx_seconds=0)
    eg_dists = [endgame_margin_dist(lam, mu, 0.1, 1, eg),
                endgame_margin_dist(lam, mu, 0.05, -1, eg, pulled=True, pp_team="home",
                                    pp_seconds=120)]
    print(f"  pull-time model sums to 1:          "
          f"{all(abs(sum(x.values()) - 1) < 1e-9 for x in eg_dists)}")
    print(f"  never pulling == base model:        "
          f"{max(abs(endgame_margin_dist(lam, mu, 0.1, 1, no_pull).get(k, 0) - v) for k, v in regulation_margin_dist(lam, mu, 0.1, 1).items()) < 1e-9}")
    print(f"  down-1 pulls at 121/120/91/90/30s:  "
          f"{[assumed_pull_time(x, eg) for x in (121, 120, 91, 90, 30)]}")
    print(f"  down-2 pulls at 230/200/171/170s:   "
          f"{[assumed_pull_time(x, eg, 2) for x in (230, 200, 171, 170)]}")
    gap = 0.0
    for tt, dd in ((0.6, 1), (0.3, -2), (0.05, -1)):
        mg = endgame_margin_dist(lam, mu, tt, dd, eg)
        J = endgame_goals_dist(lam, mu, tt, dd, eg, max_goals=20)
        hh, aa = np.indices(J.shape)
        for k, v in mg.items():
            gap = max(gap, abs(J[hh - aa + dd == k].sum() - v))
    print(f"  totals chain margins == margin chain: {gap < 1e-6}")

    # --- Double minors ---------------------------------------------------
    print("\nDouble minor (tied, start of 2nd, home PP win-equiv):")
    pp_we = {}
    for kind, secs in (("minor", 120), ("double", 240), ("major", 240)):
        pp_we[kind] = in_game_probs_pp(lam, mu, t2, 0, "home", secs, kind).win_equiv
        print(f"  {kind:<7} {secs}s: {pp_we[kind]:.4f}")

    def _gap(a, b):
        return max(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))

    second_half = all(_gap(f(penalty="double"), f(penalty="minor")) < 1e-12 for f in (
        lambda **kw: regulation_margin_dist(lam, mu, 0.5, 0, pp_team="away", pp_seconds=100, **kw),
        lambda **kw: endgame_margin_dist(lam, mu, 0.05, 1, eg, pp_team="home",
                                         pp_seconds=100, **kw)))
    neutral_dbl = _gap(regulation_margin_dist(lam, mu, 0.5, 1),
                       regulation_margin_dist(lam, mu, 0.5, 1, pp_team="home", pp_seconds=240,
                                              penalty="double", pp_mult=1.0, sh_mult=1.0))
    # with pulled / score-effect rates equal to the base rates, the chain must
    # reproduce the Skellam double minor
    r150 = 3.0 * 150 / REG_SECONDS
    flat = EndgameParams(r150, r150, r150, r150, e2_lead=r150, e2_trail=r150,
                         final_lead=r150, final_trail=r150, sfx_seconds=0,
                         pp_pull_lead=None, pp_pull_trail=None,      # PP rates
                         sh_pull_lead=None, sh_pull_trail=None)      # while pulled
    chain_gap = max(_gap(
        endgame_margin_dist(3.0, 3.0, T / REG_SECONDS, dd, flat, pull_at=(T, T),
                            pp_team=team, pp_seconds=secs, penalty="double"),
        regulation_margin_dist(3.0, 3.0, T / REG_SECONDS, dd, pp_team=team,
                               pp_seconds=secs, penalty="double"))
        for T, dd, secs, team in ((1100, 0, 240, "home"), (180, -1, 230, "away")))
    J = endgame_goals_dist(lam, mu, 0.2, 1, eg, pp_team="away", pp_seconds=200,
                           penalty="double", max_goals=20)
    hh, aa = np.indices(J.shape)
    mg = endgame_margin_dist(lam, mu, 0.2, 1, eg, pp_team="away", pp_seconds=200,
                             penalty="double")
    tot_gap = max(abs(J[hh - aa + 1 == k].sum() - v) for k, v in mg.items())
    print(f"  minor < double < major:             {pp_we['minor'] < pp_we['double'] < pp_we['major']}")
    print(f"  second half (<= 2:00) == minor:     {second_half}")
    print(f"  multipliers=1 matches base:         {neutral_dbl < 1e-4}  ({neutral_dbl:.1e})")
    print(f"  chain == Skellam at base rates:     {chain_gap < 1e-3}  ({chain_gap:.1e})")
    print(f"  totals chain margins == margin chain: {tot_gap < 1e-6}")

    # --- Pulled with a power play on --------------------------------------
    print("\nPulled with a PP on (home win-equiv, plain PP rates -> 6v4 / 5v5 rates):")
    from dataclasses import replace as _replace
    old_eg = _replace(eg, pp_pull_lead=None, pp_pull_trail=None,
                      sh_pull_lead=None, sh_pull_trail=None)
    pp_states = [("home down 1, 1:30 left, home PP, pulled (6v4)", 90, -1, "home"),
                 ("home up 1, 1:30 left, home PP, away pulled (5v5)", 90, 1, "home"),
                 ("home up 1, 1:00 left, away PP, away pulled (6v4)", 60, 1, "away")]
    for label, T, dd, team in pp_states:
        a, b = (_outcomes_from_margins(endgame_margin_dist(
            lam, mu, T / REG_SECONDS, dd, e, pulled=True, pp_team=team, pp_seconds=60), 0.5)
            for e in (old_eg, eg))
        print(f"  {label:<50} {a.win_equiv:.3f} -> {b.win_equiv:.3f}")
    # rates set to exactly the PP rates must reproduce the plain-PP chain
    W = eg.window_seconds
    pp_h, sh_a = lam / REG_SECONDS * 2.3 * W, mu / REG_SECONDS * 0.27 * W
    as_pp = _replace(eg, pp_pull_lead=sh_a, pp_pull_trail=pp_h,   # home trails, home PP
                     sh_pull_lead=pp_h, sh_pull_trail=sh_a)       # home leads, home PP
    wiring = max(_gap(
        endgame_margin_dist(lam, mu, 0.03, dd, as_pp, pulled=True, pp_team="home", pp_seconds=60),
        endgame_margin_dist(lam, mu, 0.03, dd, old_eg, pulled=True, pp_team="home", pp_seconds=60))
        for dd in (-2, -1, 1, 2))
    J = endgame_goals_dist(lam, mu, 0.025, -1, eg, pulled=True, pp_team="home",
                           pp_seconds=60, max_goals=20)
    hh, aa = np.indices(J.shape)
    mg = endgame_margin_dist(lam, mu, 0.025, -1, eg, pulled=True, pp_team="home", pp_seconds=60)
    tot_gap = max(abs(J[hh - aa - 1 == k].sum() - v) for k, v in mg.items())
    print(f"  rates == PP rates reproduces old:   {wiring < 1e-12}")
    print(f"  totals chain margins == margin chain: {tot_gap < 1e-6}")
