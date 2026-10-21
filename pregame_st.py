"""
Special-teams ratings from the NHL pregame model (Documents/NHL/2025).

The pregame model rates each team's special-teams goals per game (PP + SH goals):

    spec goals (home) ~ Poisson(exp(INT_SPEC + HFA_SPEC + off_spec[home] - def_spec[away]))
    spec goals (away) ~ Poisson(exp(INT_SPEC            + off_spec[away] - def_spec[home]))

Here each side's matchup term becomes a multiplier on its scoring rate while
a power play is on (whether it is the PP or the SH team):

    f_home = exp(off_spec[home] - def_spec[away] + HFA_SPEC / 2)
    f_away = exp(off_spec[away] - def_spec[home] - HFA_SPEC / 2)

The home edge is split in half so an average matchup averages 1.0, matching
the league-level pp_mult / sh_mult the live model is calibrated to.

Ratings come from the newest dated row in priors_saved.csv (the daily
pipeline's output) or team_ratings_2026.csv (preseason), whichever is later.
Only the standard library is used, so the live UI's Python doesn't need pandas.
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

PREGAME_DIR = Path(r"C:/Users/Nolanator/Documents/NHL/2025")
RATING_FILES = [PREGAME_DIR / "data" / "priors_saved.csv",
                PREGAME_DIR / "data" / "team_ratings_2026.csv"]
HFA_SPEC = 0.063            # model/simulate.py HFA_S


def _load_team_list() -> list[str]:
    """(name, abbrev) in rating-index order: sorted(TEAM_LIST), as utils.team_index_df does."""
    spec = importlib.util.spec_from_file_location("_pregame_mappings",
                                                  PREGAME_DIR / "mappings.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    abbr = {name: a for a, name in mod.TEAM_MAP_NEWAPI.items()}
    return [(name, abbr.get(name, name[:3].upper())) for name in sorted(mod.TEAM_LIST)]


def _parse_date(s: str) -> date:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            pass
    raise ValueError(f"unrecognised date {s!r}")


@dataclass
class STRatings:
    as_of: date
    source: str
    teams: list[str]                # full names, in rating-index order (alphabetical)
    abbrevs: list[str]
    off_spec: list[float]           # centred to mean 0
    def_spec: list[float]

    def index(self, team: str) -> int:
        if team in self.abbrevs:
            return self.abbrevs.index(team)
        return self.teams.index(team)

    def factors(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        """(f_home, f_away): PP/SH scoring-rate multipliers for this matchup."""
        h, a = self.index(home), self.index(away)
        half = 0.0 if neutral else HFA_SPEC / 2
        f_home = math.exp(self.off_spec[h] - self.def_spec[a] + half)
        f_away = math.exp(self.off_spec[a] - self.def_spec[h] - half)
        return f_home, f_away


def load_st_ratings() -> STRatings:
    best = None
    for path in RATING_FILES:
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    d = _parse_date(row["date"])
                except (ValueError, KeyError):
                    continue
                if best is None or d >= best[0]:
                    best = (d, path.name, row)
    if best is None:
        raise FileNotFoundError(f"no special-teams ratings found in {PREGAME_DIR}")

    d, source, row = best
    off = [float(x) for x in ast.literal_eval(row["off_spec_mu"])]
    dfn = [float(x) for x in ast.literal_eval(row["def_spec_mu"])]
    teams = _load_team_list()
    if not (len(off) == len(dfn) == len(teams)):
        raise ValueError(f"{source}: {len(off)} ratings vs {len(teams)} teams")
    mo, md = sum(off) / len(off), sum(dfn) / len(dfn)
    return STRatings(d, source, [t[0] for t in teams], [t[1] for t in teams],
                     [x - mo for x in off], [x - md for x in dfn])


if __name__ == "__main__":
    r = load_st_ratings()
    print(f"Ratings as of {r.as_of} ({r.source})")
    for i in sorted(range(len(r.teams)), key=lambda i: -(r.off_spec[i] + r.def_spec[i])):
        print(f"  {r.abbrevs[i]:4s} off {r.off_spec[i]:+.3f}  def {r.def_spec[i]:+.3f}")
    fh, fa = r.factors("EDM", "SJS")
    print(f"EDM vs SJS (EDM home): f_home {fh:.3f}, f_away {fa:.3f}")
