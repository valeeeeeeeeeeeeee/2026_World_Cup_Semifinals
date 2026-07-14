"""
World Football Elo rating engine.

Replays every international match in data/results.csv in chronological
order and maintains a running Elo rating per national team. Methodology
follows the well-established World Football Elo Ratings approach
(eloratings.net):

  - K-factor depends on the importance of the competition.
  - The rating change is scaled by the goal difference of the match
    (a 4-0 win moves ratings more than a 1-0 win).
  - A neutral-venue match carries no home-advantage bonus; a match at a
    declared home venue gets a fixed home-advantage bump to the expected
    score calculation only (ratings themselves are not adjusted for it).

Only matches strictly before a given cutoff date are used, so ratings can
be reconstructed as they stood right before any match in history
(including the 2026 semifinals).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

INITIAL_RATING = 1500.0
HOME_ADVANTAGE = 100.0  # Elo points added to the home team's expected-score calc

# Competition importance weights (World Football Elo convention).
K_WORLD_CUP_FINALS = 60
K_CONTINENTAL_OR_INTERCONFEDERATION = 50
K_QUALIFIERS_AND_MAJOR_CUPS = 40
K_FRIENDLY = 20

CONTINENTAL_OR_MAJOR_KEYWORDS = (
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "UEFA Euro",
    "Gold Cup",
    "CONCACAF Championship",
    "Confederations Cup",
    "Oceania Nations Cup",
    "Copa America",
)
QUALIFIER_OR_NATIONS_LEAGUE_KEYWORDS = ("qualification", "Nations League")


def match_k_factor(tournament: str) -> int:
    if tournament == "FIFA World Cup":
        return K_WORLD_CUP_FINALS
    if any(key in tournament for key in CONTINENTAL_OR_MAJOR_KEYWORDS):
        return K_CONTINENTAL_OR_INTERCONFEDERATION
    if any(key in tournament for key in QUALIFIER_OR_NATIONS_LEAGUE_KEYWORDS):
        return K_QUALIFIERS_AND_MAJOR_CUPS
    if tournament == "Friendly":
        return K_FRIENDLY
    return K_QUALIFIERS_AND_MAJOR_CUPS


def goal_diff_multiplier(goal_diff: int) -> float:
    """Standard World Football Elo goal-difference scaling."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


@dataclass
class EloEngine:
    ratings: dict[str, float] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def get(self, team: str) -> float:
        return self.ratings.get(team, INITIAL_RATING)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** (-(rating_a - rating_b) / 400.0))

    def process_match(
        self,
        date,
        home_team: str,
        away_team: str,
        home_score: float,
        away_score: float,
        tournament: str,
        neutral: bool,
    ) -> None:
        home_elo_before = self.get(home_team)
        away_elo_before = self.get(away_team)

        home_adj = home_elo_before + (0.0 if neutral else HOME_ADVANTAGE)
        exp_home = self.expected_score(home_adj, away_elo_before)
        exp_away = 1.0 - exp_home

        if home_score > away_score:
            actual_home, actual_away = 1.0, 0.0
        elif home_score < away_score:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        k = match_k_factor(tournament)
        g = goal_diff_multiplier(int(home_score - away_score))

        delta_home = k * g * (actual_home - exp_home)
        delta_away = k * g * (actual_away - exp_away)

        self.ratings[home_team] = home_elo_before + delta_home
        self.ratings[away_team] = away_elo_before + delta_away

        self.history.append(
            {
                "date": date,
                "home_team": home_team,
                "away_team": away_team,
                "home_elo_pre": home_elo_before,
                "away_elo_pre": away_elo_before,
                "home_elo_post": self.ratings[home_team],
                "away_elo_post": self.ratings[away_team],
                "neutral": neutral,
                "tournament": tournament,
                "exp_home": exp_home,
            }
        )


def build_elo_ratings(results: pd.DataFrame, cutoff_date: str | None = None) -> EloEngine:
    """Replay all played matches (non-null scores) up to cutoff_date (exclusive)."""
    df = results.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    if cutoff_date is not None:
        df = df[df["date"] < pd.Timestamp(cutoff_date)]
    df = df.sort_values("date")

    engine = EloEngine()
    for row in df.itertuples(index=False):
        engine.process_match(
            date=row.date,
            home_team=row.home_team,
            away_team=row.away_team,
            home_score=row.home_score,
            away_score=row.away_score,
            tournament=row.tournament,
            neutral=bool(row.neutral),
        )
    return engine


if __name__ == "__main__":
    results = pd.read_csv("data/results.csv")
    engine = build_elo_ratings(results, cutoff_date="2026-07-14")
    top20 = sorted(engine.ratings.items(), key=lambda kv: -kv[1])[:20]
    print("Top 20 Elo ratings as of 2026-07-13 (eve of the semifinals):")
    for i, (team, rating) in enumerate(top20, 1):
        print(f"{i:2d}. {team:20s} {rating:7.1f}")

    print()
    for team in ["Argentina", "England", "France", "Spain"]:
        print(f"{team:12s} Elo = {engine.get(team):.1f}")
