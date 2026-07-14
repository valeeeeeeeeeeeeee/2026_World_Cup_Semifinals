"""
Dixon-Coles style Poisson goal model.

Each team gets an "attack" and "defense" strength learned via an L2
regularized Poisson regression (sklearn PoissonRegressor) on a
team/opponent/home-flag design matrix built from every match goal tally:

    goals_for(team, opponent, is_home) ~ Poisson(exp(b0 + attack[team]
                                                       + defense[opponent]
                                                       + home * is_home))

Restricted to a recent window (2015-present) so the ratings reflect
current squads rather than being dragged down by teams' form from
50 years ago. L2 regularization keeps thinly-sampled teams close to
average instead of overfitting on a handful of matches.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder

WINDOW_START = "2015-01-01"


def _long_format(results: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    df = results.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[(df["date"] >= pd.Timestamp(WINDOW_START)) & (df["date"] < pd.Timestamp(cutoff_date))]

    home_rows = pd.DataFrame(
        {
            "team": df["home_team"],
            "opponent": df["away_team"],
            "is_home": np.where(df["neutral"], 0, 1),
            "goals": df["home_score"],
        }
    )
    away_rows = pd.DataFrame(
        {
            "team": df["away_team"],
            "opponent": df["home_team"],
            "is_home": 0,
            "goals": df["away_score"],
        }
    )
    return pd.concat([home_rows, away_rows], ignore_index=True)


class PoissonGoalModel:
    def __init__(self):
        self.encoder: OneHotEncoder | None = None
        self.model: PoissonRegressor | None = None
        self.teams: set[str] = set()

    def fit(self, results: pd.DataFrame, cutoff_date: str) -> "PoissonGoalModel":
        long_df = _long_format(results, cutoff_date)
        self.teams = set(long_df["team"]) | set(long_df["opponent"])

        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        team_dummies = self.encoder.fit_transform(long_df[["team"]])
        opp_dummies = self.encoder.transform(
            long_df[["opponent"]].rename(columns={"opponent": "team"})
        )

        from scipy.sparse import hstack, csr_matrix

        home_col = csr_matrix(long_df[["is_home"]].to_numpy(dtype=float))
        X = hstack([team_dummies, opp_dummies, home_col]).tocsr()
        y = long_df["goals"].to_numpy(dtype=float)

        # alpha picked via a 2015-2023 train / 2024-2026 validation split on
        # mean Poisson deviance (see notes in README) - light regularization
        # mainly to stabilise thinly-sampled minor nations.
        self.model = PoissonRegressor(alpha=0.0005, max_iter=3000, tol=1e-8)
        self.model.fit(X, y)
        return self

    def expected_goals(self, team: str, opponent: str, neutral: bool) -> float:
        from scipy.sparse import hstack, csr_matrix

        team_vec = self.encoder.transform(pd.DataFrame({"team": [team]}))
        opp_vec = self.encoder.transform(pd.DataFrame({"team": [opponent]}))
        home_val = csr_matrix(np.array([[0.0 if neutral else 1.0]]))
        X = hstack([team_vec, opp_vec, home_val]).tocsr()
        return float(self.model.predict(X)[0])


if __name__ == "__main__":
    results = pd.read_csv("data/results.csv")
    goal_model = PoissonGoalModel().fit(results, cutoff_date="2026-07-14")

    for home, away in [("Argentina", "England"), ("France", "Spain")]:
        mu_home = goal_model.expected_goals(home, away, neutral=True)
        mu_away = goal_model.expected_goals(away, home, neutral=True)
        print(f"{home} vs {away}: expected goals {mu_home:.2f} - {mu_away:.2f}")
