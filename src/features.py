"""
Chronological feature builder.

Single pass over history (oldest -> newest) that, for every match, records
the *pre-match* state of each team (Elo rating, form over the last 10
games, goal-difference form) and then updates that state with the match
result. This guarantees no data leakage: every feature used to predict a
match is only built from information available strictly before that match
was played.
"""
from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

from elo import EloEngine, HOME_ADVANTAGE

FORM_WINDOW = 10


class FormTracker:
    """Rolling points-per-game and goal-difference form, last N matches."""

    def __init__(self, window: int = FORM_WINDOW):
        self.window = window
        self.points: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.goal_diff: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def get(self, team: str) -> tuple[float, float]:
        pts = self.points[team]
        gds = self.goal_diff[team]
        ppg = sum(pts) / len(pts) if pts else 1.0  # neutral prior: 1 pt/game
        avg_gd = sum(gds) / len(gds) if gds else 0.0
        return ppg, avg_gd

    def update(self, team: str, points: int, goal_diff: int) -> None:
        self.points[team].append(points)
        self.goal_diff[team].append(goal_diff)


def build_dataset(results: pd.DataFrame, cutoff_date: str | None = None):
    """Returns (feature_df, fitted EloEngine, fitted FormTracker) using all
    played matches strictly before cutoff_date."""
    df = results.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    if cutoff_date is not None:
        df = df[df["date"] < pd.Timestamp(cutoff_date)]
    df = df.sort_values("date").reset_index(drop=True)

    elo = EloEngine()
    form = FormTracker()
    rows = []

    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        hs, as_ = row.home_score, row.away_score
        neutral = bool(row.neutral)

        home_elo = elo.get(home)
        away_elo = elo.get(away)
        home_ppg, home_gd_form = form.get(home)
        away_ppg, away_gd_form = form.get(away)

        home_adv = 0.0 if neutral else HOME_ADVANTAGE
        elo_diff = (home_elo + home_adv) - away_elo

        if hs > as_:
            outcome = "H"
            home_pts, away_pts = 3, 0
        elif hs < as_:
            outcome = "A"
            home_pts, away_pts = 0, 3
        else:
            outcome = "D"
            home_pts, away_pts = 1, 1

        rows.append(
            {
                "date": row.date,
                "home_team": home,
                "away_team": away,
                "elo_diff": elo_diff,
                "neutral": int(neutral),
                "home_ppg_form": home_ppg,
                "away_ppg_form": away_ppg,
                "form_ppg_diff": home_ppg - away_ppg,
                "form_gd_diff": home_gd_form - away_gd_form,
                "outcome": outcome,
            }
        )

        # Update state AFTER recording features (no leakage).
        elo.process_match(row.date, home, away, hs, as_, row.tournament, neutral)
        form.update(home, home_pts, int(hs - as_))
        form.update(away, away_pts, int(as_ - hs))

    return pd.DataFrame(rows), elo, form


def fixture_features(elo: EloEngine, form: FormTracker, home: str, away: str, neutral: bool):
    home_elo = elo.get(home)
    away_elo = elo.get(away)
    home_ppg, home_gd_form = form.get(home)
    away_ppg, away_gd_form = form.get(away)
    home_adv = 0.0 if neutral else HOME_ADVANTAGE
    elo_diff = (home_elo + home_adv) - away_elo
    return {
        "elo_diff": elo_diff,
        "neutral": int(neutral),
        "home_ppg_form": home_ppg,
        "away_ppg_form": away_ppg,
        "form_ppg_diff": home_ppg - away_ppg,
        "form_gd_diff": home_gd_form - away_gd_form,
        "home_elo": home_elo,
        "away_elo": away_elo,
    }


if __name__ == "__main__":
    results = pd.read_csv("data/results.csv")
    feat_df, elo, form = build_dataset(results, cutoff_date="2026-07-14")
    print(feat_df.tail())
    print(feat_df["outcome"].value_counts(normalize=True))
    print(fixture_features(elo, form, "Argentina", "England", neutral=True))
    print(fixture_features(elo, form, "France", "Spain", neutral=True))
