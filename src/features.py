"""
Chronological feature builder.

Single pass over history (oldest -> newest) that, for every match, records
the *pre-match* state of each team and then updates that state with the
match result. This guarantees no data leakage: every feature used to
predict a match is only built from information available strictly before
that match was played.

Features recorded per match:
  - elo_diff     : (home Elo + home advantage) - away Elo
  - elo_sum      : home Elo + away Elo (overall strength/level of the tie)
  - neutral      : 1 if played at a neutral venue
  - form_ppg_diff: rolling 10-match points-per-game, home minus away
  - form_gf_diff : rolling 10-match goals-scored/game, home minus away
  - form_ga_diff : rolling 10-match goals-conceded/game, home minus away
  - rest_diff    : days since each team's last match, home minus away (capped)
  - k_weight     : competition-importance weight of the match (Elo K-factor)
"""
from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

from elo import EloEngine, HOME_ADVANTAGE, match_k_factor

FORM_WINDOW = 10
REST_CAP_DAYS = 120
REST_PRIOR_DAYS = 30
GOAL_FORM_PRIOR = 1.2  # neutral prior goals for/against per game


class FormTracker:
    """Rolling form over the last N matches: points, goals for, goals against,
    plus the date of each team's most recent match (for rest-day features)."""

    def __init__(self, window: int = FORM_WINDOW):
        self.window = window
        self.points: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.goals_for: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.goals_against: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.last_date: dict[str, pd.Timestamp] = {}

    def get(self, team: str) -> tuple[float, float, float]:
        pts = self.points[team]
        gf = self.goals_for[team]
        ga = self.goals_against[team]
        ppg = sum(pts) / len(pts) if pts else 1.0  # neutral prior: 1 pt/game
        avg_gf = sum(gf) / len(gf) if gf else GOAL_FORM_PRIOR
        avg_ga = sum(ga) / len(ga) if ga else GOAL_FORM_PRIOR
        return ppg, avg_gf, avg_ga

    def rest_days(self, team: str, match_date: pd.Timestamp) -> int:
        if team not in self.last_date:
            return REST_PRIOR_DAYS
        return min((match_date - self.last_date[team]).days, REST_CAP_DAYS)

    def update(self, team: str, points: int, gf: int, ga: int, match_date: pd.Timestamp) -> None:
        self.points[team].append(points)
        self.goals_for[team].append(gf)
        self.goals_against[team].append(ga)
        self.last_date[team] = match_date


FEATURE_COLS = [
    "elo_diff",
    "elo_sum",
    "neutral",
    "form_ppg_diff",
    "form_gf_diff",
    "form_ga_diff",
    "rest_diff",
    "k_weight",
]


def _row_features(elo, form, date, home, away, neutral):
    home_elo = elo.get(home)
    away_elo = elo.get(away)
    home_ppg, home_gf, home_ga = form.get(home)
    away_ppg, away_gf, away_ga = form.get(away)
    home_adv = 0.0 if neutral else HOME_ADVANTAGE
    return {
        "elo_diff": (home_elo + home_adv) - away_elo,
        "elo_sum": home_elo + away_elo,
        "neutral": int(neutral),
        "form_ppg_diff": home_ppg - away_ppg,
        "form_gf_diff": home_gf - away_gf,
        "form_ga_diff": home_ga - away_ga,
        "rest_diff": form.rest_days(home, date) - form.rest_days(away, date),
        "home_elo": home_elo,
        "away_elo": away_elo,
    }


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

        feats = _row_features(elo, form, row.date, home, away, neutral)

        if hs > as_:
            outcome, home_pts, away_pts = "H", 3, 0
        elif hs < as_:
            outcome, home_pts, away_pts = "A", 0, 3
        else:
            outcome, home_pts, away_pts = "D", 1, 1

        rows.append(
            {
                "date": row.date,
                "home_team": home,
                "away_team": away,
                "elo_diff": feats["elo_diff"],
                "elo_sum": feats["elo_sum"],
                "neutral": feats["neutral"],
                "form_ppg_diff": feats["form_ppg_diff"],
                "form_gf_diff": feats["form_gf_diff"],
                "form_ga_diff": feats["form_ga_diff"],
                "rest_diff": feats["rest_diff"],
                "k_weight": match_k_factor(row.tournament),
                "outcome": outcome,
            }
        )

        # Update state AFTER recording features (no leakage).
        elo.process_match(row.date, home, away, hs, as_, row.tournament, neutral)
        form.update(home, home_pts, int(hs), int(as_), row.date)
        form.update(away, away_pts, int(as_), int(hs), row.date)

    return pd.DataFrame(rows), elo, form


def fixture_features(elo: EloEngine, form: FormTracker, home: str, away: str,
                     neutral: bool, match_date: str, tournament: str = "FIFA World Cup"):
    """Feature vector for an upcoming (unplayed) fixture."""
    date = pd.Timestamp(match_date)
    feats = _row_features(elo, form, date, home, away, neutral)
    feats["k_weight"] = match_k_factor(tournament)
    return feats


if __name__ == "__main__":
    results = pd.read_csv("data/results.csv")
    feat_df, elo, form = build_dataset(results, cutoff_date="2026-07-14")
    print(feat_df.tail())
    print(feat_df["outcome"].value_counts(normalize=True))
    print(fixture_features(elo, form, "Argentina", "England", neutral=True, match_date="2026-07-15"))
    print(fixture_features(elo, form, "France", "Spain", neutral=True, match_date="2026-07-14"))
