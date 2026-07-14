"""
Final prediction pipeline for the 2026 World Cup semifinals:
    Argentina vs England   (Atlanta, 2026-07-15)
    France    vs Spain     (Arlington, 2026-07-14)

Combines two independent signals into a single "probability of advancing
to the final" per team:

  1. Poisson goal-model Monte Carlo simulation. Regulation time goals are
     drawn from each team's fitted attack/defense Poisson rates; drawn
     matches go to a lower-intensity extra-time draw, and matches still
     level after that are resolved by a penalty-shootout coin flip that is
     mildly tilted by the Elo gap (shootouts are close to random, but not
     perfectly so).
  2. The trained Win/Draw/Loss classifier's probabilities for a single
     90-minute match, with the drawn mass reassigned to each side using
     the same shootout-tilt logic (a knockout match cannot actually end
     level).

The two estimates are then averaged into the reported final probability.
Both signals are trained/fit using data strictly before 2026-07-14, so
none of the semifinal (or later) results leak into the model.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from elo import EloEngine
from features import build_dataset, fixture_features
from poisson_model import PoissonGoalModel

CUTOFF = "2026-07-14"
N_SIMS = 50_000
RNG = np.random.default_rng(42)

FIXTURES = [
    {"home": "France", "away": "Spain", "venue": "Arlington", "date": "2026-07-14"},
    {"home": "Argentina", "away": "England", "venue": "Atlanta", "date": "2026-07-15"},
]


def shootout_win_prob(elo_diff_raw: float) -> float:
    """Penalty shootouts are close to a coin flip; skill still nudges it.
    Dampened logistic (divisor 3x the normal Elo scale of 400)."""
    return 1.0 / (1.0 + 10 ** (-elo_diff_raw / 1200.0))


def simulate_match(goal_model: PoissonGoalModel, home: str, away: str, elo_diff_raw: float, n_sims: int = N_SIMS):
    mu_home = goal_model.expected_goals(home, away, neutral=True)
    mu_away = goal_model.expected_goals(away, home, neutral=True)

    home_reg = RNG.poisson(mu_home, size=n_sims)
    away_reg = RNG.poisson(mu_away, size=n_sims)

    home_wins = home_reg > away_reg
    away_wins = home_reg < away_reg
    draw_mask = home_reg == away_reg

    n_draw = int(draw_mask.sum())
    if n_draw:
        et_home = RNG.poisson(mu_home / 3.0, size=n_draw)  # 30 extra minutes
        et_away = RNG.poisson(mu_away / 3.0, size=n_draw)
        et_total_home = home_reg[draw_mask] + et_home
        et_total_away = away_reg[draw_mask] + et_away

        et_home_win = et_total_home > et_total_away
        et_away_win = et_total_home < et_total_away
        et_draw = et_total_home == et_total_away

        n_pens = int(et_draw.sum())
        pen_home_win = RNG.random(n_pens) < shootout_win_prob(elo_diff_raw)

        idx = np.where(draw_mask)[0]
        home_wins[idx[et_home_win]] = True
        away_wins[idx[et_away_win]] = True
        pen_idx = idx[et_draw]
        home_wins[pen_idx[pen_home_win]] = True
        away_wins[pen_idx[~pen_home_win]] = True

    return {
        "mu_home": mu_home,
        "mu_away": mu_away,
        "p_home_advance": float(home_wins.mean()),
        "p_away_advance": float(away_wins.mean()),
    }


def classifier_advance_prob(clf_bundle, feats: dict, elo_diff_raw: float):
    model = clf_bundle["model"]
    scaler = clf_bundle["scaler"]
    X = pd.DataFrame([{k: feats[k] for k in clf_bundle["features"]}])
    if scaler is not None:
        X = scaler.transform(X)
    probs = model.predict_proba(X)[0]
    class_probs = dict(zip(model.classes_, probs))
    p_home, p_draw, p_away = class_probs.get("H", 0.0), class_probs.get("D", 0.0), class_probs.get("A", 0.0)

    p_home_shootout = shootout_win_prob(elo_diff_raw)
    p_home_advance = p_home + p_draw * p_home_shootout
    p_away_advance = p_away + p_draw * (1 - p_home_shootout)
    return {"p_home_90": p_home, "p_draw_90": p_draw, "p_away_90": p_away, "p_home_advance": p_home_advance, "p_away_advance": p_away_advance}


def main():
    results = pd.read_csv("data/results.csv")

    # Elo engine and form tracker fitted on all matches before the cutoff.
    _, elo, form = build_dataset(results, cutoff_date=CUTOFF)
    goal_model = PoissonGoalModel().fit(results, cutoff_date=CUTOFF)
    clf_bundle = joblib.load("results/classifier.joblib")

    report = {}
    for fx in FIXTURES:
        home, away = fx["home"], fx["away"]
        feats = fixture_features(elo, form, home, away, neutral=True, match_date=fx["date"])
        elo_diff_raw = feats["home_elo"] - feats["away_elo"]

        mc = simulate_match(goal_model, home, away, elo_diff_raw)
        clf = classifier_advance_prob(clf_bundle, feats, elo_diff_raw)

        ensemble_home = (mc["p_home_advance"] + clf["p_home_advance"]) / 2.0
        ensemble_away = 1.0 - ensemble_home

        winner = home if ensemble_home >= 0.5 else away
        report[f"{home} vs {away}"] = {
            "venue": fx["venue"],
            "date": fx["date"],
            "elo": {home: round(feats["home_elo"], 1), away: round(feats["away_elo"], 1)},
            "poisson_expected_goals": {home: round(mc["mu_home"], 2), away: round(mc["mu_away"], 2)},
            "monte_carlo_advance_prob": {home: round(mc["p_home_advance"], 4), away: round(mc["p_away_advance"], 4)},
            "classifier_90min_prob": {"home_win": round(clf["p_home_90"], 4), "draw": round(clf["p_draw_90"], 4), "away_win": round(clf["p_away_90"], 4)},
            "classifier_advance_prob": {home: round(clf["p_home_advance"], 4), away: round(clf["p_away_advance"], 4)},
            "final_advance_probability": {home: round(ensemble_home, 4), away: round(ensemble_away, 4)},
            "predicted_finalist": winner,
        }

        print(f"\n=== {home} vs {away} ({fx['venue']}, {fx['date']}) ===")
        print(f"Elo:  {home} {feats['home_elo']:.0f}  |  {away} {feats['away_elo']:.0f}")
        print(f"Poisson expected goals: {home} {mc['mu_home']:.2f} - {mc['mu_away']:.2f} {away}")
        print(f"Monte Carlo advance prob:  {home} {mc['p_home_advance']:.1%}  |  {away} {mc['p_away_advance']:.1%}")
        print(f"Classifier advance prob:   {home} {clf['p_home_advance']:.1%}  |  {away} {clf['p_away_advance']:.1%}")
        print(f">>> FINAL PREDICTION: {home} {ensemble_home:.1%}  vs  {away} {ensemble_away:.1%}  -> {winner} advances")

    with open("results/semifinal_predictions.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nSaved -> results/semifinal_predictions.json")


if __name__ == "__main__":
    main()
