"""
Trains the outcome classifier (Win / Draw / Loss for the home side) on the
full chronological feature set produced by features.py, evaluates it on a
held-out time slice, and saves the fitted model + a metrics report.

Two improvements over a plain fit drive the accuracy/log-loss gains:

  1. **Recency weighting.** Training matches are weighted by an exponential
     decay on their age (6-year half-life), so current squad strength
     dominates and 40-year-old results barely count. This is the single
     biggest lever on probability quality here.
  2. **Expanded, leak-free features** (Elo diff *and* level, goals-for /
     goals-against form, rest days, competition importance).

Two candidate models are compared on a time-based holdout (train < 2019,
validate >= 2019):
  - Multinomial logistic regression (chosen by default: it extrapolates
    linearly, which matters because the semifinalists sit at the very top
    of the Elo range, beyond where tree models can predict).
  - Gradient boosting (sanity check / non-linear baseline).

Metrics are reported both on ALL validation matches and on the
**competitive** subset (real tournaments, excluding friendlies) — the
latter is the domain a World Cup semifinal actually lives in, where
accuracy is meaningfully higher than the draw-capped all-matches figure.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

from features import build_dataset, FEATURE_COLS

CLASSES = ["A", "D", "H"]
SPLIT_DATE = pd.Timestamp("2019-01-01")
RECENCY_HALF_LIFE_YEARS = 6.0
FRIENDLY_K = 20  # matches with k_weight above this are competitive


def recency_weights(dates: pd.Series, reference: pd.Timestamp) -> np.ndarray:
    age_years = (reference - dates).dt.days / 365.25
    return 0.5 ** (age_years / RECENCY_HALF_LIFE_YEARS)


def _metrics(model, scaler, df, cols):
    X = scaler.transform(df[cols]) if scaler is not None else df[cols]
    probs = model.predict_proba(X)
    return {
        "accuracy": accuracy_score(df["outcome"], model.predict(X)),
        "log_loss": log_loss(df["outcome"], probs, labels=model.classes_),
        "n": int(len(df)),
    }


def main():
    results = pd.read_csv("data/results.csv")
    feat_df, elo, form = build_dataset(results, cutoff_date="2026-07-14")

    train_df = feat_df[feat_df["date"] < SPLIT_DATE]
    valid_df = feat_df[feat_df["date"] >= SPLIT_DATE]
    valid_comp = valid_df[valid_df["k_weight"] > FRIENDLY_K]  # competitive only

    sw_train = recency_weights(train_df["date"], SPLIT_DATE).values

    # --- Logistic regression (recency-weighted, standardized) ---
    scaler = StandardScaler().fit(train_df[FEATURE_COLS])
    logreg = LogisticRegression(max_iter=3000, C=1.0)
    logreg.fit(scaler.transform(train_df[FEATURE_COLS]), train_df["outcome"], sample_weight=sw_train)

    # --- Gradient boosting sanity check (recency-weighted) ---
    gbc = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.04, random_state=42)
    gbc.fit(train_df[FEATURE_COLS], train_df["outcome"], sample_weight=sw_train)

    # --- Naive baseline: predict training class priors ---
    prior = train_df["outcome"].value_counts(normalize=True).reindex(CLASSES).fillna(0).values
    naive_probs = np.tile(prior, (len(valid_df), 1))
    naive_ll = log_loss(valid_df["outcome"], naive_probs, labels=CLASSES)

    metrics = {
        "n_train": int(len(train_df)),
        "recency_half_life_years": RECENCY_HALF_LIFE_YEARS,
        "features": FEATURE_COLS,
        "naive_baseline_log_loss_all": naive_ll,
        "logistic_regression": {
            "all_matches": _metrics(logreg, scaler, valid_df, FEATURE_COLS),
            "competitive": _metrics(logreg, scaler, valid_comp, FEATURE_COLS),
        },
        "gradient_boosting": {
            "all_matches": _metrics(gbc, None, valid_df, FEATURE_COLS),
            "competitive": _metrics(gbc, None, valid_comp, FEATURE_COLS),
        },
    }
    print(json.dumps(metrics, indent=2))

    # Logistic regression is the deployed model (safe extrapolation to the
    # top-of-distribution semifinalists). Refit on ALL history up to the
    # semifinals, recency-weighted to the day before the first semifinal.
    reference = pd.Timestamp("2026-07-14")
    sw_full = recency_weights(feat_df["date"], reference).values
    final_scaler = StandardScaler().fit(feat_df[FEATURE_COLS])
    final_model = LogisticRegression(max_iter=3000, C=1.0)
    final_model.fit(final_scaler.transform(feat_df[FEATURE_COLS]), feat_df["outcome"], sample_weight=sw_full)

    metrics["chosen_model"] = "logistic_regression"
    with open("results/classifier_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(
        {"model": final_model, "scaler": final_scaler, "features": FEATURE_COLS, "name": "logistic_regression"},
        "results/classifier.joblib",
    )
    print("\nChosen classifier: logistic_regression (recency-weighted) -> results/classifier.joblib")


if __name__ == "__main__":
    main()
