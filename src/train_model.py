"""
Trains the outcome classifier (Win / Draw / Loss for the home side) on the
full chronological feature set produced by features.py, evaluates it on a
held-out time slice, and saves the fitted model + a metrics report.

Two candidate models are compared:
  - Multinomial logistic regression (interpretable baseline).
  - Gradient boosting classifier (captures non-linear interactions between
    Elo gap, recent form and home advantage).

The one with the lower validation log-loss is kept as `model.joblib`.
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

from features import build_dataset

FEATURE_COLS = ["elo_diff", "neutral", "form_ppg_diff", "form_gd_diff"]
CLASSES = ["A", "D", "H"]  # sklearn sorts alphabetically; keep explicit for clarity


def main():
    results = pd.read_csv("data/results.csv")
    feat_df, elo, form = build_dataset(results, cutoff_date="2026-07-14")

    # Time-based split: train on everything before 2019, validate on 2019+.
    # This mimics predicting genuinely future matches rather than randomly
    # held-out rows that could sit right next to their neighbours in time.
    split_date = pd.Timestamp("2019-01-01")
    train_df = feat_df[feat_df["date"] < split_date]
    valid_df = feat_df[feat_df["date"] >= split_date]

    X_train, y_train = train_df[FEATURE_COLS], train_df["outcome"]
    X_valid, y_valid = valid_df[FEATURE_COLS], valid_df["outcome"]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_valid_s = scaler.transform(X_valid)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_s, y_train)
    logreg_probs = logreg.predict_proba(X_valid_s)
    logreg_ll = log_loss(y_valid, logreg_probs, labels=logreg.classes_)
    logreg_acc = accuracy_score(y_valid, logreg.predict(X_valid_s))

    gbc = GradientBoostingClassifier(
        n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42
    )
    gbc.fit(X_train, y_train)
    gbc_probs = gbc.predict_proba(X_valid)
    gbc_ll = log_loss(y_valid, gbc_probs, labels=gbc.classes_)
    gbc_acc = accuracy_score(y_valid, gbc.predict(X_valid))

    # Naive baseline: always predict class priors from the training set.
    prior = y_train.value_counts(normalize=True).reindex(CLASSES).fillna(0).values
    naive_probs = np.tile(prior, (len(y_valid), 1))
    naive_ll = log_loss(y_valid, naive_probs, labels=CLASSES)

    metrics = {
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "naive_baseline_log_loss": naive_ll,
        "logistic_regression": {"log_loss": logreg_ll, "accuracy": logreg_acc},
        "gradient_boosting": {"log_loss": gbc_ll, "accuracy": gbc_acc},
    }
    print(json.dumps(metrics, indent=2))

    # Refit the winning model on ALL available history (up to the semifinals)
    # so the final prediction uses every match, not just the pre-2019 slice.
    X_full, y_full = feat_df[FEATURE_COLS], feat_df["outcome"]
    if gbc_ll <= logreg_ll:
        best_name = "gradient_boosting"
        final_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42
        )
        final_model.fit(X_full, y_full)
        final_scaler = None
    else:
        best_name = "logistic_regression"
        final_scaler = StandardScaler().fit(X_full)
        final_model = LogisticRegression(max_iter=1000)
        final_model.fit(final_scaler.transform(X_full), y_full)

    metrics["chosen_model"] = best_name
    with open("results/classifier_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(
        {"model": final_model, "scaler": final_scaler, "features": FEATURE_COLS, "name": best_name},
        "results/classifier.joblib",
    )
    print(f"\nChosen classifier: {best_name} -> saved to results/classifier.joblib")


if __name__ == "__main__":
    main()
