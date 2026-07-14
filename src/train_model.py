"""
Trains the outcome classifier (Win / Draw / Loss for the home side) on the
full chronological feature set produced by features.py, evaluates it on a
held-out time slice, and saves the fitted model + a metrics report.

Improvements that drive probability quality:

  1. **Recency weighting.** Training matches are weighted by an exponential
     decay on their age (6-year half-life), so current squad strength
     dominates and 40-year-old results barely count.
  2. **Expanded, leak-free features** (Elo diff *and* level, goals-for /
     goals-against form, rest days, competition importance).
  3. **Probability calibration selection.** Two post-hoc calibrators are
     evaluated against the raw model and the best is deployed:
       - **Platt scaling** (sigmoid) — a logistic re-mapping of the scores.
       - **Isotonic regression** — a non-parametric monotone re-mapping.
     Calibration is fit on a *separate* temporal slice (2017-2018) from a
     base model trained only on earlier data, so the calibrator never sees
     the data the base was fit on (no leakage), and all three candidates are
     scored on the 2019+ holdout. Whichever minimises holdout log-loss is
     refit on all history and deployed.

     Note: a multinomial logistic regression fit by maximum likelihood is
     already close to calibrated, so on this dataset the raw model usually
     wins and calibration is kept mainly as an automatic safeguard — if the
     data ever drifts to where calibration helps, it is adopted without a
     code change. The measured holdout log-loss of each candidate is written
     to results/classifier_metrics.json for transparency.

Metrics are reported both on ALL validation matches and on the
**competitive** subset (real tournaments, excluding friendlies) — the
latter is the domain a World Cup semifinal actually lives in.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_dataset, FEATURE_COLS

CLASSES = ["A", "D", "H"]
SPLIT_DATE = pd.Timestamp("2019-01-01")          # base/calib < this <= holdout
CALIB_START = pd.Timestamp("2017-01-01")         # base < this <= calibration slice
RECENCY_HALF_LIFE_YEARS = 6.0
FRIENDLY_K = 20  # matches with k_weight above this are competitive
SELECTION_DOMAIN = "competitive"  # pick the calibrator by competitive-domain log-loss


def recency_weights(dates: pd.Series, reference: pd.Timestamp) -> np.ndarray:
    age_years = (reference - dates).dt.days / 365.25
    return 0.5 ** (age_years / RECENCY_HALF_LIFE_YEARS)


def _base_pipeline() -> "Pipeline":
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))


def _fit_base(df: pd.DataFrame, reference: pd.Timestamp):
    pipe = _base_pipeline()
    sw = recency_weights(df["date"], reference).values
    pipe.fit(df[FEATURE_COLS], df["outcome"], logisticregression__sample_weight=sw)
    return pipe


def _metrics(model, df: pd.DataFrame) -> dict:
    probs = model.predict_proba(df[FEATURE_COLS])
    return {
        "accuracy": accuracy_score(df["outcome"], model.predict(df[FEATURE_COLS])),
        "log_loss": log_loss(df["outcome"], probs, labels=model.classes_),
        "n": int(len(df)),
    }


def _evaluate_candidates(feat_df: pd.DataFrame) -> dict:
    """Fit base + Platt + isotonic on leak-free temporal slices and score each
    on the 2019+ holdout. Returns per-candidate metrics."""
    base_df = feat_df[feat_df["date"] < CALIB_START]
    calib_df = feat_df[(feat_df["date"] >= CALIB_START) & (feat_df["date"] < SPLIT_DATE)]
    holdout = feat_df[feat_df["date"] >= SPLIT_DATE]
    holdout_comp = holdout[holdout["k_weight"] > FRIENDLY_K]

    base = _fit_base(base_df, CALIB_START)

    candidates = {"uncalibrated": base}
    for method in ("sigmoid", "isotonic"):
        cal = CalibratedClassifierCV(FrozenEstimator(base), method=method)
        cal.fit(calib_df[FEATURE_COLS], calib_df["outcome"])
        candidates["platt" if method == "sigmoid" else "isotonic"] = cal

    report = {}
    for name, model in candidates.items():
        report[name] = {
            "all_matches": _metrics(model, holdout),
            "competitive": _metrics(model, holdout_comp),
        }
    return report


def _deploy(feat_df: pd.DataFrame, method: str):
    """Refit the chosen method on ALL history up to the cutoff for deployment."""
    reference = pd.Timestamp("2026-07-14")
    if method == "uncalibrated":
        return _fit_base(feat_df, reference)
    # Cross-fitted calibration on all data (uses every match for both the base
    # fit and the calibrator via out-of-fold predictions, no data wasted).
    sk_method = "sigmoid" if method == "platt" else "isotonic"
    base = _base_pipeline()
    cal = CalibratedClassifierCV(base, method=sk_method, cv=5)
    sw = recency_weights(feat_df["date"], reference).values
    cal.fit(feat_df[FEATURE_COLS], feat_df["outcome"], sample_weight=sw)
    return cal


def main():
    results = pd.read_csv("data/results.csv")
    feat_df, elo, form = build_dataset(results, cutoff_date="2026-07-14")

    # Naive baseline: predict training class priors.
    train_df = feat_df[feat_df["date"] < SPLIT_DATE]
    holdout = feat_df[feat_df["date"] >= SPLIT_DATE]
    prior = train_df["outcome"].value_counts(normalize=True).reindex(CLASSES).fillna(0).values
    naive_ll = log_loss(holdout["outcome"], np.tile(prior, (len(holdout), 1)), labels=CLASSES)

    candidates = _evaluate_candidates(feat_df)

    # Choose the calibrator with the lowest holdout log-loss on the target domain.
    best = min(candidates, key=lambda k: candidates[k][SELECTION_DOMAIN]["log_loss"])

    metrics = {
        "n_train": int(len(train_df)),
        "recency_half_life_years": RECENCY_HALF_LIFE_YEARS,
        "features": FEATURE_COLS,
        "naive_baseline_log_loss_all": naive_ll,
        "calibration_selection_domain": SELECTION_DOMAIN,
        "candidates_holdout": candidates,
        "chosen_calibration": best,
    }
    print(json.dumps(metrics, indent=2))
    print(f"\nCalibration comparison ({SELECTION_DOMAIN} holdout log-loss):")
    for name in ("uncalibrated", "platt", "isotonic"):
        ll = candidates[name][SELECTION_DOMAIN]["log_loss"]
        mark = "  <- chosen" if name == best else ""
        print(f"  {name:13s} {ll:.4f}{mark}")

    final_model = _deploy(feat_df, best)
    metrics["chosen_model"] = f"logistic_regression ({best})"

    with open("results/classifier_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump(
        {"model": final_model, "scaler": None, "features": FEATURE_COLS,
         "name": f"logistic_regression_{best}"},
        "results/classifier.joblib",
    )
    print(f"\nDeployed: logistic_regression + {best} calibration -> results/classifier.joblib")


if __name__ == "__main__":
    main()
