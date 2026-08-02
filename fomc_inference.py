"""
FOMC Policy Direction Model — Live Inference (v2)
=================================================
Loads the artifacts produced by fomc_trainer.py, rebuilds the latest
feature row from FRED, and predicts the direction of the NEXT month's
change in the official Fed target rate (Cut / Hold / Hike).

Run:  FRED_API_KEY=yourkey python fomc_inference.py
"""

import os
import json

import joblib
import numpy as np
import pandas as pd

# Reuse the exact same data pipeline as training — the single most common
# source of silent inference bugs is re-implementing it differently here.
from fomc_trainer import (
    build_dataset_features_only, FEATURE_COLS, CLASS_NAMES, RESULT_DIR,
    proba_3class,
)


def load_artifacts():
    def p(name):
        return os.path.join(RESULT_DIR, name)
    for f in ["scaler.pkl", "rfe_selector.pkl", "rf_model.pkl",
              "xgb_model.pkl", "feature_cols.json"]:
        if not os.path.exists(p(f)):
            raise SystemExit(
                f"Missing artifact {f} in {RESULT_DIR}. "
                "Run fomc_trainer.py first."
            )
    with open(p("feature_cols.json")) as f:
        cols = json.load(f)
    if cols != FEATURE_COLS:
        raise SystemExit(
            "feature_cols.json does not match FEATURE_COLS in fomc_trainer.py."
            " Retrain before running inference."
        )
    return (joblib.load(p("scaler.pkl")), joblib.load(p("rfe_selector.pkl")),
            joblib.load(p("rf_model.pkl")), joblib.load(p("xgb_model.pkl")))


def run_live_prediction():
    feats = build_dataset_features_only()
    latest = feats.dropna().iloc[[-1]]
    asof = latest.index[0]

    scaler, rfe, rf, xgb = load_artifacts()
    X = rfe.transform(scaler.transform(latest[FEATURE_COLS].values))
    probs = 0.5 * proba_3class(rf, X)[0] + 0.5 * proba_3class(xgb, X)[0]
    best = int(np.argmax(probs))

    next_month = (asof + pd.offsets.MonthBegin(1)).strftime("%B %Y")
    report = f"""
    --- FOMC POLICY DIRECTION REPORT ---
    Features as of      : {asof.strftime('%B %Y')}
    Predicting change in target rate during: {next_month}
    Prediction          : {CLASS_NAMES[best]}
    Probabilities       : Cut={probs[0]:.2f}  Hold={probs[1]:.2f}  Hike={probs[2]:.2f}

    Note: probabilities are from a model whose honest skill level is the
    out-of-sample walk-forward numbers in Result/Evaluation_Report.txt —
    NOT the in-sample fit. Interpret accordingly.
    """
    print(report)

    latest.to_csv(os.path.join(RESULT_DIR, "latest_prediction_data.csv"))
    with open(os.path.join(RESULT_DIR, "Prediction_Report.txt"), "w") as f:
        f.write(report)


if __name__ == "__main__":
    run_live_prediction()
