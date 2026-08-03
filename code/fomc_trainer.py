"""
FOMC Policy Direction Model — Trainer (v2)
==========================================
Fixes over v1:
  1. Label is FORWARD-LOOKING: month-t features predict the month t -> t+1
     change in the Fed's TARGET rate (not the same-month effective rate).
  2. 3-class target (Cut / Hold / Hike) built from the official target rate
     (DFEDTAR pre-Dec-2008, DFEDTARU after), threshold +/- 10bp.
  3. Honest evaluation: expanding-window walk-forward (TimeSeriesSplit).
     Scaler + RFE are fit INSIDE each training fold only (no leakage).
  4. Random-noise "sentiment" placeholder removed from the feature set.
  5. GDPC1 (quarterly) is forward-filled before the final dropna, and
     FRED's short-history SP500 is replaced with NASDAQCOM (full history),
     so the dataset is ~300 monthly rows instead of ~40.
  6. API key comes from the FRED_API_KEY environment variable.
  7. Class imbalance handled (balanced class/sample weights).
  8. SHAP via TreeExplainer (exact for tree models, fast).

Run:  FRED_API_KEY=yourkey python fomc_trainer.py
Deps: pandas numpy scikit-learn xgboost shap matplotlib joblib
"""

import os
import json
import urllib.request

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import shap

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
CLASS_NAMES = {0: "Cut", 1: "Hold", 2: "Hike"}
HIKE_CUT_THRESHOLD = 0.10   # bp threshold (in %) for calling a move vs a hold
N_FEATURES_TO_SELECT = 6
N_SPLITS = 5                # walk-forward folds
START_DATE = "1999-01-01"   # 1 extra year so 12m transforms exist by 2000

FEATURE_COLS = [
    "tgt_rate",       # current target rate level (policy stance)
    "cpi_yoy",        # CPI inflation, % YoY
    "pce_yoy",        # PCE inflation, % YoY
    "gdp_yoy",        # real GDP growth, % YoY (quarterly, ffilled)
    "unrate",         # unemployment rate level
    "unrate_chg_3m",  # 3-month change in unemployment
    "vix",            # VIX level
    "t10y2y",         # 10y-2y curve slope
    "eq_ret_12m",     # equity 12-month return, %
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "Result")
os.makedirs(RESULT_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def _fred_key() -> str:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "FRED_API_KEY environment variable is not set.\n"
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "Then run e.g.:  export FRED_API_KEY=yourkey"
        )
    return key


def fetch_series(series_id: str, start_date: str = START_DATE) -> pd.Series:
    """Fetch one FRED series as a pandas Series indexed by date."""
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={_fred_key()}"
        f"&file_type=json&observation_start={start_date}"
    )
    with urllib.request.urlopen(url) as r:
        js = json.loads(r.read().decode())
    obs = js.get("observations", [])
    if not obs:
        raise RuntimeError(f"FRED returned no observations for {series_id}")
    s = pd.Series(
        [float(o["value"]) if o["value"] != "." else np.nan for o in obs],
        index=pd.to_datetime([o["date"] for o in obs]),
        name=series_id,
    )
    return s


def fetch_target_rate() -> pd.Series:
    """
    Official Fed target rate, spliced:
      DFEDTAR  : single target rate, ends 2008-12-15
      DFEDTARU : upper bound of target range, starts 2008-12-16
    End-of-month value (the rate in force going into next month).
    """
    old = fetch_series("DFEDTAR")
    new = fetch_series("DFEDTARU")
    tgt = pd.concat([old[old.index < "2008-12-16"], new]).sort_index()
    tgt = tgt.resample("MS").last().ffill()
    tgt.name = "tgt_rate"
    return tgt


def build_dataset_features_only() -> pd.DataFrame:
    """
    Fetch raw series and construct the monthly feature matrix (no label).
    Shared by trainer AND inference so the two pipelines cannot diverge.
    """
    print("Fetching data from FRED...")
    raw = {}
    for sid in ["CPIAUCSL", "PCEPI", "GDPC1", "UNRATE", "VIXCLS",
                "T10Y2Y", "NASDAQCOM"]:
        raw[sid] = fetch_series(sid).resample("MS").mean()
        print(f"  {sid}: {raw[sid].dropna().shape[0]} monthly obs")

    tgt = fetch_target_rate()
    print(f"  target rate (DFEDTAR+DFEDTARU): {tgt.dropna().shape[0]} monthly obs")

    df = pd.concat([tgt] + list(raw.values()), axis=1)

    # Quarterly GDP -> monthly via forward-fill BEFORE any dropna
    df["GDPC1"] = df["GDPC1"].ffill()

    # --- features (stationary-ish transforms of the raw levels) ---
    feats = pd.DataFrame(index=df.index)
    feats["tgt_rate"] = df["tgt_rate"]
    feats["cpi_yoy"] = df["CPIAUCSL"].pct_change(12) * 100
    feats["pce_yoy"] = df["PCEPI"].pct_change(12) * 100
    feats["gdp_yoy"] = df["GDPC1"].pct_change(12) * 100
    feats["unrate"] = df["UNRATE"]
    feats["unrate_chg_3m"] = df["UNRATE"].diff(3)
    feats["vix"] = df["VIXCLS"]
    feats["t10y2y"] = df["T10Y2Y"]
    feats["eq_ret_12m"] = df["NASDAQCOM"].pct_change(12) * 100
    return feats


def build_dataset() -> pd.DataFrame:
    """Monthly feature matrix + forward-looking 3-class label."""
    feats = build_dataset_features_only()

    # --- forward-looking label: what does the TARGET rate do next month? ---
    d_next = feats["tgt_rate"].shift(-1) - feats["tgt_rate"]
    label = pd.Series(1.0, index=feats.index, name="y")     # Hold
    label[d_next >= HIKE_CUT_THRESHOLD] = 2                 # Hike
    label[d_next <= -HIKE_CUT_THRESHOLD] = 0                # Cut
    label[d_next.isna()] = np.nan                           # last row: unknown

    data = feats.copy()
    data["y"] = label
    data = data.loc["2000-01-01":]
    data = data.dropna()
    data["y"] = data["y"].astype(int)

    print(f"\nFinal dataset shape: {data.shape}")
    counts = data["y"].value_counts().sort_index()
    print("Class counts:", {CLASS_NAMES[k]: int(v) for k, v in counts.items()})

    # sanity guards (these mirror the acceptance checks)
    assert data.shape[0] > 200, (
        f"Only {data.shape[0]} rows — data assembly is broken (expected ~300). "
        "Do NOT proceed; check which series failed to fetch."
    )
    assert counts.idxmax() == 1, (
        "Hold is not the majority class — label construction is suspect."
    )
    return data


# ----------------------------------------------------------------------------
# Model fitting (fold-safe)
# ----------------------------------------------------------------------------
def fit_models(X_train: np.ndarray, y_train: np.ndarray):
    """
    Fit scaler + RFE + RF + XGB on TRAINING data only.
    Returns the fitted artifacts.
    """
    scaler = StandardScaler().fit(X_train)
    X_tr = scaler.transform(X_train)

    rfe = RFE(
        RandomForestClassifier(n_estimators=300, random_state=42,
                               class_weight="balanced"),
        n_features_to_select=N_FEATURES_TO_SELECT,
    ).fit(X_tr, y_train)
    X_sel = rfe.transform(X_tr)

    rf = RandomForestClassifier(
        n_estimators=500, max_depth=6, min_samples_leaf=5,
        class_weight="balanced", random_state=42,
    ).fit(X_sel, y_train)

    sw = compute_sample_weight("balanced", y_train)
    xgb = XGBClassifier(
        objective="multi:softprob", num_class=3,
        n_estimators=400, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="mlogloss",
    ).fit(X_sel, y_train, sample_weight=sw)

    return scaler, rfe, rf, xgb


def proba_3class(model, X: np.ndarray) -> np.ndarray:
    """predict_proba expanded to a full 3-column matrix even if a class
    was absent from the training fold."""
    p = model.predict_proba(X)
    out = np.zeros((X.shape[0], 3))
    for i, c in enumerate(model.classes_):
        out[:, int(c)] = p[:, i]
    return out


def ensemble_proba(rf, xgb, X_sel: np.ndarray) -> np.ndarray:
    return 0.5 * proba_3class(rf, X_sel) + 0.5 * proba_3class(xgb, X_sel)


# ----------------------------------------------------------------------------
# Walk-forward evaluation
# ----------------------------------------------------------------------------
def walk_forward_eval(data: pd.DataFrame) -> str:
    X = data[FEATURE_COLS].values
    y = data["y"].values

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    lines = ["--- WALK-FORWARD (OUT-OF-SAMPLE) EVALUATION ---",
             f"{N_SPLITS} expanding-window folds, TimeSeriesSplit\n"]
    accs, f1s = [], []
    total_cm = np.zeros((3, 3), dtype=int)

    for k, (tr_idx, te_idx) in enumerate(tscv.split(X), start=1):
        scaler, rfe, rf, xgb = fit_models(X[tr_idx], y[tr_idx])
        X_te = rfe.transform(scaler.transform(X[te_idx]))
        pred = np.argmax(ensemble_proba(rf, xgb, X_te), axis=1)

        acc = accuracy_score(y[te_idx], pred)
        f1m = f1_score(y[te_idx], pred, average="macro")
        cm = confusion_matrix(y[te_idx], pred, labels=[0, 1, 2])
        total_cm += cm
        accs.append(acc)
        f1s.append(f1m)

        tr_end = data.index[tr_idx[-1]].strftime("%Y-%m")
        te_span = (f"{data.index[te_idx[0]].strftime('%Y-%m')}"
                   f" -> {data.index[te_idx[-1]].strftime('%Y-%m')}")
        flag = "  <-- SUSPICIOUSLY HIGH, investigate" if acc > 0.95 else ""
        lines.append(f"Fold {k}: train ends {tr_end}, test {te_span} | "
                     f"acc={acc:.3f}  macroF1={f1m:.3f}{flag}")

    lines.append(f"\nMean OOS accuracy : {np.mean(accs):.3f}")
    lines.append(f"Mean OOS macro F1 : {np.mean(f1s):.3f}")
    lines.append("\nAggregate confusion matrix (rows=true, cols=pred)"
                 " [Cut, Hold, Hike]:")
    for i, row in enumerate(total_cm):
        lines.append(f"  {CLASS_NAMES[i]:<5} {row.tolist()}")

    # naive baseline: always predict Hold
    base = accuracy_score(y, np.ones_like(y))
    lines.append(f"\nBaseline (always 'Hold') accuracy: {base:.3f}")
    lines.append("A useful model must beat this on macro F1, since accuracy"
                 " alone is inflated by the Hold majority.")

    report = "\n".join(lines)
    print("\n" + report)
    return report


# ----------------------------------------------------------------------------
# SHAP (TreeExplainer, Hike class)
# ----------------------------------------------------------------------------
def make_shap_plot(rf, rfe, scaler, data: pd.DataFrame):
    print("\nGenerating SHAP summary plot (TreeExplainer, RF, 'Hike' class)...")
    X_sel = rfe.transform(scaler.transform(data[FEATURE_COLS].values))
    selected = np.array(FEATURE_COLS)[rfe.support_].tolist()

    sv = shap.TreeExplainer(rf).shap_values(X_sel)
    if isinstance(sv, list):                       # older shap: list per class
        sv_hike = sv[2]
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:  # newer: (n, feat, class)
        sv_hike = sv[:, :, 2]
    else:
        sv_hike = sv

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    })
    plt.figure(figsize=(14, 8))
    shap.summary_plot(sv_hike, X_sel, feature_names=selected,
                      show=False, alpha=0.6, plot_size=None)
    ax = plt.gca()
    ax.xaxis.grid(True, linestyle="--", color="grey", alpha=0.25)
    ax.set_axisbelow(True)
    plt.title("SHAP Analysis: Feature Impact on Next-Month FOMC Hike"
              " Probability", pad=25, fontweight="bold", fontsize=17)
    plt.xlabel("SHAP value (positive = pushes toward Hike next month)",
               labelpad=15, fontweight="bold", fontsize=13)
    caption = (
        "Figure 1. SHAP summary (TreeExplainer, Random Forest, 'Hike' class).\n"
        "Month-t macro features vs. the month t\u2192t+1 change in the official"
        " Fed target rate (3-class: Cut/Hold/Hike)."
    )
    plt.figtext(0.5, -0.06, caption, wrap=True, ha="center",
                fontsize=11, color="#444444")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(RESULT_DIR, f"shap_summary_top.{ext}"),
                    dpi=400 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved Result/shap_summary_top.png and .pdf")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    data = build_dataset()
    data.to_csv(os.path.join(RESULT_DIR, "training_dataset.csv"))

    eval_report = walk_forward_eval(data)
    with open(os.path.join(RESULT_DIR, "Evaluation_Report.txt"), "w") as f:
        f.write(eval_report + "\n")

    # Final production model: fit on ALL labeled data (for live inference
    # only — every performance number above is from held-out folds).
    print("\nFitting final model on full labeled history for inference...")
    X = data[FEATURE_COLS].values
    y = data["y"].values
    scaler, rfe, rf, xgb = fit_models(X, y)

    joblib.dump(scaler, os.path.join(RESULT_DIR, "scaler.pkl"))
    joblib.dump(rfe, os.path.join(RESULT_DIR, "rfe_selector.pkl"))
    joblib.dump(rf, os.path.join(RESULT_DIR, "rf_model.pkl"))
    joblib.dump(xgb, os.path.join(RESULT_DIR, "xgb_model.pkl"))
    with open(os.path.join(RESULT_DIR, "feature_cols.json"), "w") as f:
        json.dump(FEATURE_COLS, f)

    make_shap_plot(rf, rfe, scaler, data)
    print("\nDone. Artifacts in ./Result/")


if __name__ == "__main__":
    main()
