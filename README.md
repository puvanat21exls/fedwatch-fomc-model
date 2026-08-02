# 🏛️ FedWatch: FOMC Policy Direction Machine Learning Model

A machine learning framework that predicts the direction of the Federal Reserve's next-month interest rate decisions (**Cut / Hold / Hike**) using macroeconomic fundamentals, evaluated with strict expanding-window walk-forward validation (`TimeSeriesSplit`).

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![XGBoost](https://img.shields.io/badge/Model-XGBoost%20%26%20Random%20Forest-green.svg)
![SHAP](https://img.shields.io/badge/Interpretability-SHAP%20TreeExplainer-red.svg)

---

## 📌 Executive Summary & Methodological Rebuild (v1 vs. v2)

Initial versions of macro-prediction models often suffer from **silent data leakage**. Version 1 of this model reported an artificial 99% accuracy because it evaluated same-month rate changes rather than forward-looking shifts, and relied on overfitting features.

**Version 2 completely rebuilds the pipeline to guarantee honest out-of-sample evaluation:**
1. **Forward-Looking Target ($t \rightarrow t+1$):** Month-$t$ macroeconomic features predict the *next* month's target rate change using the official spliced Fed target rate (`DFEDTAR` + `DFEDTARU`).
2. **3-Class Policy Definition:** Classifies decisions as **Cut** ($\le -10\text{bp}$), **Hold** ($>-10\text{bp}$ and $<+10\text{bp}$), or **Hike** ($\ge +10\text{bp}$).
3. **Strict Walk-Forward Evaluation (`TimeSeriesSplit`):** Feature scaling (`StandardScaler`) and Recursive Feature Elimination (`RFE`) are fit **inside each training fold only** to prevent lookahead bias.
4. **Class Imbalance Handling:** Uses balanced sample weights to ensure rare rate-cutting and hiking cycles are not swallowed by the dominant "Hold" baseline.

---

## 📊 Walk-Forward Out-Of-Sample Results

* **Dataset Size:** ~315 monthly observations (2000 – Present)
* **Class Distribution:** Hold (~245), Hike (~40), Cut (~30)

| Fold | Test Window | Out-of-Sample Accuracy | Macro F1 |
| :--- | :--- | :--- | :--- |
| **Fold 1** | 2004-08 → 2008-11 | 0.500 | 0.324 |
| **Fold 2** | 2008-12 → 2013-03 | 0.865 | 0.464 |
| **Fold 3** | 2013-04 → 2017-07 | 0.865 | 0.464 |
| **Fold 4** | 2017-08 → 2021-11 | 0.692 | 0.359 |
| **Fold 5** | 2021-12 → 2026-05 | 0.442 | 0.283 |
| **Mean** | **Full History** | **0.673** | **0.379** |

*> **Honest Evaluation Note:** While an "Always Hold" naive baseline achieves 0.778 accuracy due to hold dominance, its Macro F1 is only ~0.29 (zero detection of cuts/hikes). The ensemble model's **0.379 Macro F1** reflects genuine predictive skill in detecting policy regime shifts.*

---

## 🔍 Feature Interpretability (SHAP Analysis)

Feature importance was validated using **SHAP (SHapley Additive exPlanations) TreeExplainer** on the Random Forest ensemble for the **Hike** class:

![SHAP Summary]<img width="5035" height="3429" alt="shap_summary_top" src="https://github.com/user-attachments/assets/23cdfec5-f9c4-435d-9d4e-7e80fc8add32" />


### Key Economic Drivers Identified:
* **CPI & PCE YoY Inflation:** Strong positive impact on Hike probability when elevated.
* **Unemployment Rate & 3M Change:** High unemployment and rising joblessness strongly reduce Hike probabilities.
* **Yield Curve Slope ($T10Y2Y$) & VIX:** Elevated market stress and yield curve inversions act as natural "policy vetoes."

---

## 📁 Repository Architecture

```text
├── code/
│   ├── fomc_trainer.py       # Data pipeline, walk-forward training & SHAP generation
│   └── fomc_inference.py     # Live prediction script querying latest FRED metrics
├── results/                  # Serialized models, evaluation reports, and SHAP plots
├── README.md
└── requirements.txt
