# Stock Price Direction Prediction Using LSTM + Attention Mechanism with Explainable AI (SHAP vs Correlation)

**Advisor**: Assistant Professor, Ph.D. Wirat Jareevongpiboon

**Authors**:
1. Wachirawit Kaewdaeng
2. Tumrong Saechoen

## Overview

**Question.** Can technical and market-wide indicators predict the 3-day direction (Up/Down) of large US stocks, and do the two common ways of ranking feature importance, Pearson correlation and SHAP, agree on which indicators matter?

**Short answer.** The feature rankings disagree a lot, but the models have little predictive skill, so the rankings should be read as a description of what the models use, not as evidence of what drives prices. Details and caveats are in [Key findings](#key-findings) and [Limitations](#limitations).

## Approach

| Step | What was done |
|---|---|
| Data | Daily OHLCV from Yahoo Finance (`yfinance`) for 10 stocks in 5 sectors (NVDA, TSLA · JNJ, UNH · XOM, CVX · JPM, GS · AMZN, COST), plus S&P 500, bond yield and VIX. Jan 2020 to Apr 2025, 1,283 usable trading days per stock |
| Features | 27 features: price/returns, trend, momentum, volume, market-wide (VIX added as an extra experiment) |
| Target | Direction of the 3-day forward return (Up = 1). Time-ordered 80/20 train/test split |
| Models | LSTM + attention (3 layers, hidden size 128, 60-day windows, focal loss); Random Forest and Logistic Regression as baselines; soft-voting ensemble |
| Explainability | SHAP (`KernelExplainer`) vs absolute Pearson correlation with the target; cross-model consistency; sector-level view; effect of VIX features |

## Key findings

1. **Correlation and SHAP rank features very differently.** Examples from the notebook output: for NVDA, `Volatility_20d` is rank 1 by correlation but rank 19 by SHAP; for TSLA, `SP500_return` moves from rank 22 to rank 2; for JNJ, from rank 26 to rank 4.
2. **`Bond_yield` shows up near the top of SHAP rankings for several stocks** (for example rank 1 for TSLA) while correlation ranks it low. <!-- TODO: replace with the exact count from your data, e.g. "in the SHAP top-5 for X of 10 stocks" -->
3. **Predictive skill is close to chance.** Test accuracy of the LSTM ranged from about 0.49 to 0.60 across the 10 stocks, and ensemble AUC from about 0.50 to 0.66. Adding VIX features and ensembling did not give a consistent improvement.

## Limitations

Stated up front so the results are read correctly:

- **Test set used for model selection.** In the submitted version, early stopping (best epoch by F1) and the decision threshold were both chosen on the test set, so reported metrics are optimistic. A validation split is planned.
- **F1 on the "Up" class is misleading here.** Most stocks ended up with a low threshold (0.30), which makes the model predict Up almost every day. That gives an F1 close to a trivial "always Up" baseline, so accuracy and AUC are the more honest numbers.
- **Scaler fitted on the full series** before the train/test split (minor leakage).
- **SHAP estimated from 30 test samples** with `nsamples=100`, so the exact rankings are noisy and should be treated as indicative.
- **Explaining a weak model.** With AUC near 0.5, SHAP tells us what the model relies on, not what moves the market.
- **Single period and single split** (2020 to Apr 2025), overlapping 3-day targets, no transaction costs. This is a study of explainability methods, not a trading strategy.

## My contribution

I wrote the project report. The code (data pipeline, model training and SHAP experiments) 
was mainly implemented by my teammate, Wachirawit Kaewdaeng. I understand the overall design and results.

---

## Prerequisites

The following software must be installed before running this project:

| Software | Version | Notes |
|----------|---------|-------|
| Python | 3.10.x | Via Anaconda or Miniconda |
| Git | Latest | For cloning the repository |
| CUDA Toolkit | 11.8 | Optional — NVIDIA GPU only. CPU fallback is automatic. |

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/Tumrong-Saechoen/Graduating-Project.git
cd Graduating-Project
```

### Step 2 — Create and activate conda environment

```bash
conda create -n stock_project python=3.10 -y
conda activate stock_project
```

### Step 3 — Install PyTorch

**With GPU (CUDA 11.8):**
```bash
pip install torch==2.7.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CPU only:**
```bash
pip install torch==2.7.1 torchvision torchaudio
```

### Step 4 — Install all dependencies

```bash
pip install -r requirements.txt
```

This installs: `yfinance==1.3.0`, `pandas==2.3.3`, `numpy==2.2.6`, `matplotlib==3.10.9`, `seaborn==0.13.2`, `shap==0.49.1`, `ta==0.11.0`, `scikit-learn==1.7.2`, `jupyter==1.1.1`

---

## Usage

### Step 1 — Launch Jupyter Notebook

```bash
jupyter notebook stock_project.ipynb
```

### Step 2 — Run all cells top to bottom

The notebook is fully self-contained. Running all cells in order will:

1. **Download stock data** — fetches daily OHLCV data for 10 stocks (NVDA, TSLA, JNJ, UNH, XOM, CVX, JPM, GS, AMZN, COST) plus S&P 500, Bond Yield, and VIX from Yahoo Finance via `yfinance`. No API key required.
2. **Build features** — computes 27 technical and market-wide features per stock.
3. **Train the LSTM model** — trains a 3-layer LSTM with Attention Mechanism for each stock using Focal Loss (gamma=2) and Early Stopping (patience=20, monitor F1-Score).
4. **Run all 5 experiments** — SHAP vs Pearson Correlation, Cross-model SHAP consistency, VIX Sentiment, Ensemble Model, and Sector-level SHAP analysis.
5. **Save output graphs** — all result charts are saved automatically to the `outputs/` folder.

> ⚠️ **Note:** Training 10 stocks takes significant time (~30–60 min on GPU, longer on CPU). It is recommended to run the notebook once and keep the cell outputs saved.

### Project Structure

```
Graduating-Project/
│
├── README.md
├── requirements.txt
├── stock_project.ipynb
│
├── reports/
│   ├── Report.pdf
│
├── demo/
│   └── demo_link.txt
│
└── outputs/
    ├── correlation_vs_shap.png
    ├── rank_disagreement_summary.png
    ├── ensemble_comparison.png
    ├── sector_shap.png
    └── final_summary.png
```