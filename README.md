**ชื่อโครงงาน (ไทย)**: การพยากรณ์ทิศทางราคาหุ้นด้วย LSTM และ Attention Mechanism ร่วมกับ Explainable AI: การเปรียบเทียบ SHAP กับ Pearson Correlation

**Project Title (Eng)**: Stock Price Direction Prediction Using LSTM + Attention Mechanism with Explainable AI (SHAP vs Correlation)

**อาจารย์ที่ปรึกษาโครงงาน**: ผศ.ดร.วิรัตน์ จารีวงศ์ไพบูลย์

**ผู้จัดทำโครงงาน**:

1. นายวชิรวิทย์ แก้วแดง
2. นายธำรง แซ่เฉิน

---

# Stock Price Direction Prediction using LSTM + Attention Mechanism with Explainable AI (SHAP vs Correlation)

## Topic 1 — Prerequisites (สิ่งที่ต้องติดตั้งก่อน)

The following software must be installed before running this project:

| Software | Version | Notes |
|----------|---------|-------|
| Python | 3.10.x | Via Anaconda or Miniconda |
| Git | Latest | For cloning the repository |
| CUDA Toolkit | 11.8 | Optional — NVIDIA GPU only. CPU fallback is automatic. |

---

## Topic 2 — Installation (วิธีติดตั้งโปรแกรม)

### Step 1 — Clone the repository

```bash
git clone https://github.com/ComSciThammasatU-classroom_2568-2_CS403-SpecialProjects2--CSAR/2568-2_CS403_Final-Submission-68-1_43_wjr-r2.git
cd 2568-2_CS403_Final-Submission-68-1_43_wjr-r2
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

## Topic 3 — Usage (วิธีการใช้งาน)

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
2568-2_CS403_Final-Submission-68-1_43_wjr-r2/
│
├── README.md
├── requirements.txt
├── stock_project.ipynb
│
├── final_reports/
│   ├── 68-2_CS403_68-1_43_wjr-r2.pdf
│
├── demo/
│   └── 68-2_CS403_68-1_43_wjr-r2_demo_link.txt
│
└── outputs/
    ├── correlation_vs_shap.png
    ├── rank_disagreement_summary.png
    ├── ensemble_comparison.png
    ├── sector_shap.png
    └── final_summary.png
```