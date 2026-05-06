<div align="center">

# 🛢️ Wellbore Geology Prediction
### ROGII Kaggle Competition — AI-Powered Geosteering

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-00BFC4)](https://lightgbm.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*AI-powered geosteering and wellbore geology prediction using machine learning,
petrophysics, and subsurface analytics for horizontal drilling optimization.*

</div>

---

## 📌 Project Overview

This repository is a professional MLOps-grade implementation for the
**[ROGII — Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)**
Kaggle competition. The goal is to accurately predict geological formations
(lithology / stratigraphy) along horizontal wellbores using:

- **Drilling mechanics data** — ROP, WOB, RPM, Torque, Flow Rate, ECD
- **LWD / MWD petrophysical logs** — Gamma Ray, Resistivity, Neutron, Density, Sonic
- **Directional survey data** — Inclination, Azimuth, TVD, MD

Accurate formation prediction while drilling enables real-time geosteering
decisions that keep the wellbore in the optimal reservoir zone, maximising
hydrocarbon recovery and reducing well costs.

---

## 🏆 Competition Description

| Field | Details |
|-------|---------|
| **Host** | ROGII |
| **Task** | Regression — predict continuous formation index |
| **Evaluation metric** | Root Mean Squared Error (RMSE) |
| **Data** | Multi-well horizontal drilling dataset with LWD logs |
| **Challenge** | Temporal / depth-sequential structure, well-level data leakage |

---

## 🔁 Workflow

```
Raw Data (train.csv / test.csv)
        │
        ▼
  01_eda.ipynb          ← Explore distributions, missing values, well stats
        │
        ▼
  02_feature_engineering.ipynb
        │  Rolling stats · Lag features · Depth features
        │  Drilling mechanics (MSE) · Petrophysical ratios (Vsh, AI)
        │  Directional encodings (sin/cos inclination & azimuth)
        ▼
  03_baseline_lightgbm.ipynb
        │  GroupKFold CV (by well) · OOF RMSE · Feature importance
        │  Optuna tuning (optional) · Ensemble blending
        ▼
  submission.csv  →  Kaggle Leaderboard
```

---

## 📁 Repository Structure

```
wellbore-geology-prediction/
├── data/
│   ├── raw/               ← Competition CSV files (not tracked)
│   ├── processed/         ← Engineered parquet files (not tracked)
│   └── submissions/       ← Output submission CSVs
│
├── notebooks/
│   ├── 01_eda.ipynb                   ← Exploratory data analysis
│   ├── 02_feature_engineering.ipynb   ← Feature generation pipeline
│   └── 03_baseline_lightgbm.ipynb     ← LightGBM training & submission
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       ← Data loading, cleaning, scaling
│   ├── feature_engineering.py ← Rolling, lag, depth, petro, drilling features
│   ├── validation.py          ← GroupKFold, RMSE, MAE, R², OOF utilities
│   ├── model_training.py      ← LightGBM / XGBoost / CatBoost + Optuna
│   ├── inference.py           ← Prediction aggregation & submission builder
│   ├── ensemble.py            ← Blending, stacking, rank-averaging
│   ├── deep_learning.py       ← LSTM & Transformer sequence models (PyTorch)
│   └── plotting.py            ← Log curves, feature importance, OOF diagnostics
│
├── configs/
│   ├── config.yaml            ← Main project configuration
│   ├── lgbm_config.yaml       ← LightGBM hyperparameters & Optuna search space
│   └── dl_config.yaml         ← LSTM / Transformer architecture & training config
│
├── outputs/
│   ├── models/                ← Saved model files (.pkl, .pt)
│   ├── plots/                 ← Saved figures
│   └── logs/                  ← Training logs
│
├── docs/                      ← Project documentation
├── assets/                    ← Static assets (diagrams, images)
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Languages** | Python 3.10+ |
| **Data** | pandas, numpy, scipy |
| **ML** | scikit-learn, LightGBM, XGBoost, CatBoost |
| **Deep Learning** | PyTorch (LSTM, Transformer) |
| **Hyperparameter Tuning** | Optuna |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Serialization** | joblib, PyYAML |
| **Notebooks** | JupyterLab |

---

## ⚡ Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/Ashraf-ISM/wellbore-geology-prediction.git
cd wellbore-geology-prediction
pip install -r requirements.txt
```

### 2. Download competition data

```bash
# Place files in data/raw/
kaggle competitions download -c rogii-wellbore-geology-prediction
unzip rogii-wellbore-geology-prediction.zip -d data/raw/
```

### 3. Run notebooks in order

```bash
jupyter lab
# Open and run:
# 1. notebooks/01_eda.ipynb
# 2. notebooks/02_feature_engineering.ipynb
# 3. notebooks/03_baseline_lightgbm.ipynb
```

### 4. Submit to Kaggle

The final submission CSV will be saved at `data/submissions/submission_lgbm_baseline.csv`.

---

## 🧪 Key Features

### Validation Strategy
- **GroupKFold** cross-validation grouped by `well_id` prevents data leakage
  between wells with similar geological profiles.

### Feature Engineering
| Group | Features |
|-------|----------|
| Rolling statistics | Mean, Std, Min, Max across windows [3, 5, 10, 20, 50] |
| Lag features | Lagged values at steps [1, 2, 3, 5, 10] |
| Depth features | Normalized depth, depth-from-top, TVD/MD ratio, horizontal departure |
| Drilling mechanics | MSE, Rev/ft, WOB/RPM ratio, specific torque |
| Petrophysical | Vsh (from GR), log(Rt), N-D separation, acoustic impedance |
| Directional | sin/cos encoded inclination & azimuth |

### Models Implemented
- ✅ **LightGBM** — GBDT regression with Optuna tuning
- ✅ **XGBoost** — Gradient boosting alternative
- ✅ **CatBoost** — Symmetric tree boosting
- ✅ **LSTM** — Bidirectional LSTM for depth sequences (PyTorch)
- ✅ **Transformer** — Positional-encoding encoder for sequence modelling

### Ensemble Methods
- Weighted average blending
- Inverse-RMSE weighting
- Meta-learner stacking (Ridge / Lasso)
- Rank averaging

---

## 🔮 Future Work

- [ ] **Sequence modelling** — Train LSTM / Transformer on sliding depth windows
- [ ] **Multi-target prediction** — Simultaneously predict multiple petrophysical properties
- [ ] **Transfer learning** — Pre-train on public well log databases (FORCE, NOPIMS)
- [ ] **Uncertainty quantification** — Bayesian deep learning for prediction intervals
- [ ] **Real-time inference API** — FastAPI endpoint for live geosteering dashboard
- [ ] **3D formation mapping** — Integrate directional survey for spatial interpolation
- [ ] **Physics-informed features** — Borehole mechanics and formation pressure constraints

---

## 📊 Model Performance

| Model | OOF RMSE | Notes |
|-------|----------|-------|
| LightGBM Baseline | — | To be populated after training |
| LightGBM + Optuna | — | Hyperparameter tuned |
| Ensemble (LGBM + XGB) | — | Weighted blending |

*Leaderboard scores to be added once competition data is available.*

---

## 🗂️ Configuration

All hyperparameters and paths are managed via YAML config files in `configs/`.
No hardcoded paths or magic numbers appear in the source code.

```yaml
# configs/config.yaml — excerpt
validation:
  strategy: group_kfold
  n_folds: 5
  group_column: well_id
  metric: rmse

features:
  rolling_windows: [3, 5, 10, 20, 50]
  lag_steps: [1, 2, 3, 5, 10]
```

---

## 👨‍💻 Author

<div align="center">

**Md Ashraf**
M.Sc (Tech) Applied Geophysics
Indian Institute of Technology (ISM) Dhanbad

*Specialised in subsurface data analytics, petrophysics,
machine learning for geoscience, and MLOps workflows.*

[![GitHub](https://img.shields.io/badge/GitHub-Ashraf--ISM-181717?logo=github)](https://github.com/Ashraf-ISM)

</div>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
<i>Built with ❤️ for AI-driven subsurface analytics and intelligent geosteering.</i>
</div>
