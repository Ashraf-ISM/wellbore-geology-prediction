# AI Geosteering & Wellbore Geology Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Geophysics](https://img.shields.io/badge/Geophysics-Subsurface-green)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF)
![Status](https://img.shields.io/badge/Status-Active-success)

</div>

---

## Overview

This repository contains my solution framework for the **ROGII Wellbore Geology Prediction Challenge** hosted on Kaggle.

The objective of this competition is to develop machine learning models capable of predicting geological positioning along horizontal wellbores using drilling and subsurface data. Accurate prediction of geological layers during drilling can significantly improve geosteering efficiency, reduce drilling uncertainty, minimize operational costs, and optimize hydrocarbon recovery.

This project combines:

- Machine Learning
- Sequential Modeling
- Well Log Analytics
- Petrophysical Interpretation
- Geosteering Concepts
- Subsurface Data Analytics
- Feature Engineering for Drilling Data

---

# Problem Statement

Horizontal drilling operations require continuous interpretation of subsurface geology while drilling. However, direct visualization of the subsurface is impossible, and geological interpretation relies on indirect measurements such as logging and drilling data.

The goal of this project is to predict the target variable:

- **TVD / TVT related geological positioning**

using available drilling and subsurface measurements.

---

# Objectives

- Build robust ML models for wellbore geology prediction
- Develop leakage-free validation strategies
- Explore geological feature engineering
- Investigate sequence-based learning approaches
- Compare boosting and deep learning architectures
- Create ensemble-based predictive systems

---

# Planned Workflow

## 1. Exploratory Data Analysis (EDA)

- Well-wise visualization
- Missing value analysis
- Geological trend analysis
- Correlation studies
- Distribution analysis

---

## 2. Feature Engineering

Planned features include:

- Rolling statistics
- Log derivatives
- Trajectory curvature
- Lag features
- Formation transition indicators
- Neighbor-aware contextual features
- Geological continuity measures

---

## 3. Machine Learning Models

### Baseline Models
- LightGBM
- XGBoost
- CatBoost

### Deep Learning Models
- LSTM
- GRU
- Temporal CNN
- Transformer-based architectures

---

## 4. Validation Strategy

Special care is taken to avoid data leakage using:

- GroupKFold validation
- Well-based splitting
- Sequential validation

---

## 5. Ensemble Learning

Final predictions may combine:

- Gradient Boosting Models
- Sequence Models
- Statistical Aggregation
- Blending and Stacking

---

# Repository Structure

```bash
├── data/
├── notebooks/
├── src/
├── configs/
├── outputs/
├── docs/
└── assets/
```

---

# Tech Stack

## Languages
- Python

## Libraries
- pandas
- numpy
- scikit-learn
- lightgbm
- xgboost
- catboost
- pytorch
- optuna
- matplotlib
- plotly

---

# Potential Advanced Ideas

- Physics-guided ML
- Electrofacies-assisted prediction
- Geological sequence modeling
- Multi-scale contextual learning
- Hybrid ML + Geoscience workflows

---

# Competition

ROGII - Wellbore Geology Prediction  
Hosted on Kaggle

Competition Link:
https://kaggle.com/competitions/rogii-wellbore-geology-prediction

---

# Author

## Md Ashraf

M.Sc (Tech) Applied Geophysics  
Indian Institute of Technology (ISM) Dhanbad

### Research Interests
- Machine Learning in Geoscience
- Petrophysics
- Seismology
- Geophysical Inversion
- Subsurface Analytics
- AI for Energy Applications

---

# Current Status

🚧 Project Under Active Development

---

# Future Work

- Real-time geosteering systems
- Uncertainty-aware prediction
- Explainable AI for drilling analytics
- Digital subsurface twin development

---

# License

This project is intended for research and educational purposes.
