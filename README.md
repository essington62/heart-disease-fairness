# 🩺 Heart Disease Fairness — XGBoost Governance Pipeline

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-blue)
![Fairness](https://img.shields.io/badge/Fairness-Aequitas-success)
![Explainability](https://img.shields.io/badge/Explainability-SHAP-lightgrey)
![Governance](https://img.shields.io/badge/Governance-Active-orange)
![Reproducibility](https://img.shields.io/badge/Reproducibility-100%25-brightgreen)

---

### End-to-End Machine Learning Pipeline for Heart Disease Prediction with Model Governance and Fairness Auditing

This repository implements a **reproducible, auditable, and explainable ML workflow** for heart disease prediction.  
The pipeline focuses on:
- Full experiment governance and versioning  
- Bias and fairness auditing using **Aequitas**  
- Explainability with **SHAP** (local + global)  
- Automatic results consolidation (`results_v2.txt`)  
- Reproducible rounds for continuous model evolution  

---

## 💾 Dataset — Heart Disease (UCI Repository)

The dataset used comes from the **UCI Machine Learning Repository**,  
widely known as the *Cleveland Heart Disease Dataset*.

- **Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- **Samples:** 820 (after balancing)
- **Features:** 14 clinical attributes  
  such as `age`, `sex`, `chol`, `thalach`, `cp`, `trestbps`, `oldpeak`, and more.
- **Target variable:** `num`  
  → converted to binary (`label = 1` if `num > 0`, else `0`)

The data was **preprocessed and balanced using SMOTENC** to mitigate class imbalance before training.

---

## 🚀 Pipeline Overview

The full pipeline is organized into **three main stages**, ensuring full traceability and reproducibility:

| Stage | Notebook | Description |
|--------|-----------|-------------|
| **1️⃣ Data Exploration & Preparation** | `EHR_EDA.ipynb` | Loads the dataset, performs exploratory data analysis, cleans and encodes categorical features, applies SMOTENC balancing, and generates consistent `train.csv` and `test.csv` splits for all rounds. Produces summary reports and visual insights for key clinical features. |
| **2️⃣ Training & Governance** | `EHR_BT_v2.ipynb` | Trains **XGBoost** models for each round (`round_01`, `round_02`, `round_03`). Logs hyperparameters, metrics, and metadata automatically to `results_v2.txt`. Implements a governance layer with traceable experiment identifiers and timestamps. |
| **3️⃣ Model & Fairness Analysis** | `model_analysis_v2.ipynb` | Performs post-training evaluation, including: Confusion Matrix, Training Curves, Automatic Overfitting Check, SHAP Explainability (Global + FP/FN), and **Aequitas Fairness Audit**. Generates consolidated analysis summaries and reports. |

---

## 🧠 Model Details

| Parameter | Value |
|------------|--------|
| Model | XGBoostClassifier |
| Objective | `binary:logistic` |
| Estimators | 400 |
| Learning Rate | 0.03 |
| Max Depth | 4 |
| Early Stopping | 30 |
| Subsample | 0.8 |
| Colsample by Tree | 0.8 |
| Scale Pos Weight | 1.2 |
| Device | CPU |
| Random State | 42 |

---

## 🏆 Results — Round 03

Round 03 achieved **perfect classification** performance:

| Metric | Value |
|----------|--------|
| Accuracy | **1.0000** |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1-Score | 1.0000 |
| AUC-ROC | 1.0000 |
| Log Loss | 0.0589 |

**Confusion Matrix — Round 03 (thr = 0.50)**  

| **True / Pred** | **0** | **1** |
|:----------------:|:------:|:------:|
| **0 (No Disease)** | 33 | 0 |
| **1 (Disease)** | 0 | 131 |

> ✅ The model achieved perfect predictions on the test set — no false positives or false negatives.

---

## ⚖️ Fairness Audit (Aequitas Integration)

As part of the model analysis stage, the **Aequitas framework** was used to assess potential bias in predictions.

**Audited attributes:**
- `sex` → comparing `male` and `female`
- `age_group` → `<40`, `40–55`, `55–70`, `70+`

**Disparity metrics:**
- `FOR disparity` — False Omission Rate (FN bias)
- `FPR disparity` — False Positive Rate
- `PPR disparity` — Predicted Positive Rate

💬 Results indicate **no significant bias** between demographic groups,  
demonstrating a fair and balanced classifier.



