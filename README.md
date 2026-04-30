<div align="center">
  <h1>Customer Churn Prediction 📉🤖💙</h1>
  <p>End-to-end churn analysis and modeling (EDA 🔍 → preprocessing 🧹 → modeling ⚙️ → best model export 💾) using Python.</p>

  <p>
    <a href="#overview">Overview 🧾</a> •
    <a href="#business-problem">Business Problem 🎯</a> •
    <a href="#project-structure">Project Structure 🗂️</a> •
    <a href="#dataset">Dataset 🧩</a> •
    <a href="#notebook-journey">Notebook Journey 📒</a> •
    <a href="#results">Results 📊</a> •
    <a href="#challenges">Challenges ⚠️</a>
  </p>

  <p>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.x-blue" />
    <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebook-orange" />
    <img alt="Pandas" src="https://img.shields.io/badge/Pandas-Data_Analysis-150458" />
    <img alt="Scikit-Learn" src="https://img.shields.io/badge/scikit--learn-ML-F7931E" />
    <img alt="XGBoost" src="https://img.shields.io/badge/XGBoost-Modeling-EC6B23" />
  </p>
</div>

---

## Table of Contents 🧭

- [Overview 🧾](#overview)
- [Business Problem 🎯](#business-problem)
- [Project Structure 🗂️](#project-structure)
- [Dataset 🧩](#dataset)
- [Notebook Journey 📒](#notebook-journey)
- [Results 📊](#results)
- [Challenges ⚠️](#challenges)
- [Streamlit Preview 🖼️](#streamlit-preview)
- [How to Run ▶️](#how-to-run)
- [Author ✍️](#author)

---

<a id="overview"></a>
## Overview 🧾

This repository focuses on **Customer Churn Prediction**, building a full business-oriented machine learning workflow:

- **Data Understanding 🔍**: Explore train/test behavior and target distribution.
- **Data Preparation 🧹**: Clean, align, and prepare data for modeling.
- **Feature Engineering ⚙️**: Build stronger behavioral features to improve robustness.
- **Modeling & Comparison 📈**: Train and compare multiple models fairly.
- **Model Export 💾**: Automatically save the best model for later use.

---

<a id="business-problem"></a>
## Business Problem 🎯

The company needs to reduce customer churn and retain high-value customers.

Without an early warning model:

- Retention campaigns become expensive and less targeted 💸
- Teams react too late to prevent churn ⏰
- Revenue and CLV are negatively impacted 📉

Goal: predict whether a customer will churn (`Churn = 1`) to enable proactive retention actions 🤝

---

<a id="project-structure"></a>
## Project Structure 🗂️

- [`Datasets/` 🧩](#dataset) — project datasets
  - `train.csv`
  - `test.csv`
- [`notebook.ipynb` 📒](#notebook-journey) — full analysis and modeling workflow
- `README.md` — project documentation
- `best_model.pkl` — exported best model (generated after running final notebook cells)

Tree 🌳:

```text
Project/
├─ Datasets/
│  ├─ train.csv
│  └─ test.csv
├─ notebook.ipynb
├─ README.md
├─ image.png 
├─ app.py
└─ best_model.pkl
```

---

<a id="dataset"></a>
## Dataset 🧩

The project uses customer-level records for churn classification.

Target Column 🎯:
- **`Churn`**: Binary target (`0 = No Churn`, `1 = Churn`)

Example feature groups 🧾:
- Customer profile (demographic-like fields)
- Usage behavior (frequency/intensity indicators)
- Service experience (e.g., support interactions)
- Payment/spend behavior (delay and spending patterns)

---

<a id="notebook-journey"></a>
## Notebook Journey 📒

The notebook is organized as a clear end-to-end flow:

1. **Required Libraries** 📦  
2. **Read the Dataset** 📥  
3. **Exploratory Data Analysis (EDA)** 🔍  
   - info, describe, missing values, duplicates, target checks  
   - numeric/categorical visual analysis  
   - churn-rate exploration by groups  
4. **Data Preprocessing** 🧹  
   - drop unnecessary columns  
   - categorical encoding  
   - feature engineering (`Usage_Per_Tenure`)  
5. **Split Features/Target** ✂️  
6. **Train/Test Split** ⚙️  
7. **Modeling** 🤖  
   - Random Forest  
   - Logistic Regression  
   - XGBoost  
8. **Modeling Summary & Best Model Selection** 🏆  
   - compare models on key churn metrics  
   - save best model to `best_model.pkl`

---

<a id="results"></a>
## Results 📊

### Modeling Output ✅

- All candidate models are trained and evaluated.
- A final summary table compares:
  - `Accuracy`
  - `Precision (Churn=1)`
  - `Recall (Churn=1)`
  - `F1 (Churn=1)`

### Best Model Selection 🏆

- The best model is selected based on the highest **`F1 (Churn=1)`**.
- Final artifact is saved as:
  - `best_model.pkl`

---

<a id="challenges"></a>
## Challenges ⚠️

- **Data consistency issues**: required cleaning and type handling.
- **Domain shift risk**: noticeable distribution differences between train/test in some features.
- **Metric trade-off**: balancing precision and recall is critical in churn use cases.

These challenges are addressed through careful EDA, feature engineering, and model comparison strategy 💡

---

<a id="streamlit-preview"></a>
## Streamlit Preview 🖼️

<img width="1919" height="885" alt="Screenshot 2026-04-30 105140" src="https://github.com/user-attachments/assets/c7d5bb81-15c3-4582-8fc0-5af2defee133" /> 

```bash
streamlit run app.py
```

---

<a id="how-to-run"></a>
## How to Run ▶️

### 1) Setup Environment 🧪
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 2) Install Dependencies 📦
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
```

### 3) Launch Notebook 🚀
```bash
jupyter notebook notebook.ipynb
```

Run all cells in order, then verify `best_model.pkl` is generated.

---

<a id="author"></a>
## Author ✍️

- **Name**: Mohammed Younis
- **Program**: MSC KFS - Data Science Phase 2 💙
