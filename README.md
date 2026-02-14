# 🏘️ HDB Resale Price Predictor – Machine Learning Project

![Python](https://img.shields.io/badge/Built%20With-Python%20%7C%20scikit--learn%20%7C%20Streamlit-blue)
![Model](https://img.shields.io/badge/Champion-HistGradientBoostingRegressor-orange)
![Explainability](https://img.shields.io/badge/Explainability-SHAP-purple)

This is a **Machine Learning Project**. It builds a supervised regression model to predict **Singapore HDB resale prices**, using a disciplined pipeline (EDA → preprocessing → feature engineering → model selection → CV → tuning → evaluation → Streamlit-ready artifacts).

---

## 📊 Objective

Predict **HDB resale price (SGD)** from key flat and location-related features, while prioritising:
- **Robust Generalisation** (validated with CV + learning curves)
- **Deployment Practicality** (user-friendly inputs for Streamlit)

---

## 🧰 Tools & Libraries Used

- **Python**, **Pandas**, **NumPy**
- **Scikit-learn**
  - Baselines + robust models: `LinearRegression`, `Ridge`, `HuberRegressor`
  - Tree/ensemble models: `DecisionTreeRegressor`, `GradientBoostingRegressor`, `HistGradientBoostingRegressor`
  - Model selection: `KFold`, `cross_validate`
  - Tuning: `GridSearchCV`, `RandomizedSearchCV`
- **Matplotlib** (visualisations)
- **SHAP** (model explainability)
- **Streamlit** (interactive web app)
- **Joblib / JSON** (artifact saving)

---

## 📈 Project Workflow

### 1) 🧹 Data Preparation & Cleaning
- Verified datatypes and handled missing values.
- Removed high-cardinality / non-deployable identifiers to reduce noise and OHE explosion:
  - `postal_code`, `address`, `block`, `street_name`
- Dropped redundant columns after feature engineering (e.g., raw `month`, raw `storey_range`).

---

### 2) 🔎 Exploratory Data Analysis (EDA)
EDA focused on clear, decision-making insights:
- **Resale price distribution** showed right-skew → motivated log target transform.
- **Flat type vs resale price** (boxplots) showed strong separation across categories.
- Town-level checks (volume vs median price) highlighted market segmentation.

---

### 3) 🏗️ Feature Engineering
- Extracted from `month`:
  - `transaction_year`, `transaction_month`
- Fixed inconsistent storey labels by creating:
  - `storey_range_binned` (clean, non-overlapping bins)

---

### 4) 🔤 Encoding Strategy (Deployment-friendly)
Used **One-Hot Encoding** for nominal categories:
- `town`, `flat_model`, `flat_type`, `storey_range_binned`

> Note: Some categories are treated as **baseline (reference)** due to `drop_first=True` logic. In the Streamlit app, these baselines are still selectable and correctly encoded as “all zeros” for that category group.

---

### 5) 🎯 Target Transformation (Critical Improvement)
Model training is done on:
- `y_log = log1p(resale_price)`

Predictions are converted back to SGD using:
- `expm1(y_pred_log)`

This improves stability and reduces heteroscedasticity common in housing prices.

---

## 🤖 Models Compared
A focused set of models was evaluated (MAE in **SGD** + R²):
- `LinearRegression`, `Ridge`, `HuberRegressor`
- `DecisionTreeRegressor`
- `GradientBoostingRegressor`
- `HistGradientBoostingRegressor`

**Top performer:** `HistGradientBoostingRegressor` (best balance of accuracy + stability + speed).

---

## 🔁 Cross-Validation + Learning Curves
- **5-fold K-Fold CV** confirmed model stability.
- Learning curves were used to detect fit quality:
  - DecisionTree showed overfitting
  - GradientBoosting was comparatively weaker
  - HistGradientBoosting had the best generalisation tradeoff

---

## 🛠️ Hyperparameter Tuning 
Two finalists were tuned using different strategies based on runtime efficiency:
- **HistGradientBoostingRegressor → GridSearchCV**
  - Fast model, systematic search feasible.
- **GradientBoostingRegressor → RandomizedSearchCV**
  - Slower model, budgeted search for efficiency.

Final holdout results confirmed tuned **HistGradientBoosting** as champion.

---

## 🏁 Final Deployment Model Performance

After hyperparameter tuning, the final **HistGradientBoostingRegressor** model was retrained using the best parameters and a deployment-friendly feature set.

To improve usability in the Streamlit application, the feature `closest_mrt_dist` was removed (as end-users cannot reliably input this value). The model was then retrained and evaluated on the holdout test set.

### 📊 Final Holdout Results (Deployment Version)

| Metric | Value |
|--------|-------|
| **MAE (SGD)** | **15,442.30** |
| **R² (SGD)** | **0.9705** |

### 🔎 Interpretation

- The model achieves an average prediction error of approximately **$15.4K**.
- It explains **97.05% of the variance** in resale prices.
- Removing `closest_mrt_dist` did **not significantly degrade performance**, demonstrating that the model remains robust even after simplifying inputs for deployment.

---

## 🧠 Explainability (SHAP)
SHAP is included for:
- **Global importance** (bar plot)
- **Beeswarm** (direction + impact spread)
- **Per-prediction waterfall** inside Streamlit

> Important: SHAP values reflect **impact in log-price space** (because the model is trained on log(target)), so explanations are used for ranking and direction—not direct “SGD contribution” claims.

---

## 🚀 Streamlit Web App
A Streamlit app was built with:
- Clean UI + background styling
- User inputs for flat attributes (area, lease, year/month, flat type/model, storey bin, town)
- **Town lookup auto-fill** for deployment usability:
  - `latitude`, `longitude`, `cbd_dist` are derived from `town_lookup.json`
- Handles **baseline categories** correctly (reference groups with “all zeros” encoding)
- Displays:
  - Predicted resale price (SGD)
  - Input summary table
  - SHAP waterfall explanation

---

## 📦 Final Deployment Artifacts
Saved for reproducibility + app loading:
- `hgb_best_model.pkl` *(final tuned + retrained model)*
- `X_sample.pkl` *(sample for SHAP baseline / speed)*
- `town_lookup.json` *(median geo + cbd distance per town)*

---

## ✅ Conclusion
The final solution uses **HistGradientBoostingRegressor** because it delivered the best overall balance of:
- low **MAE (SGD)**
- high **R²**
- stable CV performance
- fast tuning/training
- deployment-friendly feature design

---

