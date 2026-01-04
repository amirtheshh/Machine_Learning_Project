# 🏘️ HDB Resale Price Predictor – Machine Learning for Developers Project

![Python](https://img.shields.io/badge/Built%20With-Python%20%7C%20HistGradientBoosting%20%7C%20Streamlit-blue)
![Deployment](https://img.shields.io/badge/Deployed-On%20Streamlit-green)

This project was developed as part of the **Machine Learning for Developers Project (MLDP)** module. It focuses on building and deploying a machine learning model to predict **Singapore HDB resale prices**. After extensive model evaluation and tuning, the final model was deployed with **Streamlit** for interactive use.

---

## 📊 Objective

The objective was to create a machine learning model that can predict HDB resale prices using input features about the flat. 

---

## 🧰 Tools & Libraries Used

- **Python**, **Pandas**, **NumPy**
- **Scikit-learn**:
  - Linear models: `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `HuberRegressor`, `TheilSenRegressor`
  - Tree-based models: `DecisionTreeRegressor`, `ExtraTreesRegressor`
  - Ensemble models: `GradientBoostingRegressor`, `AdaBoostRegressor`, `HistGradientBoostingRegressor`
  - Support Vector: `SVR`
- **Matplotlib**, **Seaborn** (Visulisations)
- **SHAP** (Model explainability)
- **Streamlit** (Web app deployment)
- **Joblib** (Model serialization)

---

## 📈 Project Workflow

### 1. 🧹 Data Preprocessing
- Cleaned dataset (`hdb_resale_price.csv`)
- Handled missing data, outliers (95th percentile filter)
- Extracted features like transaction_year, transaction_month
- Created new columns such as storey_range_binned
- Encoded categorical features (Label + OneHot)
- Removed low-importance or high-cardinality features

### 2. 🏗️ Feature Engineering
- Derived `transaction_year` and `transaction_month` from month column
- Created `storey_range_binned` from storey range column
- Transformed `postal_code` to correct datatype

### 3. 🤖 Models Trained & Evaluated

A total of **12 models** were trained and compared using consistent preprocessing pipelines. The following models were the top 4: 

| Model                      | Notes                                                                 |
|---------------------------|------------------------------------------------------------------------|
| **HistGradientBoosting**   | ⭐ Final selected model – fast, scalable, and highly accurate           |



---

## 🏆 Final Model: HistGradientBoostingRegressor

After comparing 12 models, **HistGradientBoosting** was chosen due to:

- ✅ High accuracy and fast training
- ✅ Good performance on unseen data
- ✅ Easy integration into pipeline and deployment


- Final model serialized as `hdb_resale_price_hgb.pkl`

---

## 🚀 Streamlit Web App

A frontend was built using **Streamlit**, which allows users to:

- Input housing attributes (flat type, years_remaining, storey_range etc.)
- Instantly view the predicted resale price
- View input features
- App uses SHAP to explain prediction logic

---

## 🧪 Model Evaluation Techniques

- K-Fold Cross-Validation
- Learning Curve Analysis
- MAE, MSE, RMSE, R² metrics
- Residual analysis, scatter plots
- SHAP summary and force plots
