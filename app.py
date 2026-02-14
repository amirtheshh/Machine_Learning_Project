import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import base64
import numpy as np
import json

st.set_page_config(page_title="HDB Resale Price Predictor", layout="wide")


def apply_custom_styling(image_file):
    b64_img = ""
    try:
        with open(image_file, "rb") as img:
            b64_img = base64.b64encode(img.read()).decode()
    except FileNotFoundError:
        pass

    style = f"""
    <style>
    /* Global Background (supports both selectors) */
    .stApp, [data-testid="stApp"] {{
        background-image: url("data:image/png;base64,{b64_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Main Container Glassmorphism (less tight) */
    .block-container {{
        max-width: 1200px;
        background: rgba(0, 0, 0, 0.68);
        backdrop-filter: blur(10px);
        padding: 3rem 3rem;
        border-radius: 20px;
        margin-top: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.10);
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: rgba(20, 20, 20, 0.80) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.10);
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding: 2rem 1.5rem;
    }}

    /* Title & Text Colors */
    h1, h2, h3, p, span, label {{
        color: white !important;
    }}

    /* Prediction Result Box */
    .prediction-box {{
        background: linear-gradient(135deg, #1e7e34, #28a745);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin: 2rem auto;
        text-align: center;
        max-width: 520px;
        border: 1px solid rgba(255, 255, 255, 0.20);
    }}

    /* Predict Button Styling */
    div.stButton > button {{
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        padding: 0.85rem;
        border-radius: 10px;
        border: none;
        transition: 0.3s;
    }}
    div.stButton > button:hover {{
        background-color: #ff3333;
        transform: scale(1.02);
    }}

    /* Expander styling */
    div[data-testid="stExpander"] {{
        background-color: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 10px;
    }}

    /*  Dataframe fix: prevent squished table */
    div[data-testid="stDataFrame"] {{
        width: 100% !important;
    }}
    div[data-testid="stDataFrame"] div[role="grid"] {{
        width: 100% !important;
    }}
    .stTabs [data-testid="stDataFrame"] {{
        width: 100% !important;
    }}
    div[data-testid="stDataFrame"] * {{
        font-size: 14px !important;
    }}
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)

apply_custom_styling("hdb_bg.png")


@st.cache_resource
def load_artifacts():
    model = joblib.load("hgb_best_model.pkl")
    X_sample = joblib.load("X_sample.pkl")
    explainer = shap.TreeExplainer(model, X_sample)
    with open("town_lookup.json", "r") as f:
        town_lookup = json.load(f)
    return model, explainer, X_sample, town_lookup

model, explainer, X_sample, town_lookup = load_artifacts()
FEATURES = list(model.feature_names_in_)  # exact 58 columns in order


BASELINE_TOWN = "ANG MO KIO"
BASELINE_FLAT_MODEL = "Adjoined flat"
BASELINE_FLAT_TYPE = "1 ROOM"

def _ensure_dtype(df_row: pd.DataFrame) -> pd.DataFrame:
    numeric_float = ["latitude", "longitude", "cbd_dist", "floor_area_sqm"]
    numeric_int = ["years_remaining", "transaction_year", "transaction_month"]

    for c in numeric_float:
        if c in df_row.columns:
            df_row[c] = df_row[c].astype(float)

    for c in numeric_int:
        if c in df_row.columns:
            df_row[c] = df_row[c].astype(int)

    for c in df_row.columns:
        if c not in set(numeric_float + numeric_int):
            df_row[c] = df_row[c].astype(bool)

    return df_row

def build_input_row(
    floor_area_sqm: float,
    years_remaining: int,
    transaction_year: int,
    transaction_month: int,
    town: str,
    flat_model: str,
    flat_type: str,
    storey_bin: str,
) -> pd.DataFrame:
    row = {c: False for c in FEATURES}

    lookup_town_key = town
    if town == BASELINE_TOWN and town not in town_lookup:
        lookup_town_key = next(iter(town_lookup.keys()))

    row["latitude"] = float(town_lookup[lookup_town_key]["latitude"])
    row["longitude"] = float(town_lookup[lookup_town_key]["longitude"])
    row["cbd_dist"] = float(town_lookup[lookup_town_key]["cbd_dist"])

    row["floor_area_sqm"] = float(floor_area_sqm)
    row["years_remaining"] = int(years_remaining)
    row["transaction_year"] = int(transaction_year)
    row["transaction_month"] = int(transaction_month)

    # Town OHE 
    if town != BASELINE_TOWN:
        town_col = f"town_{town}"
        if town_col in row:
            row[town_col] = True

    # Flat model OHE 
    if flat_model != BASELINE_FLAT_MODEL:
        fm_col = f"flat_model_{flat_model}"
        if fm_col in row:
            row[fm_col] = True

    # Flat type OHE 
    if flat_type != BASELINE_FLAT_TYPE:
        ft_col = f"flat_type_{flat_type}"
        if ft_col in row:
            row[ft_col] = True

    # Storey bin OHE 
    sb_col = f"storey_range_binned_{storey_bin}"
    if sb_col in row:
        row[sb_col] = True

    input_df = pd.DataFrame([row], columns=FEATURES)
    return _ensure_dtype(input_df)

def shap_waterfall(input_df: pd.DataFrame):
    x = input_df.iloc[0:1]
    sv = explainer.shap_values(x)

    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.array(base).ravel()[0])
    else:
        base = float(base)

    exp = shap.Explanation(
        values=np.array(sv)[0],
        base_values=base,
        data=x.values[0],
        feature_names=x.columns.tolist(),
    )

    fig, ax = plt.subplots(figsize=(20, 6))
    plt.subplots_adjust(left=0.4)
    shap.plots.waterfall(exp, max_display=12, show=False)
    return fig


st.title("🏠 HDB Resale Price Prediction App")

col_info, col_img = st.columns([2, 1])
with col_info:
    st.subheader("🔍 Overview")
    st.write(
        "Welcome to the **HDB Resale Price Predictor**! Use the sliders and dropdowns on the left to input the flat's features, "
        "and the app will estimate its resale price using a machine learning model."
    )
    st.subheader("📊 How it Works")
    st.write(
        "This app uses a machine learning model trained on HDB resale data to predict the resale price of a flat based on its different features. "
        "You can also view your input details and the SHAP values for a better understanding of the model's decision-making process in predicting the price."
    )

st.sidebar.header("🔧 Input Features")
st.sidebar.markdown("Adjust the flat's features below:")

town_options_model = sorted([c.replace("town_", "", 1) for c in FEATURES if c.startswith("town_")])
flat_model_options_model = sorted([c.replace("flat_model_", "", 1) for c in FEATURES if c.startswith("flat_model_")])
flat_type_options_model = sorted([c.replace("flat_type_", "", 1) for c in FEATURES if c.startswith("flat_type_")])

town_options = sorted(list(set([BASELINE_TOWN] + town_options_model)))
flat_model_options = sorted(list(set([BASELINE_FLAT_MODEL] + flat_model_options_model)))
flat_type_options = sorted(list(set([BASELINE_FLAT_TYPE] + flat_type_options_model)))

existing_bins = [c.replace("storey_range_binned_", "", 1) for c in FEATURES if c.startswith("storey_range_binned_")]
storey_bin_options = sorted(["01–09"] + existing_bins)

town = st.sidebar.selectbox("Town", town_options)
flat_model = st.sidebar.selectbox("Flat Model", flat_model_options)
flat_type = st.sidebar.selectbox("Flat Type", flat_type_options)
storey_bin = st.sidebar.selectbox("Storey Range", storey_bin_options)

lookup_town_key = town
if town == BASELINE_TOWN and town not in town_lookup:
    lookup_town_key = next(iter(town_lookup.keys()))


floor_area_sqm = st.sidebar.slider("Floor Area (sqm)", 31.0, 266.0, 96.0, 0.5)
years_remaining = st.sidebar.slider("Lease Years Remaining", 52, 98, 76, 1)
transaction_year = st.sidebar.selectbox("Transaction Year", [2012, 2013, 2014])
transaction_month = st.sidebar.slider("Transaction Month", 1, 12, 6, 1)

input_df = build_input_row(
    floor_area_sqm=floor_area_sqm,
    years_remaining=years_remaining,
    transaction_year=transaction_year,
    transaction_month=transaction_month,
    town=town,
    flat_model=flat_model,
    flat_type=flat_type,
    storey_bin=storey_bin
)

if st.sidebar.button("💰 Predict Resale Price"):
    with st.spinner("Analyzing market data..."):
        price = float(np.expm1(model.predict(input_df)[0]))

    st.markdown(f"""
        <div class="prediction-box">
            <h2 style="margin:0; font-size: 28px;">🏡 Estimated Price</h2>
            <h1 style="margin:0; font-size: 48px; color: #ffffff !important;">${price:,.2f}</h1>
        </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📋 Data Details", "📊 Explainability (SHAP)"])

    
    pretty_cols = ["latitude", "longitude", "town", "cbd_dist", "floor_area_sqm", "years_remaining", "transaction_year", "transaction_month"]

    with tab1:
        display_df = input_df.copy()

    # Convert boolean columns to 0/1 for cleaner viewing
        for col in display_df.columns:
            if display_df[col].dtype == bool:
                display_df[col] = display_df[col].astype(int)

        st.dataframe(display_df, use_container_width=True, height=300)

    with tab2:
        fig = shap_waterfall(input_df)
        st.pyplot(fig)

st.markdown("---")
st.markdown("<p style='text-align: center;'>Created by Kumar Amirtheswaran | MLDP Project</p>", unsafe_allow_html=True)
