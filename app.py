import joblib
import pandas as pd
import streamlit as st
import xgboost
import plotly.graph_objects as go
import time

st.set_page_config(
    page_title="ChurnGuard AI | Final Project",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path: str = r'C:\Users\moham\Desktop\MSC - KFS\Data Scinece - Phase 2\Final Project\best_model.pkl'):
    return joblib.load(model_path)

def build_features(raw: dict) -> pd.DataFrame:
    gender_map = {"Female": 0, "Male": 1}
    subscription_map = {"Basic": 0, "Premium": 1, "Standard": 2}
    contract_map = {"Annual": 0, "Monthly": 1, "Quarterly": 2}

    row = {
        "Age": float(raw["Age"]),
        "Tenure": float(raw["Tenure"]),
        "Usage Frequency": float(raw["Usage Frequency"]),
        "Support Calls": float(raw["Support Calls"]),
        "Payment Delay": float(raw["Payment Delay"]),
        "Total Spend": float(raw["Total Spend"]),
        "Last Interaction": float(raw["Last Interaction"]),
        "Gender": gender_map[raw["Gender"]],
        "Subscription Type": subscription_map[raw["Subscription Type"]],
        "Contract Length": contract_map[raw["Contract Length"]],
    }

    row["Usage_Per_Tenure"] = row["Usage Frequency"] / (row["Tenure"] + 1.0)
    row["Spend_Per_Usage"] = row["Total Spend"] / (row["Usage Frequency"] + 1.0)
    row["Spend_Per_Tenure"] = row["Total Spend"] / (row["Tenure"] + 1.0)
    row["Payment_Delay_Ratio"] = row["Payment Delay"] / 30.0
    row["Engagement_Score"] = (row["Usage Frequency"] * row["Total Spend"]) / 1000.0

    return pd.DataFrame([row])

def reorder_features_for_model(df: pd.DataFrame, model) -> pd.DataFrame:
    if hasattr(model, "feature_names_in_"):
        return df.reindex(columns=list(model.feature_names_in_))
    
    fallback_order = [
        "Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay",
        "Total Spend", "Last Interaction", "Gender", "Subscription Type",
        "Contract Length", "Usage_Per_Tenure", "Spend_Per_Usage",
        "Spend_Per_Tenure", "Payment_Delay_Ratio", "Engagement_Score"
    ]
    return df.reindex(columns=fallback_order)

with st.sidebar:
    st.image(r"image.png")
    st.title("Project Info 🎓")
    st.info("""
    **Student:** Mohamed Younis  
    **Project:** Customer Attrition Analysis  
    **Model:** Drift-Aware Engine
    """)
    st.success("Clean Data: CustomerID removed")

# --- Main Page ---
st.title("📉 ChurnGuard: Intelligent Analytics")
st.markdown("---")

try:
    model = load_model()
except Exception as exc:
    st.error(f"Error loading model: {exc}")
    st.stop()

col1, col2, col3 = st.columns([1, 1, 1], gap="large")

with col1:
    st.subheader("👤 Demographics")
    age = st.slider("Age", 18, 100, 35)
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    tenure = st.number_input("Tenure (Months)", 0, 120, 12)

with col2:
    st.subheader("💳 Billing & Contract")
    sub_type = st.selectbox("Subscription", ["Basic", "Premium", "Standard"])
    contract = st.selectbox("Contract", ["Annual", "Monthly", "Quarterly"])
    spend = st.number_input("Total Spend ($)", 0.0, 100000.0, 500.0)

with col3:
    st.subheader("📞 Engagement")
    calls = st.number_input("Support Calls", 0, 50, 2)
    delay = st.number_input("Payment Delay (Days)", 0, 90, 5)
    last_int = st.number_input("Last Interaction (Days ago)", 0, 365, 15)
    freq = st.number_input("Usage Frequency", 0, 100, 10)

st.markdown("---")
predict_btn = st.button("Calculate Churn Risk 🚀")

if predict_btn:
    with st.spinner('Analyzing customer behavior...'):
        time.sleep(1) 
        
        input_data = {
            "Age": age, "Tenure": tenure, "Usage Frequency": freq,
            "Support Calls": calls, "Payment Delay": delay, "Total Spend": spend,
            "Last Interaction": last_int, "Gender": gender,
            "Subscription Type": sub_type, "Contract Length": contract
        }

        features_df = build_features(input_data)
        features_df = reorder_features_for_model(features_df, model)
        
        prediction = model.predict(features_df)[0]
        prob = model.predict_proba(features_df)[0][1] if hasattr(model, "predict_proba") else 0

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown("### Status")
            if prediction == 1:
                st.error("🚨 HIGH RISK")
                st.write("Customer is likely to leave.")
            else:
                st.success("✅ LOYAL CUSTOMER")
                st.write("Customer is likely to stay.")
                st.balloons()

        with res_col2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Probability %"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#ff4b4b" if prob > 0.5 else "#00cc96"},
                    'steps': [
                        {'range': [0, 30], 'color': "#d1fae5"},
                        {'range': [30, 70], 'color': "#fef3c7"},
                        {'range': [70, 100], 'color': "#fee2e2"}
                    ],
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 Deep Dive: Synthetic Features"):
        st.write("These features were calculated to mitigate **Domain Shift** issues identified during EDA:")
        st.table(features_df)
