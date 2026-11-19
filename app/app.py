"""
Streamlit Web Dashboard for Customer Churn Intelligence Platform
Interactive UI for real-time customer churn risk scoring, what-if analysis,
batch prediction, and model performance visualization.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

# Append project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    SAVED_MODEL_PATH,
    MODEL_METRICS_PATH,
    FIGURES_DIR,
    RAW_DATA_PATH
)
from src.predict import get_predictor

# Set page configuration
st.set_page_config(
    page_title="ChurnGuard AI | Customer Retention Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .high-risk {
        color: #ef4444;
        font-weight: bold;
    }
    .moderate-risk {
        color: #f59e0b;
        font-weight: bold;
    }
    .low-risk {
        color: #10b981;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.title("🛡️ ChurnGuard AI : Customer Retention Intelligence")
st.markdown(
    "**End-to-End Machine Learning System** predicting customer churn risk, explaining attrition drivers, "
    "and recommending proactive retention interventions."
)
st.divider()

# Ensure model is available
try:
    predictor = get_predictor()
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Model artifact not loaded: {e}. Please run `python src/train.py`.")
    model_loaded = False

# Sidebar info
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1551836022-d5d88e9218df?w=600&auto=format&fit=crop&q=60", use_container_width=True)
    st.markdown("### 📊 Platform Overview")
    st.markdown("""
    - **Champion Model**: HistGradientBoosting
    - **Optimization Goal**: ROC-AUC / Recall
    - **Data Preprocessing**: Scikit-learn Pipeline
    - **Deployment**: Local Streamlit / Docker ready
    """)
    st.divider()
    st.markdown("👨‍💻 **Project Author**: Itamio-create")
    st.markdown("🌐 **GitHub**: [Repository](https://github.com/Itamio-create)")

# Tabs for Main View
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Single Customer Risk Profiler",
    "📁 Batch Prediction & Export",
    "📈 Model Diagnostics & Metrics",
    "🏗️ Architecture & Pipeline"
])

# ----------------- TAB 1: Single Prediction -----------------
with tab1:
    st.subheader("Interactive Customer Churn Risk Scorer")
    st.caption("Adjust customer attributes below to evaluate churn probability in real time.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 👤 Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with col2:
        st.markdown("##### 📑 Contract & Billing")
        tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=130.0, value=75.5, step=0.5)
        total_charges = round(monthly_charges * tenure, 2)
        st.caption(f"Estimated Total Charges: **${total_charges:,.2f}**")

    with col3:
        st.markdown("##### 🌐 Subscribed Services")
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No phone service", "No", "Yes"] if phone == "No" else ["No", "Yes"]
        )
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        security = st.selectbox("Online Security", ["No", "Yes"] if internet != "No" else ["No internet service"])
        backup = st.selectbox("Online Backup", ["No", "Yes"] if internet != "No" else ["No internet service"])
        device_prot = st.selectbox("Device Protection", ["No", "Yes"] if internet != "No" else ["No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"] if internet != "No" else ["No internet service"])
        stream_tv = st.selectbox("Streaming TV", ["No", "Yes"] if internet != "No" else ["No internet service"])
        stream_movies = st.selectbox("Streaming Movies", ["No", "Yes"] if internet != "No" else ["No internet service"])

    st.markdown("---")
    predict_btn = st.button("🚀 Calculate Churn Probability", type="primary", use_container_width=True)

    if predict_btn and model_loaded:
        sample_input = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device_prot,
            "TechSupport": tech_support,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        result = predictor.predict_single(sample_input)
        prob = result["churn_probability"]

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown("#### 🎯 Prediction Summary")
            risk_class = (
                "high-risk" if result["risk_tier"] == "High Risk"
                else "moderate-risk" if result["risk_tier"] == "Moderate Risk"
                else "low-risk"
            )

            st.metric(
                label="Predicted Status",
                value=result["churn_prediction"],
                delta=f"{result['risk_tier']} ({result['churn_percentage']})",
                delta_color="inverse" if result["churn_prediction"] == "Churn" else "normal"
            )

            st.progress(prob, text=f"Churn Risk Probability: {prob * 100:.1f}%")

        with res_col2:
            st.markdown("#### 🔍 Key Risk Factors & Drivers")
            for factor in result["key_risk_factors"]:
                st.markdown(f"- ⚠️ {factor}")

            st.markdown("#### 💡 Recommended Retention Action")
            if result["risk_tier"] == "High Risk":
                st.info("🚨 **High Priority Intervention**: Offer annual contract discount voucher (-15%), complimentary premium Tech Support add-on, and follow up via customer success rep.")
            elif result["risk_tier"] == "Moderate Risk":
                st.info("⚡ **Proactive Engagement**: Prompt customer to enroll in automatic bank transfer and bundle streaming entertainment packages.")
            else:
                st.success("✅ **Healthy Account**: Low churn probability. Continue regular engagement and recommend loyalty tier upgrades.")

# ----------------- TAB 2: Batch Prediction -----------------
with tab2:
    st.subheader("Batch Customer File Scoring")
    st.caption("Upload a CSV file of customer profiles to run batch inference and export predictions.")

    sample_csv_btn = st.button("📥 Load Sample 20-Customer Batch")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    df_to_score = None
    if uploaded_file is not None:
        df_to_score = pd.read_csv(uploaded_file)
    elif sample_csv_btn and RAW_DATA_PATH.exists():
        raw_df = pd.read_csv(RAW_DATA_PATH)
        df_to_score = raw_df.sample(20, random_state=42).copy()

    if df_to_score is not None and model_loaded:
        st.write(f"Scoring {len(df_to_score)} customer records...")
        scored_df = predictor.predict_batch(df_to_score)

        st.dataframe(
            scored_df[[
                "customerID", "Contract", "tenure", "MonthlyCharges",
                "Churn_Probability", "Churn_Prediction", "Risk_Tier"
            ] if "customerID" in scored_df.columns else [
                "Contract", "tenure", "MonthlyCharges",
                "Churn_Probability", "Churn_Prediction", "Risk_Tier"
            ]],
            use_container_width=True
        )

        csv_download = scored_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Scored Predictions CSV",
            data=csv_download,
            file_name="customer_churn_predictions.csv",
            mime="text/csv",
            type="primary"
        )

# ----------------- TAB 3: Model Diagnostics -----------------
with tab3:
    st.subheader("Model Validation & Evaluation Metrics")
    st.caption("Benchmarking and diagnostic curves on held-out test data.")

    if MODEL_METRICS_PATH.exists():
        with open(MODEL_METRICS_PATH, "r") as f:
            metrics_data = json.load(f)

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        tm = metrics_data.get("test_metrics", {})
        m_col1.metric("ROC-AUC", f"{tm.get('roc_auc', 0):.4f}")
        m_col2.metric("F1-Score", f"{tm.get('f1_score', 0):.4f}")
        m_col3.metric("Recall", f"{tm.get('recall', 0):.4f}")
        m_col4.metric("Accuracy", f"{tm.get('accuracy', 0):.4f}")

        st.markdown("#### 🔬 Cross-Validation Model Comparison")
        cv_data = metrics_data.get("cross_validation_comparison", {})
        cv_rows = []
        for model_name, stats in cv_data.items():
            cv_rows.append({
                "Algorithm": model_name,
                "5-Fold CV ROC-AUC": f"{stats.get('cv_mean_roc_auc', 0):.4f} ± {stats.get('cv_std_roc_auc', 0):.4f}",
                "Training Time (s)": f"{stats.get('training_time_sec', 0):.2f}"
            })
        st.table(pd.DataFrame(cv_rows))

    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        cm_file = FIGURES_DIR / "confusion_matrix.png"
        if cm_file.exists():
            st.image(str(cm_file), caption="Confusion Matrix on Test Dataset", use_container_width=True)

    with fig_col2:
        roc_file = FIGURES_DIR / "roc_pr_curves.png"
        if roc_file.exists():
            st.image(str(roc_file), caption="ROC & Precision-Recall Curves", use_container_width=True)

# ----------------- TAB 4: Architecture -----------------
with tab4:
    st.subheader("System Architecture & Engineering Pipeline")
    st.markdown("""
    ```mermaid
    flowchart LR
        A[Raw Customer Data] --> B[Data Validation & Preprocessing]
        B --> C[ColumnTransformer]
        C -->|Numerical Scaling| D[StandardScaler & Median Imputer]
        C -->|Categorical Encoding| E[OneHotEncoder & Mode Imputer]
        D --> F[Champion Model: HistGradientBoosting]
        E --> F
        F --> G[Probability Calibration & Risk Classification]
        G --> H[Streamlit UI & REST Inference API]
    ```
    """)
    st.markdown("#### Key Highlights:")
    st.markdown("""
    - **Zero Data Leakage**: All scalers and encoders are strictly fitted only inside cross-validation folds using Scikit-Learn pipelines.
    - **Imbalance Handling**: Handled via sample/class weighting in both linear and tree models.
    - **Container Ready**: Includes Dockerfile and GitHub Actions CI workflow for seamless production deployment.
    """)
