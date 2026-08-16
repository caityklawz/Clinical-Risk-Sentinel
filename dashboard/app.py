"""
Clinical Risk Sentinel — Streamlit Dashboard

Two-tab clinical risk platform:
- Discharge Risk: 30-day readmission risk at time of discharge
- ICU Monitoring: hourly sepsis risk during an ICU stay

Run with: streamlit run dashboard/app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

# ---------------------------------------------------------------------------
# Paths (works whether streamlit is launched from repo root or dashboard/)
# ---------------------------------------------------------------------------
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(DASHBOARD_DIR)
MODELS_DIR = os.path.join(REPO_ROOT, "models")
SAMPLE_DIR = os.path.join(DASHBOARD_DIR, "sample_data")

st.set_page_config(page_title="Clinical Risk Sentinel", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders — models and data only load once per session, not per click
# ---------------------------------------------------------------------------
@st.cache_resource
def load_diabetes_model():
    model = joblib.load(os.path.join(MODELS_DIR, "diabetes_readmission_model.pkl"))
    encoder = joblib.load(os.path.join(MODELS_DIR, "diabetes_encoder.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "diabetes_feature_names.pkl"))
    explainer = joblib.load(os.path.join(MODELS_DIR, "diabetes_shap_explainer.pkl"))
    return model, encoder, feature_names, explainer


@st.cache_resource
def load_sepsis_model():
    model = joblib.load(os.path.join(MODELS_DIR, "sepsis_model.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "sepsis_feature_names.pkl"))
    explainer = joblib.load(os.path.join(MODELS_DIR, "sepsis_shap_explainer.pkl"))
    return model, feature_names, explainer


@st.cache_data
def load_diabetes_sample():
    return pd.read_csv(os.path.join(SAMPLE_DIR, "sample_diabetes.csv"))


@st.cache_data
def load_sepsis_sample():
    return pd.read_csv(os.path.join(SAMPLE_DIR, "sample_sepsis.csv"))


def render_shap_waterfall(explainer, shap_values_row, base_value, data_row, feature_names):
    """Renders a SHAP waterfall plot into a matplotlib figure for Streamlit to display."""
    explanation = shap.Explanation(
        values=shap_values_row,
        base_values=base_value,
        data=data_row,
        feature_names=feature_names,
    )
    fig = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(explanation, show=False, max_display=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏥 Clinical Risk Sentinel")
st.caption(
    "A two-module clinical risk platform — sepsis early-warning during ICU stays, "
    "and 30-day readmission risk at discharge. Trained on public, de-identified data. "
    "Not deployed on real patients."
)

tab_discharge, tab_icu = st.tabs(["📋 Discharge Risk", "🫀 ICU Monitoring"])

# ---------------------------------------------------------------------------
# TAB 1: Discharge Risk (readmission)
# ---------------------------------------------------------------------------
with tab_discharge:
    st.subheader("30-Day Readmission Risk")
    st.write(
        "Select a patient below to see their predicted risk of readmission within 30 "
        "days of discharge, along with the factors driving that score."
    )

    model, encoder, feature_names, explainer = load_diabetes_model()
    sample_df = load_diabetes_sample()

    patient_id = st.selectbox(
        "Select patient", sample_df["patient_display_id"].tolist(), key="diabetes_patient"
    )
    row_idx = sample_df[sample_df["patient_display_id"] == patient_id].index
    row = sample_df.loc[row_idx[0]]

    # Build model input — slice with .loc[[...]] (not a Series .to_frame().T) so
    # original column dtypes are preserved and the encoder sees the right categoricals.
    X_row = sample_df.loc[row_idx].drop(columns=["patient_display_id", "readmit_30d"])
    cat_cols = X_row.select_dtypes(include="object").columns.tolist()
    X_row[cat_cols] = encoder.transform(X_row[cat_cols])
    X_row = X_row[feature_names]

    risk_prob = model.predict_proba(X_row)[0, 1]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Predicted 30-day readmission risk", f"{risk_prob:.1%}")
        if risk_prob >= 0.5:
            st.error("⚠️ Elevated risk")
        elif risk_prob >= 0.25:
            st.warning("Moderate risk")
        else:
            st.success("Lower risk")

        st.write("**Patient snapshot**")
        st.write(f"- Age group: {row['age']}")
        st.write(f"- Time in hospital: {row['time_in_hospital']} days")
        st.write(f"- Number of medications: {row['num_medications']}")
        st.write(f"- Prior inpatient visits: {row['number_inpatient']}")
        st.write(f"- Primary diagnosis group: {row['diag_1_group']}")

    with col2:
        st.write("**Why this score — top contributing factors**")
        shap_values = explainer.shap_values(X_row)
        fig = render_shap_waterfall(
            explainer, shap_values[0], explainer.expected_value, X_row.iloc[0], feature_names
        )
        st.pyplot(fig)
        plt.close(fig)

# ---------------------------------------------------------------------------
# TAB 2: ICU Monitoring (sepsis)
# ---------------------------------------------------------------------------
with tab_icu:
    st.subheader("Sepsis Early-Warning")
    st.write(
        "Select an ICU patient to see their vital-sign trends over their stay and their "
        "current hourly sepsis risk score."
    )

    model_s, feature_names_s, explainer_s = load_sepsis_model()
    sepsis_df = load_sepsis_sample()

    icu_patient_id = st.selectbox(
        "Select patient", sorted(sepsis_df["patient_display_id"].unique()), key="sepsis_patient"
    )
    patient_series = sepsis_df[sepsis_df["patient_display_id"] == icu_patient_id].sort_values("ICULOS")

    # Predict risk for every hour of this patient's stay
    X_patient = patient_series[feature_names_s]
    risk_scores = model_s.predict_proba(X_patient)[:, 1]
    patient_series = patient_series.assign(risk_score=risk_scores)

    latest = patient_series.iloc[-1]
    latest_risk = latest["risk_score"]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Current hour sepsis risk", f"{latest_risk:.1%}")
        if latest_risk >= 0.5:
            st.error("⚠️ High risk — clinical review recommended")
        elif latest_risk >= 0.2:
            st.warning("Elevated risk — monitor closely")
        else:
            st.success("Lower risk")

        st.write("**Current vitals**")
        st.write(f"- Heart rate: {latest['HR']:.0f} bpm")
        st.write(f"- Respiratory rate: {latest['Resp']:.0f} /min")
        st.write(f"- Temperature: {latest['Temp']:.1f} °C")
        st.write(f"- MAP: {latest['MAP']:.0f} mmHg")
        st.write(f"- Hours in ICU: {latest['ICULOS']:.0f}")
        if latest["SepsisLabel"] == 1:
            st.caption("📋 This hour is labeled sepsis-positive in the source data.")

    with col2:
        st.write("**Risk score trend over this ICU stay**")
        chart_data = patient_series[["ICULOS", "risk_score"]].rename(
            columns={"ICULOS": "Hour in ICU", "risk_score": "Sepsis risk"}
        ).set_index("Hour in ICU")
        st.line_chart(chart_data)

    st.write("**Why the current score — top contributing factors**")
    X_latest = X_patient.iloc[[-1]]
    shap_values_latest = explainer_s.shap_values(X_latest)
    fig = render_shap_waterfall(
        explainer_s, shap_values_latest[0], explainer_s.expected_value,
        X_latest.iloc[0], feature_names_s
    )
    st.pyplot(fig)
    plt.close(fig)

st.divider()
st.caption(
    "Clinical Risk Sentinel — built on public, de-identified data (PhysioNet/CinC 2019 "
    "Sepsis Challenge; UCI Diabetes 130-US Hospitals). Prototype for demonstration only; "
    "not a medical device and not deployed on real patients."
)
