"""
PulsePoint AI — Clinical Risk Dashboard

Three views:
- Command Center: ward-wide overview of every monitored patient, sorted by risk
- Discharge Risk: 30-day readmission risk at time of discharge
- ICU Monitoring: hourly sepsis risk with real-time playback through a patient's actual stay

Run with: streamlit run dashboard/app.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(DASHBOARD_DIR)
MODELS_DIR = os.path.join(REPO_ROOT, "models")
SAMPLE_DIR = os.path.join(DASHBOARD_DIR, "sample_data")

st.set_page_config(page_title="PulsePoint AI", layout="wide", page_icon="🩺")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.risk-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
}
.risk-critical { background-color: #fee2e2; color: #991b1b; border: 2px solid #dc2626; }
.risk-elevated { background-color: #fef3c7; color: #92400e; border: 2px solid #d97706; }
.risk-low { background-color: #d1fae5; color: #065f46; border: 2px solid #059669; }
.sentinel-box {
    background-color: #f0f4ff;
    border-left: 4px solid #4f46e5;
    padding: 14px 18px;
    border-radius: 6px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

READABLE_NAMES = {
    "ICULOS": "time spent in the ICU", "HospAdmTime": "hospital admission timing",
    "Lactate_measured": "a lactate test being ordered", "Temp": "temperature",
    "HR": "heart rate", "Resp": "respiratory rate", "MAP": "mean arterial pressure",
    "SBP": "systolic blood pressure", "DBP": "diastolic blood pressure",
    "Temp_roll6_mean": "recent temperature trend", "Resp_roll6_mean": "recent respiratory rate trend",
    "HR_roll6_mean": "recent heart rate trend", "MAP_roll6_mean": "recent blood pressure trend",
    "WBC": "white blood cell count", "WBC_measured": "a white blood cell test being ordered",
    "Creatinine": "creatinine level", "BUN": "BUN (kidney function) level",
    "Glucose": "glucose level", "Platelets": "platelet count", "Age": "age",
    "discharge_disposition_id": "discharge disposition", "num_lab_procedures": "number of lab procedures",
    "age": "age group", "number_inpatient": "prior inpatient visits",
    "diag_1_group": "primary diagnosis category", "time_in_hospital": "length of stay",
    "num_medications": "number of medications", "admission_type_id": "admission type",
    "admission_source_id": "admission source", "insulin": "insulin regimen",
    "change": "recent medication change", "diabetesMed": "diabetes medication status",
    "max_glu_serum": "glucose serum test result", "A1Cresult": "A1C test result",
    "gender": "gender", "race": "race", "diag_2_group": "secondary diagnosis category",
    "diag_3_group": "additional diagnosis category", "medical_specialty": "admitting specialty",
    "num_procedures": "number of procedures", "number_diagnoses": "number of diagnoses",
    "number_outpatient": "prior outpatient visits", "number_emergency": "prior ER visits",
}


def readable(feat):
    return READABLE_NAMES.get(feat, feat.replace("_", " "))


def risk_badge_html(risk, high=0.5, moderate=0.2):
    if risk >= high:
        return f'<span class="risk-badge risk-critical">🚨 CRITICAL — {risk:.1%}</span>'
    elif risk >= moderate:
        return f'<span class="risk-badge risk-elevated">⚠️ ELEVATED — {risk:.1%}</span>'
    else:
        return f'<span class="risk-badge risk-low">🟢 LOWER RISK — {risk:.1%}</span>'


def sentinel_narrative(shap_series, direction="up"):
    """Generate a plain-language explanation from the top SHAP contributors."""
    if direction == "up":
        top = shap_series[shap_series > 0].head(3)
        if len(top) == 0:
            return "No single factor stands out as a major driver of elevated risk right now."
        phrases = [readable(f) for f in top.index]
    else:
        top = shap_series[shap_series < 0].head(3)
        if len(top) == 0:
            return "No single factor stands out as reducing risk right now."
        phrases = [readable(f) for f in top.index]

    if len(phrases) == 1:
        joined = phrases[0]
    elif len(phrases) == 2:
        joined = f"{phrases[0]} and {phrases[1]}"
    else:
        joined = f"{phrases[0]}, {phrases[1]}, and {phrases[2]}"
    verb = "pushing risk up" if direction == "up" else "helping keep risk down"
    return f"Primarily {verb}: {joined}."


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_diabetes_model():
    model = joblib.load(os.path.join(MODELS_DIR, "diabetes_readmission_model.pkl"))
    encoder = joblib.load(os.path.join(MODELS_DIR, "diabetes_encoder.pkl"))
    with open(os.path.join(MODELS_DIR, "diabetes_feature_names.json")) as f:
        feature_names = json.load(f)
    explainer = joblib.load(os.path.join(MODELS_DIR, "diabetes_shap_explainer.pkl"))
    return model, encoder, feature_names, explainer


@st.cache_resource
def load_sepsis_model():
    model = joblib.load(os.path.join(MODELS_DIR, "sepsis_model.pkl"))
    with open(os.path.join(MODELS_DIR, "sepsis_feature_names.json")) as f:
        feature_names = json.load(f)
    explainer = joblib.load(os.path.join(MODELS_DIR, "sepsis_shap_explainer.pkl"))
    return model, feature_names, explainer


@st.cache_data
def load_diabetes_sample():
    return pd.read_csv(os.path.join(SAMPLE_DIR, "sample_diabetes.csv"))


@st.cache_data
def load_sepsis_sample():
    return pd.read_csv(os.path.join(SAMPLE_DIR, "sample_sepsis.csv"))


@st.cache_data
def score_all_diabetes_patients(_model, _encoder, feature_names, sample_df):
    scores = []
    for idx in sample_df.index:
        X_row = sample_df.loc[[idx]].drop(columns=["patient_display_id", "readmit_30d"])
        cat_cols = X_row.select_dtypes(include="object").columns.tolist()
        X_row[cat_cols] = _encoder.transform(X_row[cat_cols])
        X_row = X_row[feature_names]
        scores.append(_model.predict_proba(X_row)[0, 1])
    return pd.Series(scores, index=sample_df.index)


@st.cache_data
def score_all_sepsis_patients(_model, feature_names, sepsis_df, lookback=5):
    rows = []
    for pid in sorted(sepsis_df["patient_display_id"].unique()):
        patient = sepsis_df[sepsis_df["patient_display_id"] == pid].sort_values("ICULOS").reset_index(drop=True)
        X = patient[feature_names]
        risk_scores = _model.predict_proba(X)[:, 1]
        latest = risk_scores[-1]
        lookback_idx = max(0, len(risk_scores) - 1 - lookback)
        trend = latest - risk_scores[lookback_idx]
        rows.append({
            "patient": pid, "latest_risk": latest, "trend": trend,
            "n_hours": len(patient), "ever_sepsis": patient["SepsisLabel"].max(),
        })
    return pd.DataFrame(rows).sort_values("latest_risk", ascending=False).reset_index(drop=True)


def render_shap_waterfall(shap_values_row, base_value, data_row, feature_names, max_display=8):
    explanation = shap.Explanation(
        values=shap_values_row, base_values=base_value, data=data_row, feature_names=feature_names,
    )
    fig = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(explanation, show=False, max_display=max_display)
    plt.tight_layout()
    return fig


def decision_buttons(key_prefix):
    """Renders Acknowledge / Escalate / Continue Monitoring buttons with sticky feedback."""
    state_key = f"{key_prefix}_decision"
    col1, col2, col3 = st.columns(3)
    if col1.button("✅ Acknowledge", key=f"{key_prefix}_ack"):
        st.session_state[state_key] = "acknowledged"
    if col2.button("🚨 Escalate to team", key=f"{key_prefix}_esc"):
        st.session_state[state_key] = "escalated"
    if col3.button("👁️ Continue monitoring", key=f"{key_prefix}_mon"):
        st.session_state[state_key] = "monitoring"

    decision = st.session_state.get(state_key)
    if decision == "acknowledged":
        st.success("Acknowledged. This case is marked as reviewed.")
    elif decision == "escalated":
        st.error("Escalated. Clinical team would be notified in a live deployment.")
    elif decision == "monitoring":
        st.info("Marked for continued monitoring — no immediate action taken.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🩺 PulsePoint AI")
st.caption(
    "See deterioration before it becomes an emergency. A two-module clinical risk "
    "platform — sepsis early-warning during ICU stays, and 30-day readmission risk "
    "at discharge. Built on public, de-identified data. Prototype for demonstration "
    "only — not a medical device, not deployed on real patients."
)

model, encoder, feature_names, explainer = load_diabetes_model()
sample_df = load_diabetes_sample()
model_s, feature_names_s, explainer_s = load_sepsis_model()
sepsis_df = load_sepsis_sample()

tab_command, tab_discharge, tab_icu = st.tabs(["🏠 Command Center", "📋 Discharge Risk", "🫀 ICU Monitoring"])

# ---------------------------------------------------------------------------
# TAB 0: Command Center
# ---------------------------------------------------------------------------
with tab_command:
    st.subheader("Ward Overview")
    st.write("Every monitored patient, ranked by current risk. Click into a tab to review any individual case.")

    diabetes_scores = score_all_diabetes_patients(model, encoder, feature_names, sample_df)
    sepsis_ward = score_all_sepsis_patients(model_s, feature_names_s, sepsis_df)

    n_high_readmit = (diabetes_scores >= 0.5).sum()
    n_high_sepsis = (sepsis_ward["latest_risk"] >= 0.5).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Discharge patients monitored", len(sample_df))
    c2.metric("ICU patients monitored", len(sepsis_ward))
    c3.metric("High readmission risk", int(n_high_readmit))
    c4.metric("High sepsis risk", int(n_high_sepsis))

    total_alerts = int(n_high_readmit + n_high_sepsis)
    if total_alerts > 0:
        st.error(f"🚨 {total_alerts} patient(s) currently require attention")
    else:
        st.success("No patients currently at critical risk")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.write("**🫀 ICU — Sepsis Risk**")
        display_sepsis = sepsis_ward.copy()
        display_sepsis["Risk"] = display_sepsis["latest_risk"].apply(lambda x: f"{x:.1%}")
        display_sepsis["Trend"] = display_sepsis["trend"].apply(
            lambda x: "↑↑" if x > 0.1 else ("↑" if x > 0.02 else ("↓" if x < -0.02 else "→"))
        )
        st.dataframe(
            display_sepsis[["patient", "Risk", "Trend", "n_hours"]].rename(
                columns={"patient": "Patient", "n_hours": "Hours in ICU"}
            ),
            hide_index=True, use_container_width=True,
        )

    with col_b:
        st.write("**📋 Discharge — Readmission Risk**")
        display_diabetes = sample_df[["patient_display_id"]].copy()
        display_diabetes["Risk"] = diabetes_scores.apply(lambda x: f"{x:.1%}")
        display_diabetes["_sort"] = diabetes_scores.values
        display_diabetes = display_diabetes.sort_values("_sort", ascending=False).drop(columns="_sort")
        st.dataframe(
            display_diabetes.rename(columns={"patient_display_id": "Patient"}),
            hide_index=True, use_container_width=True, height=350,
        )

# ---------------------------------------------------------------------------
# TAB 1: Discharge Risk
# ---------------------------------------------------------------------------
with tab_discharge:
    st.subheader("30-Day Readmission Risk")
    st.write("Select a patient to see their predicted risk of readmission within 30 days of discharge.")

    patient_id = st.selectbox(
        "Select patient", sample_df["patient_display_id"].tolist(), key="diabetes_patient"
    )
    row_idx = sample_df[sample_df["patient_display_id"] == patient_id].index

    X_row = sample_df.loc[row_idx].drop(columns=["patient_display_id", "readmit_30d"])
    cat_cols = X_row.select_dtypes(include="object").columns.tolist()
    X_row[cat_cols] = encoder.transform(X_row[cat_cols])
    X_row = X_row[feature_names]

    risk_prob = model.predict_proba(X_row)[0, 1]
    row = sample_df.loc[row_idx[0]]

    st.markdown(risk_badge_html(risk_prob), unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Patient snapshot**")
        st.write(f"- Age group: {row['age']}")
        st.write(f"- Time in hospital: {row['time_in_hospital']} days")
        st.write(f"- Number of medications: {row['num_medications']}")
        st.write(f"- Prior inpatient visits: {row['number_inpatient']}")
        st.write(f"- Primary diagnosis group: {row['diag_1_group']}")

        decision_buttons("diabetes")

    with col2:
        shap_values = explainer.shap_values(X_row)
        shap_series = pd.Series(shap_values[0], index=feature_names).sort_values(key=abs, ascending=False)

        st.markdown(
            f'<div class="sentinel-box">🤖 <b>PulsePoint says:</b><br>{sentinel_narrative(shap_series, "up")}</div>',
            unsafe_allow_html=True,
        )

        st.write("**Full breakdown**")
        fig = render_shap_waterfall(shap_values[0], explainer.expected_value, X_row.iloc[0], feature_names)
        st.pyplot(fig)
        plt.close(fig)

# ---------------------------------------------------------------------------
# TAB 2: ICU Monitoring — with playback
# ---------------------------------------------------------------------------
with tab_icu:
    st.subheader("Sepsis Early-Warning")
    st.write(
        "Select a patient, then scrub through their actual ICU stay hour-by-hour to watch "
        "risk evolve in real time."
    )

    icu_patient_id = st.selectbox(
        "Select patient", sorted(sepsis_df["patient_display_id"].unique()), key="sepsis_patient"
    )
    patient_series = sepsis_df[sepsis_df["patient_display_id"] == icu_patient_id].sort_values("ICULOS").reset_index(drop=True)

    X_patient = patient_series[feature_names_s]
    risk_scores = model_s.predict_proba(X_patient)[:, 1]
    patient_series = patient_series.assign(risk_score=risk_scores)

    max_hour_idx = len(patient_series) - 1
    hour_idx = st.slider(
        "⏱️ Hour in ICU stay — drag to scrub through this patient's real stay",
        min_value=0, max_value=max_hour_idx, value=max_hour_idx, key="hour_slider"
    )

    current = patient_series.iloc[hour_idx]
    current_risk = current["risk_score"]

    st.markdown(risk_badge_html(current_risk), unsafe_allow_html=True)
    if current["SepsisLabel"] == 1:
        st.caption("📋 This hour is labeled sepsis-positive in the source clinical data.")
    st.write("")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Current vitals**")
        st.write(f"- Heart rate: {current['HR']:.0f} bpm")
        st.write(f"- Respiratory rate: {current['Resp']:.0f} /min")
        st.write(f"- Temperature: {current['Temp']:.1f} °C")
        st.write(f"- MAP: {current['MAP']:.0f} mmHg")
        st.write(f"- Hour of stay: {current['ICULOS']:.0f}")

        st.write("**Data confidence**")
        measured_cols = [c for c in patient_series.columns if c.endswith("_measured")]
        pct_measured = current[measured_cols].mean()
        st.progress(pct_measured, text=f"{pct_measured:.0%} of key labs actually measured this hour")
        if pct_measured < 0.5:
            st.caption("⚠️ Limited lab data this hour — risk score relies more on vitals and carried-forward values.")

        st.write("**What changed (last 4 hours)**")
        lookback_idx = max(0, hour_idx - 4)
        past = patient_series.iloc[lookback_idx]
        for vital, label in [("HR", "Heart rate"), ("Resp", "Resp. rate"), ("Temp", "Temp"), ("MAP", "MAP")]:
            delta = current[vital] - past[vital]
            arrow = "↑" if delta > 0.5 else ("↓" if delta < -0.5 else "→")
            st.write(f"- {label}: {past[vital]:.1f} → {current[vital]:.1f} ({arrow} {delta:+.1f})")

        decision_buttons("sepsis")

    with col2:
        st.write("**Risk trajectory over this stay**")
        chart_data = patient_series[["ICULOS", "risk_score"]].rename(
            columns={"ICULOS": "Hour in ICU", "risk_score": "Sepsis risk"}
        ).set_index("Hour in ICU")
        st.line_chart(chart_data)
        st.caption(f"Currently viewing hour {hour_idx} of {max_hour_idx} — drag the slider above to scrub.")

        X_current = X_patient.iloc[[hour_idx]]
        shap_values_current = explainer_s.shap_values(X_current)
        shap_series_s = pd.Series(shap_values_current[0], index=feature_names_s).sort_values(key=abs, ascending=False)

        st.markdown(
            f'<div class="sentinel-box">🤖 <b>PulsePoint says:</b><br>{sentinel_narrative(shap_series_s, "up")}</div>',
            unsafe_allow_html=True,
        )

        st.write("**Full breakdown**")
        fig = render_shap_waterfall(
            shap_values_current[0], explainer_s.expected_value, X_current.iloc[0], feature_names_s
        )
        st.pyplot(fig)
        plt.close(fig)

st.divider()
st.caption(
    "PulsePoint AI — built on public, de-identified data (PhysioNet/CinC 2019 Sepsis "
    "Challenge; UCI Diabetes 130-US Hospitals). Prototype for demonstration only; not a "
    "medical device and not deployed on real patients."
)
