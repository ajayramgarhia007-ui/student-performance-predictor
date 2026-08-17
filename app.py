"""
app.py
------
Streamlit web app for the Student Performance Predictor.

Run with:
    streamlit run app.py

Requires models/best_regressor.joblib, models/best_classifier.joblib,
models/category_classifier.joblib, models/regression_coefficients.json
and models/metadata.json to already exist (run train_model.py first).

Features:
  - Animated intro splash screen
  - Theme selector (Light / Dark / Ocean / Sunset)
  - Single-student prediction with sliders/dropdowns and a name field
  - "Why this prediction?" contribution chart (explainability)
  - Performance category badge (Poor -> Excellent)
  - Batch prediction from an uploaded CSV, with downloadable results
  - In-session prediction history with a trend chart
"""

import json
import io
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

# ---------------------------------------------------------------------
# Theme selector — a few flat color palettes applied via CSS overrides.
# Chosen in the sidebar, persisted across reruns via session_state.
# ---------------------------------------------------------------------
THEMES = {
    "Light":  {"bg": "#ffffff", "text": "#1a1a1a", "accent": "#e74c3c", "card": "#f5f5f5"},
    "Dark":   {"bg": "#0e1117", "text": "#f1f1f1", "accent": "#ff6b6b", "card": "#1c1f26"},
    "Ocean":  {"bg": "#eef6fb", "text": "#0b2545", "accent": "#1f77b4", "card": "#dcedf9"},
    "Sunset": {"bg": "#fff8f2", "text": "#4a1b0c", "accent": "#e8593c", "card": "#fbe8dc"},
}

if "theme_name" not in st.session_state:
    st.session_state.theme_name = "Light"

with st.sidebar:
    st.markdown("### 🎨 Appearance")
    st.session_state.theme_name = st.selectbox(
        "Theme", list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme_name),
    )

_theme = THEMES[st.session_state.theme_name]
st.markdown(f"""
<style>
.stApp {{
    background-color: {_theme['bg']};
    color: {_theme['text']};
}}
[data-testid="stSidebar"] {{
    background-color: {_theme['card']};
}}
h1, h2, h3, h4, p, label, .stMarkdown, span {{
    color: {_theme['text']};
}}
.stButton>button {{
    background-color: {_theme['accent']};
    color: #ffffff;
    border: none;
}}
.stButton>button:hover {{
    opacity: 0.85;
    color: #ffffff;
}}
[data-testid="stMetricValue"] {{
    color: {_theme['accent']};
}}
div[data-baseweb="tab-list"] button[aria-selected="true"] {{
    color: {_theme['accent']};
    border-bottom-color: {_theme['accent']};
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Animated intro splash (shown once per browser session, like an app's
# opening logo animation). Uses session_state so it doesn't replay on
# every button click — only on a fresh page load.
# ---------------------------------------------------------------------
if "intro_done" not in st.session_state:
    st.session_state.intro_done = False

if not st.session_state.intro_done:
    st.markdown("""
    <style>
    .spd-splash {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 65vh;
        animation: spd-fade 2.3s ease forwards;
    }
    .spd-logo {
        font-size: 96px;
        line-height: 1;
        animation: spd-pop 0.9s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
    }
    .spd-title {
        font-size: 30px;
        font-weight: 600;
        margin-top: 14px;
        opacity: 0;
        animation: spd-slide-up 0.7s ease 0.45s forwards;
    }
    .spd-sub {
        font-size: 15px;
        color: #9a9a9a;
        margin-top: 6px;
        opacity: 0;
        animation: spd-slide-up 0.7s ease 0.75s forwards;
    }
    .spd-bar-track {
        width: 160px;
        height: 4px;
        background: rgba(150,150,150,0.25);
        border-radius: 4px;
        margin-top: 22px;
        overflow: hidden;
        opacity: 0;
        animation: spd-slide-up 0.7s ease 1.0s forwards;
    }
    .spd-bar-fill {
        height: 100%;
        width: 0%;
        background: #e74c3c;
        border-radius: 4px;
        animation: spd-load 1.3s ease 1.0s forwards;
    }
    @keyframes spd-pop {
        0% { transform: scale(0) rotate(-8deg); opacity: 0; }
        60% { transform: scale(1.15) rotate(4deg); opacity: 1; }
        100% { transform: scale(1) rotate(0deg); opacity: 1; }
    }
    @keyframes spd-slide-up {
        from { transform: translateY(14px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    @keyframes spd-load {
        from { width: 0%; }
        to { width: 100%; }
    }
    @keyframes spd-fade {
        0%, 78% { opacity: 1; }
        100% { opacity: 0; }
    }
    </style>
    <div class="spd-splash">
        <div class="spd-logo">🎓</div>
        <div class="spd-title">Student Performance Predictor</div>
        <div class="spd-sub">Powered by Machine Learning</div>
        <div class="spd-bar-track"><div class="spd-bar-fill"></div></div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(2.3)
    st.session_state.intro_done = True
    st.rerun()


# ---------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    regressor = joblib.load("models/best_regressor.joblib")
    classifier = joblib.load("models/best_classifier.joblib")
    category_clf = joblib.load("models/category_classifier.joblib")
    with open("models/metadata.json") as f:
        metadata = json.load(f)
    with open("models/regression_coefficients.json") as f:
        coef_data = json.load(f)
    return regressor, classifier, category_clf, metadata, coef_data


regressor, classifier, category_clf, metadata, coef_data = load_artifacts()

CATEGORY_COLORS = {
    "Poor": "#d62728",
    "Below Average": "#ff7f0e",
    "Average": "#bcbd22",
    "Good": "#2ca02c",
    "Excellent": "#1f77b4",
}

if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def make_input_row(study_hours, attendance, previous_score, sleep_hours, absences,
                    parental_education, internet_access, extracurricular,
                    part_time_job, study_method):
    return pd.DataFrame([{
        "study_hours_per_week": study_hours,
        "attendance_percent": attendance,
        "previous_score": previous_score,
        "sleep_hours": sleep_hours,
        "absences": absences,
        "parental_education": parental_education,
        "internet_access": internet_access,
        "extracurricular_activities": extracurricular,
        "part_time_job": part_time_job,
        "study_method": study_method,
    }])


def predict_all(input_df):
    """Run all three models on a dataframe of one or more students."""
    scores = np.clip(regressor.predict(input_df), 0, 100)
    pass_preds = classifier.predict(input_df)
    pass_proba = classifier.predict_proba(input_df)[:, 1]
    categories = category_clf.predict(input_df)
    return scores, pass_preds, pass_proba, categories


def pretty_feature_name(raw_name):
    name = raw_name.replace("num__", "").replace("cat__", "")
    return name.replace("_", " ").title()


def compute_contributions(input_df):
    """Break down the predicted score into per-feature contributions using
    the saved linear model coefficients, for a single-row input_df."""
    preprocessor = regressor.named_steps.get("preprocessor")
    # Use the same preprocessor structure as saved in coef_data
    # (coef_data was built from its own pipeline's preprocessor, so we
    # rebuild the transform using the coefficient pipeline's feature order)
    transformed = _explain_preprocessor.transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = transformed[0]

    feat_names = coef_data["feature_names"]
    coefs = coef_data["coefficients"]
    contributions = [
        (pretty_feature_name(name), coef * val)
        for name, coef, val in zip(feat_names, coefs, transformed)
        if abs(coef * val) > 1e-6
    ]
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions[:8], coef_data["intercept"]


# Rebuild the exact preprocessor used for explainability (fit on full data,
# matching train_model.py's explain_pipe). We refit it here quickly so the
# app doesn't need to pickle a second preprocessor separately.
@st.cache_resource
def _fit_explain_preprocessor():
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    df = pd.read_csv("data/student_data.csv")
    pre = ColumnTransformer(transformers=[
        ("num", StandardScaler(), metadata["numeric_features"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), metadata["categorical_features"]),
    ])
    pre.fit(df[metadata["numeric_features"] + metadata["categorical_features"]])
    return pre


_explain_preprocessor = _fit_explain_preprocessor()


def plot_contributions(contributions, intercept):
    labels = [c[0] for c in contributions][::-1]
    values = [c[1] for c in contributions][::-1]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Impact on predicted score (points)")
    ax.set_title(f"What's driving this prediction? (baseline: {intercept:.1f})")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("🎓 Student Performance Predictor")
st.caption(
    f"Regressor: **{metadata['best_regressor']}** (R² = {metadata['best_regressor_r2']}) · "
    f"Pass/Fail model: **{metadata['best_classifier']}** (F1 = {metadata['best_classifier_f1']}) · "
    f"Category model accuracy: {metadata['category_classifier_accuracy']}"
)

tab_single, tab_batch = st.tabs(
    ["🧑‍🎓 Single Prediction", "📄 Batch Prediction (CSV)"]
)

# ---------------------------------------------------------------------
# TAB 1 — Single Prediction
# ---------------------------------------------------------------------
with tab_single:
    student_name = st.text_input("Student name (optional)", placeholder="e.g. Aman Sharma")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider("Study hours per week", 0.0, 40.0, 12.0, 0.5)
        attendance = st.slider("Attendance (%)", 0.0, 100.0, 80.0, 1.0)
        previous_score = st.slider("Previous exam score", 0.0, 100.0, 65.0, 1.0)
        sleep_hours = st.slider("Average sleep hours", 3.0, 10.0, 7.0, 0.5)
        absences = st.slider("Number of absences", 0, 40, 5)

    with col2:
        parental_education = st.selectbox(
            "Parental education", metadata["categorical_options"]["parental_education"]
        )
        internet_access = st.selectbox(
            "Internet access", metadata["categorical_options"]["internet_access"]
        )
        extracurricular = st.selectbox(
            "Extracurricular activities", metadata["categorical_options"]["extracurricular_activities"]
        )
        part_time_job = st.selectbox(
            "Part-time job", metadata["categorical_options"]["part_time_job"]
        )
        study_method = st.selectbox(
            "Preferred study method", metadata["categorical_options"]["study_method"]
        )

    st.divider()

    if st.button("Predict Performance", type="primary", use_container_width=True):
        input_df = make_input_row(
            study_hours, attendance, previous_score, sleep_hours, absences,
            parental_education, internet_access, extracurricular,
            part_time_job, study_method,
        )
        scores, pass_preds, pass_proba, categories = predict_all(input_df)
        score, pass_pred, proba, category = scores[0], pass_preds[0], pass_proba[0], categories[0]
        result_label = "Pass" if pass_pred == 1 else "Fail"

        display_name = student_name.strip() if student_name.strip() else "This student"
        st.subheader(f"Results for {display_name}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Final Score", f"{score:.1f} / 100")
        m2.metric("Predicted Result", result_label, f"{proba*100:.1f}% pass probability")
        cat_color = CATEGORY_COLORS.get(category, "#888")
        m3.markdown(
            f"**Performance Category**<br>"
            f"<span style='background-color:{cat_color};color:white;padding:4px 12px;"
            f"border-radius:12px;font-size:1.1em;'>{category}</span>",
            unsafe_allow_html=True,
        )

        if result_label == "Pass":
            st.success(f"{display_name} is predicted to **PASS** with an estimated score of {score:.1f}.")
        else:
            st.error(f"{display_name} is predicted to **FAIL** with an estimated score of {score:.1f}. "
                      "Consider more study hours and improved attendance.")
        st.progress(min(1.0, score / 100))

        # --- Explainability ---
        st.subheader("Why this prediction?")
        contributions, intercept = compute_contributions(input_df)
        fig = plot_contributions(contributions, intercept)
        st.pyplot(fig)
        st.caption(
            "Green bars push the score up, red bars pull it down, relative to a "
            f"baseline of {intercept:.1f} (the model's average starting point)."
        )

        # --- Compare to dataset average ---
        st.subheader("How this student compares to the dataset average")
        stats = metadata["dataset_stats"]
        compare_df = pd.DataFrame({
            "This student": [study_hours, attendance, previous_score, sleep_hours, absences],
            "Dataset average": [
                stats["study_hours_per_week"]["mean"], stats["attendance_percent"]["mean"],
                stats["previous_score"]["mean"], stats["sleep_hours"]["mean"], stats["absences"]["mean"],
            ],
        }, index=["Study Hrs/Week", "Attendance %", "Previous Score", "Sleep Hrs", "Absences"])
        st.bar_chart(compare_df)

        # --- Save to history ---
        st.session_state.history.append({
            "Name": student_name.strip() if student_name.strip() else "(unnamed)",
            "Study Hrs": study_hours, "Attendance %": attendance,
            "Previous Score": previous_score, "Predicted Score": round(score, 1),
            "Result": result_label, "Category": category,
        })

    if st.session_state.history:
        st.divider()
        st.subheader("Prediction history (this session)")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
        st.line_chart(hist_df["Predicted Score"])
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# ---------------------------------------------------------------------
# TAB 2 — Batch Prediction
# ---------------------------------------------------------------------
with tab_batch:
    st.write(
        "Upload a CSV with one row per student. Required columns:"
    )
    required_cols = metadata["numeric_features"] + metadata["categorical_features"]
    st.code(", ".join(required_cols))
    st.caption("You can also add an optional `student_name` column — it'll be carried through to the results.")

    template_df = pd.DataFrame([{
        "student_name": "Aman Sharma",
        "study_hours_per_week": 12, "attendance_percent": 80, "previous_score": 65,
        "sleep_hours": 7, "absences": 5, "parental_education": "Bachelors",
        "internet_access": "Yes", "extracurricular_activities": "Yes",
        "part_time_job": "No", "study_method": "Self-study",
    }])
    st.download_button(
        "Download a template CSV",
        template_df.to_csv(index=False),
        file_name="student_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload student data (CSV)", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            missing = [c for c in required_cols if c not in batch_df.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                scores, pass_preds, pass_proba, categories = predict_all(batch_df[required_cols])
                results_df = batch_df.copy()
                results_df["predicted_score"] = np.round(scores, 1)
                results_df["predicted_result"] = np.where(pass_preds == 1, "Pass", "Fail")
                results_df["pass_probability"] = np.round(pass_proba * 100, 1)
                results_df["performance_category"] = categories

                st.success(f"Predicted results for {len(results_df)} students.")
                st.dataframe(results_df, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Average Predicted Score", f"{results_df['predicted_score'].mean():.1f}")
                c2.metric("Predicted Pass Rate", f"{(results_df['predicted_result']=='Pass').mean()*100:.1f}%")
                c3.metric("Students Uploaded", len(results_df))

                st.bar_chart(results_df["performance_category"].value_counts())

                st.download_button(
                    "Download results as CSV",
                    results_df.to_csv(index=False),
                    file_name="predicted_results.csv",
                    mime="text/csv",
                    type="primary",
                )
        except Exception as e:
            st.error(f"Couldn't process this file: {e}")

st.divider()
st.caption("Built with scikit-learn + Streamlit · Student Performance Predictor project")
