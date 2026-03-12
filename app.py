"""
Employee Performance Evaluation System — Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Performance Evaluation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }

.main { background: #0f1117; }
.block-container { padding-top: 2rem; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e2130 0%, #252a3d 100%);
    border: 1px solid #2d3250;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label { font-size: 0.8rem; color: #8892b0; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }

/* Rating badge */
.rating-outstanding    { background: #065f46; color: #6ee7b7; border-radius: 8px; padding: 6px 16px; font-weight: 600; }
.rating-exceeds        { background: #1e3a5f; color: #93c5fd; border-radius: 8px; padding: 6px 16px; font-weight: 600; }
.rating-meets          { background: #3b2f00; color: #fcd34d; border-radius: 8px; padding: 6px 16px; font-weight: 600; }
.rating-below          { background: #4a1d00; color: #fdba74; border-radius: 8px; padding: 6px 16px; font-weight: 600; }
.rating-needs          { background: #4c0519; color: #fca5a5; border-radius: 8px; padding: 6px 16px; font-weight: 600; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #141724;
    border-right: 1px solid #2d3250;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label { color: #cdd6f4 !important; font-size: 0.85rem; }

/* Prediction result box */
.prediction-box {
    background: linear-gradient(135deg, #1a1f35 0%, #1e2340 100%);
    border: 1px solid #4f46e5;
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 0 30px rgba(99, 102, 241, 0.15);
}

/* Score gauge ring */
.score-display {
    font-size: 3.5rem;
    font-weight: 800;
    font-family: 'Space Grotesk', sans-serif;
    background: linear-gradient(135deg, #6366f1, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Tips box */
.tip-box {
    background: #1a2035;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #cdd6f4;
}

/* Section header */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #2d3250;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Models ──────────────────────────────────────────────────────────────
BASE  = os.path.dirname(__file__)
MDIR  = os.path.join(BASE, "models")
DATA  = os.path.join(BASE, "data", "employee_performance.csv")

@st.cache_resource
def load_models():
    reg = joblib.load(os.path.join(MDIR, "regression_model.pkl"))
    clf = joblib.load(os.path.join(MDIR, "classification_model.pkl"))
    with open(os.path.join(MDIR, "meta.json")) as f:
        meta = json.load(f)
    return reg, clf, meta

@st.cache_data
def load_data():
    return pd.read_csv(DATA)

reg_model, clf_model, meta = load_models()
df = load_data()

NUM_FEAT = meta["numeric_features"]
CAT_FEAT = meta["categorical_features"]

# ─── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 EvalPro")
    st.markdown("<p style='color:#8892b0;font-size:0.8rem;margin-top:-10px;'>Employee Performance System</p>", unsafe_allow_html=True)
    st.divider()
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔮 Predict Performance", "📁 Batch Evaluation", "📈 Analytics", "ℹ️ About & Guide"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown(f"""
    <div style='font-size:0.75rem;color:#555;'>
    <b style='color:#8892b0;'>Model Metrics</b><br><br>
    Regression R² &nbsp;&nbsp;{meta['metrics']['regression']['R2']}<br>
    Reg MAE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{meta['metrics']['regression']['MAE']}<br>
    Classification Acc &nbsp;{meta['metrics']['classification']['Accuracy']}
    </div>
    """, unsafe_allow_html=True)

# ─── Helper Functions ─────────────────────────────────────────────────────────
RATING_COLORS = {
    "Outstanding":         "#6ee7b7",
    "Exceeds Expectations":"#93c5fd",
    "Meets Expectations":  "#fcd34d",
    "Below Expectations":  "#fdba74",
    "Needs Improvement":   "#fca5a5",
}
RATING_CSS = {
    "Outstanding":          "rating-outstanding",
    "Exceeds Expectations": "rating-exceeds",
    "Meets Expectations":   "rating-meets",
    "Below Expectations":   "rating-below",
    "Needs Improvement":    "rating-needs",
}

def predict(inp: dict):
    row = pd.DataFrame([inp])
    score  = float(reg_model.predict(row)[0])
    rating = clf_model.predict(row)[0]
    proba  = clf_model.predict_proba(row)[0]
    classes = clf_model.classes_
    return round(score, 1), rating, dict(zip(classes, proba))

def generate_tips(inp: dict, score: float, rating: str) -> list[str]:
    tips = []
    if inp["Absences"] > 8:
        tips.append("📅 High absences detected. Improving attendance can significantly boost the score.")
    if inp["GoalAchievementPct"] < 65:
        tips.append("🎯 Goal achievement is low. Work on setting SMART goals and tracking them weekly.")
    if inp["TaskCompletionRate"] < 70:
        tips.append("✅ Task completion rate needs improvement. Consider using project management tools.")
    if inp["TrainingHoursLastYear"] < 20:
        tips.append("📚 Low training hours. Encourage upskilling via online courses or workshops.")
    if inp["ManagerRatingScore"] < 3.0:
        tips.append("👔 Manager rating is below average. Regular 1-on-1s and feedback sessions can help.")
    if inp["WorkLifeBalance"] < 2:
        tips.append("⚖️ Work-life balance score is low. This often leads to burnout and attrition risk.")
    if inp["JobSatisfaction"] < 2:
        tips.append("😊 Low job satisfaction. Consider role enrichment or career path discussions.")
    if not tips:
        tips.append("🌟 All key metrics look healthy! Keep maintaining high standards.")
    return tips

# ─── PAGE: Dashboard ──────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.markdown("# Employee Performance Evaluation System")
    st.markdown("<p style='color:#8892b0;margin-top:-10px;'>ML-powered insights for HR teams</p>", unsafe_allow_html=True)
    st.divider()

    # KPI Row
    total = len(df)
    avg_score = df["PerformanceScore"].mean()
    top_perf   = (df["PerformanceRating"].isin(["Outstanding","Exceeds Expectations"])).sum()
    needs_att  = (df["PerformanceRating"].isin(["Needs Improvement","Below Expectations"])).sum()

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, total,                        "Total Employees"),
        (c2, f"{avg_score:.1f}",           "Avg Performance Score"),
        (c3, f"{top_perf} ({top_perf/total*100:.0f}%)", "Top Performers"),
        (c4, f"{needs_att} ({needs_att/total*100:.0f}%)", "Need Attention"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Rating Distribution</div>', unsafe_allow_html=True)
        rating_counts = df["PerformanceRating"].value_counts().reset_index()
        rating_counts.columns = ["Rating", "Count"]
        colors = [RATING_COLORS.get(r, "#888") for r in rating_counts["Rating"]]
        fig = px.bar(rating_counts, x="Rating", y="Count", color="Rating",
                     color_discrete_map=RATING_COLORS,
                     template="plotly_dark")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, height=300, margin=dict(t=10, b=10)
        )
        fig.update_xaxes(tickfont=dict(size=11))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Dept-wise Avg Score</div>', unsafe_allow_html=True)
        dept_avg = df.groupby("Department")["PerformanceScore"].mean().sort_values().reset_index()
        fig2 = px.bar(dept_avg, x="PerformanceScore", y="Department", orientation="h",
                      color="PerformanceScore", color_continuous_scale="Purples",
                      template="plotly_dark")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, height=300, margin=dict(t=10, b=10),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Charts row 2
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Score Distribution</div>', unsafe_allow_html=True)
        fig3 = px.histogram(df, x="PerformanceScore", nbins=40,
                            color_discrete_sequence=["#6366f1"],
                            template="plotly_dark")
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(t=10, b=10), showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Goal Achievement vs Score</div>', unsafe_allow_html=True)
        sample = df.sample(min(500, len(df)), random_state=1)
        fig4 = px.scatter(sample, x="GoalAchievementPct", y="PerformanceScore",
                          color="PerformanceRating", color_discrete_map=RATING_COLORS,
                          opacity=0.6, template="plotly_dark",
                          labels={"GoalAchievementPct": "Goal Achievement %"})
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(t=10, b=10),
            legend=dict(font_size=10, x=0, y=1)
        )
        st.plotly_chart(fig4, use_container_width=True)

# ─── PAGE: Predict ────────────────────────────────────────────────────────────
elif page == "🔮 Predict Performance":
    st.markdown("# 🔮 Predict Employee Performance")
    st.markdown("<p style='color:#8892b0;margin-top:-10px;'>Fill in employee details to get an AI-powered evaluation</p>", unsafe_allow_html=True)
    st.divider()

    with st.form("predict_form"):
        # Section 1: Personal
        st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        with p1:
            age    = st.slider("Age", 18, 65, 32)
            gender = st.selectbox("Gender", ["Male","Female","Non-binary"])
        with p2:
            education = st.selectbox("Education", ["High School","Bachelor's","Master's","PhD"])
            department = st.selectbox("Department", ["Engineering","Sales","Marketing","HR","Finance","Operations","Support"])
        with p3:
            job_level        = st.slider("Job Level (1-5)", 1, 5, 2)
            monthly_income   = st.number_input("Monthly Income (₹)", 10000, 300000, 60000, step=5000)

        st.divider()

        # Section 2: Work Experience
        st.markdown('<div class="section-header">🏢 Work Experience</div>', unsafe_allow_html=True)
        w1, w2, w3 = st.columns(3)
        with w1:
            years_company    = st.slider("Years at Company", 0, 30, 3)
            years_role       = st.slider("Years in Current Role", 0, 15, 2)
        with w2:
            years_promo      = st.slider("Years Since Last Promotion", 0, 15, 1)
            num_companies    = st.slider("No. of Companies Worked", 0, 10, 2)
        with w3:
            overtime         = st.selectbox("Works Overtime?", ["No","Yes"])
            business_travel  = st.selectbox("Business Travel", ["No Travel","Rarely","Frequently"])

        st.divider()

        # Section 3: Performance Metrics
        st.markdown('<div class="section-header">📊 Performance Metrics</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            task_completion  = st.slider("Task Completion Rate (%)", 0.0, 100.0, 75.0, step=0.5)
            goal_achievement = st.slider("Goal Achievement (%)", 0.0, 100.0, 70.0, step=0.5)
            projects         = st.slider("Projects Completed", 0, 25, 5)
        with m2:
            manager_rating   = st.slider("Manager Rating (1–5)", 1.0, 5.0, 3.5, step=0.1)
            peer_rating      = st.slider("Peer Rating (1–5)", 1.0, 5.0, 3.5, step=0.1)
            training_hours   = st.slider("Training Hours (Last Year)", 0, 100, 30)
        with m3:
            absences         = st.slider("Absences (Days)", 0, 30, 3)
            work_life        = st.slider("Work-Life Balance (1–4)", 1, 4, 3)
            job_sat          = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
            rel_sat          = st.slider("Relationship Satisfaction (1–4)", 1, 4, 3)
            env_sat          = st.slider("Environment Satisfaction (1–4)", 1, 4, 3)

        submitted = st.form_submit_button("🔍 Evaluate Performance", use_container_width=True)

    if submitted:
        inp = {
            "Age": age, "Gender": gender, "Education": education,
            "Department": department, "JobLevel": job_level,
            "MonthlyIncome": monthly_income,
            "YearsAtCompany": years_company, "YearsInCurrentRole": years_role,
            "YearsSinceLastPromotion": years_promo, "NumCompaniesWorked": num_companies,
            "OverTime": overtime, "BusinessTravel": business_travel,
            "TaskCompletionRate": task_completion, "GoalAchievementPct": goal_achievement,
            "ProjectsCompleted": projects, "ManagerRatingScore": manager_rating,
            "PeerRatingScore": peer_rating, "TrainingHoursLastYear": training_hours,
            "Absences": absences, "WorkLifeBalance": work_life,
            "JobSatisfaction": job_sat, "RelationshipSatisfaction": rel_sat,
            "EnvironmentSatisfaction": env_sat,
        }
        score, rating, proba = predict(inp)
        tips = generate_tips(inp, score, rating)
        css_cls = RATING_CSS.get(rating, "rating-needs")
        color = RATING_COLORS.get(rating, "#fff")

        st.markdown("---")
        r1, r2 = st.columns([1, 1])

        with r1:
            st.markdown(f"""
            <div class="prediction-box">
                <p style='color:#8892b0;font-size:0.9rem;margin-bottom:4px;'>PERFORMANCE SCORE</p>
                <div class="score-display">{score}<span style='font-size:1.5rem;'>/100</span></div>
                <br>
                <p style='color:#8892b0;font-size:0.9rem;margin-bottom:6px;'>RATING</p>
                <span class="{css_cls}">{rating}</span>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown('<div class="section-header">Rating Probabilities</div>', unsafe_allow_html=True)
            prob_df = pd.DataFrame({
                "Rating": list(proba.keys()),
                "Probability": [v*100 for v in proba.values()]
            }).sort_values("Probability", ascending=True)
            fig_p = px.bar(prob_df, x="Probability", y="Rating", orientation="h",
                           color="Rating", color_discrete_map=RATING_COLORS,
                           template="plotly_dark", text_auto=".1f")
            fig_p.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=250, margin=dict(t=10, b=10), showlegend=False
            )
            fig_p.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
            st.plotly_chart(fig_p, use_container_width=True)

        st.markdown('<div class="section-header">💡 Improvement Recommendations</div>', unsafe_allow_html=True)
        for tip in tips:
            st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

# ─── PAGE: Batch Evaluation ───────────────────────────────────────────────────
elif page == "📁 Batch Evaluation":
    st.markdown("# 📁 Batch Employee Evaluation")
    st.markdown("<p style='color:#8892b0;margin-top:-10px;'>Upload a CSV and get predictions for all employees at once</p>", unsafe_allow_html=True)
    st.divider()

    st.info("📋 Upload a CSV with the required columns. Download the sample template to get started.")
    sample_cols = NUM_FEAT + CAT_FEAT
    sample_df = df[sample_cols].head(5)
    st.download_button(
        "⬇️ Download Sample Template",
        sample_df.to_csv(index=False),
        "sample_template.csv",
        "text/csv"
    )

    uploaded = st.file_uploader("Upload your employee CSV", type=["csv"])
    if uploaded:
        try:
            udf = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(udf)} employees")
            missing = [c for c in sample_cols if c not in udf.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                X_batch = udf[sample_cols]
                scores  = reg_model.predict(X_batch)
                ratings = clf_model.predict(X_batch)
                udf["Predicted_Score"]  = np.round(scores, 1)
                udf["Predicted_Rating"] = ratings

                st.markdown("### Results Preview")
                st.dataframe(
                    udf[["Predicted_Score","Predicted_Rating"] + sample_cols[:6]].head(20),
                    use_container_width=True
                )

                dist = udf["Predicted_Rating"].value_counts().reset_index()
                dist.columns = ["Rating","Count"]
                fig = px.pie(dist, values="Count", names="Rating",
                             color="Rating", color_discrete_map=RATING_COLORS,
                             template="plotly_dark", hole=0.4)
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350)
                st.plotly_chart(fig, use_container_width=True)

                st.download_button(
                    "⬇️ Download Results CSV",
                    udf.to_csv(index=False),
                    "batch_results.csv",
                    "text/csv"
                )
        except Exception as e:
            st.error(f"Error: {e}")

# ─── PAGE: Analytics ──────────────────────────────────────────────────────────
elif page == "📈 Analytics":
    st.markdown("# 📈 Analytics & Insights")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["🔑 Feature Importance", "🔥 Correlations", "🏢 Department Deep Dive"])

    with tab1:
        st.markdown("### Top Features Driving Performance")
        fi_data = pd.DataFrame(
            list(meta["top_features"].items()),
            columns=["Feature","Importance"]
        ).sort_values("Importance")
        fi_data["Importance"] = fi_data["Importance"] * 100
        fig = px.bar(fi_data, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Purples",
                     template="plotly_dark", text_auto=".1f")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=380, margin=dict(t=10, b=10), coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("📌 GoalAchievementPct, TaskCompletionRate, and ManagerRatingScore are the strongest performance drivers.")

    with tab2:
        st.markdown("### Feature Correlation Heatmap")
        corr_cols = ["PerformanceScore","TaskCompletionRate","GoalAchievementPct",
                     "ManagerRatingScore","PeerRatingScore","Absences",
                     "TrainingHoursLastYear","JobSatisfaction","WorkLifeBalance"]
        corr = df[corr_cols].corr()
        fig2 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         template="plotly_dark", zmin=-1, zmax=1)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=450, margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("### Department Analysis")
        dept_stats = df.groupby("Department").agg(
            Avg_Score=("PerformanceScore","mean"),
            Employees=("PerformanceScore","count"),
            Avg_Absences=("Absences","mean"),
            Avg_Training=("TrainingHoursLastYear","mean"),
        ).reset_index()

        fig3 = px.scatter(dept_stats, x="Avg_Training", y="Avg_Score",
                          size="Employees", color="Department",
                          text="Department", template="plotly_dark",
                          labels={"Avg_Training":"Avg Training Hours","Avg_Score":"Avg Performance Score"})
        fig3.update_traces(textposition="top center")
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400)
        st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(dept_stats.round(2), use_container_width=True)

# ─── PAGE: About ──────────────────────────────────────────────────────────────
elif page == "ℹ️ About & Guide":
    st.markdown("# ℹ️ About This System")
    st.divider()

    st.markdown("""
    ## 🎯 What is this?
    An end-to-end **Employee Performance Evaluation System** built with Python and ML that helps HR teams:
    - Predict employee performance scores and ratings automatically
    - Identify at-risk employees before problems escalate
    - Get AI-driven improvement recommendations
    - Run batch evaluations across entire teams

    ## 🔧 Tech Stack
    | Layer | Technology |
    |-------|-----------|
    | Dataset | Custom synthetic (2000 employees, 27 features) |
    | Regression Model | Gradient Boosting Regressor |
    | Classification Model | Random Forest Classifier |
    | Web App | Streamlit |
    | Visualization | Plotly |

    ## 📊 Dataset Features Used

    **Performance Metrics** (highest importance):
    - Goal Achievement %, Task Completion Rate, Projects Completed
    - Manager Rating, Peer Rating

    **Behavioral Factors**:
    - Absences, Overtime, Training Hours, Business Travel

    **Satisfaction Scores**:
    - Job Satisfaction, Work-Life Balance, Environment Satisfaction, Relationship Satisfaction

    **Employee Profile**:
    - Age, Education, Department, Job Level, Years at Company, Monthly Income

    ## 🧠 Models

    **Regression** → Predicts a numeric score (0–100)
    - Algorithm: GradientBoostingRegressor
    - R² Score: 0.79 | MAE: ~4.5 points

    **Classification** → Predicts rating label
    - Algorithm: RandomForestClassifier (class_weight=balanced)
    - Accuracy: ~72%
    - 5 classes: Outstanding → Exceeds → Meets → Below → Needs Improvement

    ## 🚀 How to Use
    1. **Dashboard** — Get a high-level overview of all employees
    2. **Predict** — Evaluate a single employee by entering their details
    3. **Batch** — Upload CSV of many employees and get bulk predictions
    4. **Analytics** — Explore insights, feature importance, and correlations
    """)
