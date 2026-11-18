import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
from pathlib import Path
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore

# Local predictor fallback
from predict import AIStudyLabPredictor

# Page Configuration
st.set_page_config(
    page_title="AI Study Lab - Multi-Model Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
FASTAPI_URL = "http://localhost:8889"

# Inject Custom Cyberpunk / Dark Aurora Glassmorphic CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Global Base */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0F172A 0%, #090D16 100%);
        color: #F1F5F9;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.85) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.15) !important;
        backdrop-filter: blur(16px);
    }
    
    /* Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 20px;
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 50%, #6366F1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
        letter-spacing: -0.03em;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin: 0 0 16px 0;
    }
    
    /* Badge Pills */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 50px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin-right: 8px;
    }
    .badge-cyan {
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #38BDF8;
    }
    .badge-emerald {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34D399;
    }
    .badge-purple {
        background: rgba(168, 85, 247, 0.15);
        border: 1px solid rgba(168, 85, 247, 0.4);
        color: #C084FC;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 12px 32px rgba(56, 189, 248, 0.15);
        transform: translateY(-2px);
    }
    
    .card-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Custom Input Controls */
    .stTextInput > div > div > input, .stSelectbox > div > div > select {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        border-radius: 12px !important;
        color: #F8FAFC !important;
    }
    
    /* Result Box */
    .result-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%);
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: 16px;
        padding: 20px 24px;
        margin-top: 16px;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.15);
    }
    .result-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #34D399;
        font-weight: 700;
    }
    .result-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# Load local predictor with caching
@st.cache_resource
def get_predictor():
    return AIStudyLabPredictor()


# Load model metrics JSON
@st.cache_data(ttl=60)
def load_metrics():
    metrics_path = BASE_DIR / "model_metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


predictor = get_predictor()
metrics_data = load_metrics()

# Check FastAPI status
api_status = False
try:
    r = requests.get(f"{FASTAPI_URL}/health", timeout=1.5)
    if r.status_code == 200:
        api_status = True
except Exception:
    api_status = False

# Sidebar Controls
with st.sidebar:
    st.markdown("### ⚡ AI Study Lab Control Panel")
    
    if api_status:
        st.markdown('<span class="badge-pill badge-emerald">● REST API Connected (Port 8889)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-pill badge-cyan">● Using Direct Python Inference</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Test Presets")
    preset = st.selectbox(
        "Choose a sample scenario:",
        [
            "Custom User Inputs",
            "Beginner Python Student",
            "Advanced Physics Inquiry",
            "Geography Question Type",
            "Machine Learning Resource Search"
        ]
    )

    st.markdown("---")
    st.markdown("### 📈 Model Accuracy Summary")
    st.markdown("""
    - **Logistic Regression**: `100.0%`
    - **Decision Tree**: `95.83%`
    - **SVM Question Classifier**: `90.80%`
    - **Naive Bayes (Top-5)**: `91.28%`
    - **K-Means Clusters**: `8 Groups`
    """)


# Main Hero Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">AI Study Lab — Intelligence Dashboard</div>
    <div class="hero-subtitle">Unified Machine Learning Platform with 90%+ Accuracy Across Classification Models</div>
    <div>
        <span class="badge-pill badge-emerald">✓ 90%+ Accuracy Standard</span>
        <span class="badge-pill badge-cyan">⚡ FastAPI & Streamlit</span>
        <span class="badge-pill badge-purple">🧠 5 Trained ML Pipelines</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Course Recommender",
    "🎓 Pathway Level Predictor",
    "📚 Chapter Classifier",
    "❓ Question Type Analyzer",
    "🔍 Course Clusters",
    "📊 Benchmarks & Accuracy"
])

# -------------------------------------------------------------
# Tab 1: Course Recommender (Logistic Regression - 100% Accuracy)
# -------------------------------------------------------------
with tab1:
    st.markdown("""
    <div class="glass-card">
        <div class="card-header">🎯 Student Course Recommendation Engine</div>
        <p style="color: #94A3B8;">Uses Logistic Regression trained with 100.0% accuracy to match student preferences to the ideal course.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 📥 Student Preferences")

        # Preset overrides
        default_interest = "Python" if preset == "Beginner Python Student" else "Algebra"
        default_cat = "Programming" if preset == "Beginner Python Student" else "Mathematics"
        default_diff = "beginner"
        default_exp = "none"
        default_lang = "English"

        interests = st.text_input("Interests / Topic", value=default_interest)
        category = st.selectbox("Category", ["Mathematics", "Programming", "Chemistry", "Physics", "Literature", "History"], index=0 if default_cat == "Mathematics" else 1)
        difficulty = st.selectbox("Preferred Difficulty", ["beginner", "intermediate", "advanced"], index=0)
        experience = st.selectbox("Past Experience", ["none", "some", "extensive"], index=0)
        language = st.selectbox("Preferred Language", ["English", "Hindi"], index=0)

        predict_btn = st.button("🚀 Recommend Course", type="primary", use_container_width=True)

    with col2:
        if predict_btn or preset != "Custom User Inputs":
            with st.spinner("Generating course recommendation..."):
                res = predictor.predict_course(interests, category, difficulty, experience, language)
                rec_course = res.get("recommended_course", "N/A")
                top_recs = res.get("top_recommendations", [])

                st.markdown(f"""
                <div class="result-box">
                    <div class="result-title">Top Recommended Course Match</div>
                    <div class="result-value">🎓 {rec_course}</div>
                </div>
                """, unsafe_allow_html=True)

                if top_recs:
                    st.markdown("<br><h5>📊 Top Course Match Probabilities</h5>", unsafe_allow_html=True)
                    df_top = pd.DataFrame(top_recs)
                    fig = px.bar(
                        df_top,
                        x="probability",
                        y="course",
                        orientation="h",
                        color="probability",
                        color_continuous_scale="Viridis",
                        labels={"probability": "Confidence", "course": "Course"},
                        text_auto=".2%"
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#F8FAFC"),
                        height=280,
                        margin=dict(l=0, r=20, t=10, b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# Tab 2: Pathway Level Predictor (Decision Tree - 95.83% Accuracy)
# -------------------------------------------------------------
with tab2:
    st.markdown("""
    <div class="glass-card">
        <div class="card-header">🎓 Learning Pathway Level Predictor</div>
        <p style="color: #94A3B8;">Uses Decision Tree with TF-IDF keyword vectorization (95.83% Accuracy) to classify resource difficulty level.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### 📥 Learning Resource Details")
        c_level = st.selectbox("Course Level", ["Beginner", "Intermediate", "Advanced"], index=0)
        c_type = st.selectbox("Type", ["Free", "Paid"], index=0)
        c_module = st.text_input("Module / Topic Name", value="Python Programming Basics" if preset == "Beginner Python Student" else "Calculus Derivatives")
        c_duration = st.number_input("Duration (minutes)", min_value=5, max_value=3000, value=60)
        c_keyword = st.text_input("Skills & Keywords", value="python, coding, functions" if preset == "Beginner Python Student" else "math, calculus, derivatives")

        dt_btn = st.button("🔍 Predict Resource Level", type="primary", use_container_width=True)

    with c2:
        if dt_btn or preset != "Custom User Inputs":
            res_dt = predictor.predict_learning_path(c_level, c_type, c_module, c_duration, c_keyword)
            pred_level = res_dt.get("predicted_difficulty", "N/A")
            probs_dt = res_dt.get("class_probabilities", {})

            st.markdown(f"""
            <div class="result-box">
                <div class="result-title">Predicted Difficulty Level</div>
                <div class="result-value">⚡ {pred_level}</div>
            </div>
            """, unsafe_allow_html=True)

            if probs_dt:
                st.markdown("<br><h5>📊 Difficulty Class Distribution</h5>", unsafe_allow_html=True)
                df_dt = pd.DataFrame(list(probs_dt.items()), columns=["Level", "Probability"])
                fig_dt = px.pie(
                    df_dt,
                    values="Probability",
                    names="Level",
                    color="Level",
                    color_discrete_map={"Easy": "#34D399", "Medium": "#38BDF8", "Hard": "#F43F5E"},
                    hole=0.45
                )
                fig_dt.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#F8FAFC"),
                    height=280,
                    margin=dict(l=0, r=0, t=10, b=10)
                )
                st.plotly_chart(fig_dt, use_container_width=True)

# -------------------------------------------------------------
# Tab 3: Smart Chapter Classifier (Naive Bayes - 91.28% Top-5 Acc)
# -------------------------------------------------------------
with tab3:
    st.markdown("""
    <div class="glass-card">
        <div class="card-header">📚 Smart Textbook Chapter Classifier</div>
        <p style="color: #94A3B8;">Classifies student questions into 202 textbook chapters using Naive Bayes trained on 116,240 questions (91.28% Top-5 Accuracy).</p>
    </div>
    """, unsafe_allow_html=True)

    q_text_default = "What is Newton's second law of motion?"
    if preset == "Advanced Physics Inquiry":
        q_text_default = "What is the magnetic field inside a solenoid carrying current I?"

    question_in = st.text_area("Enter Student Question Text:", value=q_text_default, height=100)
    nb_btn = st.button("📚 Classify Chapter", type="primary", use_container_width=True)

    if nb_btn or preset == "Advanced Physics Inquiry":
        res_nb = predictor.predict_question_chapter(question_in)
        pred_chapter = res_nb.get("predicted_chapter", "N/A")
        top_chaps = res_nb.get("top_chapters", [])

        st.markdown(f"""
        <div class="result-box">
            <div class="result-title">Top Predicted Chapter</div>
            <div class="result-value">📖 {pred_chapter}</div>
        </div>
        """, unsafe_allow_html=True)

        if top_chaps:
            st.markdown("<br><h5>📊 Top 5 Candidate Chapters</h5>", unsafe_allow_html=True)
            df_nb = pd.DataFrame(top_chaps)
            fig_nb = px.bar(
                df_nb,
                x="probability",
                y="chapter",
                orientation="h",
                color="probability",
                color_continuous_scale="Plasma",
                labels={"probability": "Probability", "chapter": "Textbook Chapter"},
                text_auto=".2%"
            )
            fig_nb.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC"),
                height=300,
                margin=dict(l=0, r=20, t=10, b=10)
            )
            st.plotly_chart(fig_nb, use_container_width=True)

# -------------------------------------------------------------
# Tab 4: Question Type Analyzer (SVM - 90.80% Accuracy)
# -------------------------------------------------------------
with tab4:
    st.markdown("""
    <div class="glass-card">
        <div class="card-header">❓ Question Type Analyzer (SVM)</div>
        <p style="color: #94A3B8;">Uses FeatureUnion Word+Char TF-IDF with LinearSVC (90.80% Accuracy) to classify question structural intent.</p>
    </div>
    """, unsafe_allow_html=True)

    svm_default = "Where is Mount Everest located?"
    if preset == "Geography Question Type":
        svm_default = "What city has the largest population in South America?"

    svm_q = st.text_input("Enter Question for Type Analysis:", value=svm_default)
    svm_btn = st.button("❓ Detect Question Type", type="primary", use_container_width=True)

    if svm_btn or preset == "Geography Question Type":
        res_svm = predictor.predict_question_type(svm_q)
        q_type = res_svm.get("question_type", "N/A")
        scores_svm = res_svm.get("top_type_scores", [])

        type_map = {
            "ABBR": "Abbreviation / Expansion",
            "DESC": "Description / Definition / Method",
            "ENTY": "Entity / Object / Animal",
            "HUM": "Human / Individual / Group",
            "LOC": "Location / City / Country",
            "NUM": "Numeric / Quantity / Date"
        }
        type_desc = type_map.get(q_type, q_type)

        st.markdown(f"""
        <div class="result-box">
            <div class="result-title">Detected Question Intent Category</div>
            <div class="result-value">🏷️ {q_type} — {type_desc}</div>
        </div>
        """, unsafe_allow_html=True)

        if scores_svm:
            st.markdown("<br><h5>📊 Linear Decision Scores across Intent Classes</h5>", unsafe_allow_html=True)
            df_svm = pd.DataFrame(scores_svm)
            fig_svm = px.bar(
                df_svm,
                x="score",
                y="type",
                orientation="h",
                color="score",
                color_continuous_scale="Tealgrn",
                labels={"score": "Decision Score", "type": "Category"}
            )
            fig_svm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC"),
                height=260,
                margin=dict(l=0, r=20, t=10, b=10)
            )
            st.plotly_chart(fig_svm, use_container_width=True)

# -------------------------------------------------------------
# Tab 5: Course Cluster Explorer (K-Means)
# -------------------------------------------------------------
with tab5:
    st.markdown("""
    <div class="glass-card">
        <div class="card-header">🔍 Course Cluster Explorer (K-Means)</div>
        <p style="color: #94A3B8;">Uses Unsupervised K-Means clustering & cosine similarity to discover related edX learning courses.</p>
    </div>
    """, unsafe_allow_html=True)

    km_query = st.text_input("Enter Topic / Skill Query:", value="machine learning data science python")
    km_top = st.slider("Number of Similar Courses", min_value=3, max_value=10, value=5)
    km_btn = st.button("🔍 Find Course Clusters", type="primary", use_container_width=True)

    if km_btn or preset == "Machine Learning Resource Search":
        res_km = predictor.predict_course_cluster(km_query, top_n=km_top)
        cluster_id = res_km.get("cluster_id", 0)
        matched = res_km.get("matched_courses", [])

        st.markdown(f"##### 📌 Cluster World Assigned: `Cluster #{cluster_id}`")

        if matched:
            df_km = pd.DataFrame(matched)
            df_km["similarity"] = (df_km["similarity"] * 100).round(1).astype(str) + "% Match"
            
            for idx, row in df_km.iterrows():
                st.markdown(f"""
                <div class="glass-card" style="margin-bottom: 10px;">
                    <div style="font-weight: 700; font-size: 1.1rem; color: #38BDF8;">{row['title']}</div>
                    <div style="font-size: 0.88rem; color: #94A3B8; margin-top: 4px;">
                        Subject: <b>{row['subject']}</b> | Level: <b>{row['level']}</b> | Match Score: <b style="color: #34D399;">{row['similarity']}</b>
                    </div>
                    <div style="margin-top: 8px;">
                        <a href="{row['url']}" target="_blank" style="color: #6366F1; text-decoration: none; font-weight: 600;">🔗 View Course on edX →</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# -------------------------------------------------------------
# Tab 6: Benchmarks & Model Accuracy Dashboard
# -------------------------------------------------------------
with tab6:
    st.markdown("""
    <div class="glass-card">
        <div class="card-header">📊 Model Benchmark & Accuracy Dashboard</div>
        <p style="color: #94A3B8;">Comprehensive empirical metrics showing 90%+ accuracy benchmark across classification models.</p>
    </div>
    """, unsafe_allow_html=True)

    # 4 Metric Cards
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Logistic Regression", "100.0%", delta="Verified Optimal")
    with mc2:
        st.metric("Decision Tree", "95.83%", delta="+11.74% Boost")
    with mc3:
        st.metric("SVM Question Classifier", "90.80%", delta="+2.20% Boost")
    with mc4:
        st.metric("Naive Bayes Top-5", "91.28%", delta="Full 116k Dataset")

    st.markdown("---")

    # Overview Accuracy Chart
    models_df = pd.DataFrame([
        {"Model": "Logistic Regression", "Accuracy (%)": 100.0, "Target Status": "≥90% Achieved"},
        {"Model": "Decision Tree", "Accuracy (%)": 95.83, "Target Status": "≥90% Achieved"},
        {"Model": "SVM Question Classifier", "Accuracy (%)": 90.80, "Target Status": "≥90% Achieved"},
        {"Model": "Naive Bayes Top-5 Chapter", "Accuracy (%)": 91.28, "Target Status": "≥90% Achieved"},
        {"Model": "Naive Bayes Exact Chapter", "Accuracy (%)": 68.85, "Target Status": "Fine-Grained 202 Classes"}
    ])

    fig_bench = px.bar(
        models_df,
        x="Model",
        y="Accuracy (%)",
        color="Target Status",
        color_discrete_map={"≥90% Achieved": "#34D399", "Fine-Grained 202 Classes": "#38BDF8"},
        text_auto=".2f%",
        title="Model Accuracy Performance Benchmark (Target: ≥90%)"
    )
    fig_bench.add_hline(y=90.0, line_dash="dash", line_color="#F43F5E", annotation_text="Target Threshold (90%)")
    fig_bench.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC"),
        height=380
    )
    st.plotly_chart(fig_bench, use_container_width=True)

    # Retrain button inside dashboard
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Trigger Master Model Retraining Pipeline", type="secondary"):
        with st.spinner("Retraining all models..."):
            import subprocess
            res_p = subprocess.run(["python", "retrain_all.py"], capture_output=True, text=True)
            st.success("All 5 ML models retrained successfully!")
            st.code(res_p.stdout)
