import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Set page config
st.set_page_config(page_title="IT Skills: Analytics & Graphs", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    [
        "About & Methodology",
        "Project Pipeline",
        "EDA & Statistics",
        "Demand Analytics",
        "Kazakhstan Map",
        "Skill Ecosystems (Graph)",
        "ML Model Evaluation",
        "Forecast",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About the Project")
st.sidebar.info(
    "Diploma Thesis\n\n"
    "Forecasting the demand for IT professions and skills using machine learning methods."
)

st.title("Forecasting IT Skills Demand")
st.markdown("Interactive dashboard for analyzing the demand for IT competencies in the labor market and identifying skill ecosystems.")

# Load data
@st.cache_data
def load_data():
    try:
        trends = pd.read_csv('reports/skill_analysis.csv')
        return trends
    except Exception:
        return None

@st.cache_data
def load_graph_data():
    try:
        with open('models/graph_data.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data
def load_raw_vacancies():
    """Загрузка исходных данных о вакансиях (если есть)"""
    for fname in ('results.csv', 'results_cleaned.csv'):
        try:
            df = pd.read_csv(fname, encoding='utf-8-sig')
            return df
        except Exception:
            continue
    return None

@st.cache_data
def load_model_metrics():
    try:
        return pd.read_csv('reports/model_metrics.csv').iloc[0].to_dict()
    except Exception:
        return None

@st.cache_data
def load_weekly_demand():
    try:
        return pd.read_csv('reports/weekly_demand.csv')
    except Exception:
        return None

@st.cache_data
def load_adf_results():
    try:
        return pd.read_csv('reports/adf_results.csv')
    except Exception:
        return None

@st.cache_data
def load_arima_comparison():
    try:
        return pd.read_csv('reports/arima_comparison.csv')
    except Exception:
        return None

@st.cache_data
def load_classification_report():
    try:
        return pd.read_csv('reports/classification_report.csv')
    except Exception:
        return None

@st.cache_data
def load_kmeans():
    try:
        return pd.read_csv('reports/kmeans_skills.csv')
    except Exception:
        return None

def get_cluster_name(skills):
    """Dynamically determine cluster name based on its skills"""
    skills_set = set([s.lower() for s in skills])
    
    if any(s in skills_set for s in ['python', 'django', 'flask', 'fastapi']):
        return "Python Ecosystem"
    if any(s in skills_set for s in ['react', 'javascript', 'vue', 'angular', 'typescript']):
        return "Frontend & Web Ecosystem"
    if any(s in skills_set for s in ['java', 'spring', 'c#', 'oracle', 'dotnet']):
        return "Enterprise Backend"
    if any(s in skills_set for s in ['docker', 'kubernetes', 'linux', 'jenkins', 'aws']):
        return "DevOps & Cloud Infrastructure"
    if any(s in skills_set for s in ['sql', 'postgres', 'postgresql', 'mysql', 'mongodb', 'redis']):
        return "Databases & Storage"
    if any(s in skills_set for s in ['swift', 'kotlin', 'android', 'ios']):
        return "Mobile Development"
    if any(s in skills_set for s in ['go', 'golang']):
        return "Go Ecosystem"
    if any(s in skills_set for s in ['ml', 'data', 'pandas', 'numpy', 'excel']):
        return "Data Science & Analytics"
    if any(s in skills_set for s in ['c++']):
        return "Systems Programming"
    
    # Fallback
    if skills:
        return f"{skills[0].title()} Ecosystem"
    return "Other Ecosystem"

trends_df         = load_data()
graph_data        = load_graph_data()
raw_df            = load_raw_vacancies()
model_metrics     = load_model_metrics()
weekly_demand_df  = load_weekly_demand()
adf_df            = load_adf_results()
arima_df          = load_arima_comparison()
cls_df            = load_classification_report()
kmeans_df         = load_kmeans()

# ========================
# GENERAL STATISTICS BLOCK
# ========================
if raw_df is not None:
    st.markdown("---")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Total Vacancies", len(raw_df))
    col_s2.metric("Unique Cities", raw_df['city'].nunique() if 'city' in raw_df.columns else "—")
    col_s3.metric("Unique Employers", raw_df['job'].nunique() if 'job' in raw_df.columns else "—")
    col_s4.metric("Date Range", f"{pd.to_datetime(raw_df['publish_date'], errors='coerce').min().date()} – {pd.to_datetime(raw_df['publish_date'], errors='coerce').max().date()}" if 'publish_date' in raw_df.columns else "—")
    
    if trends_df is not None:
        col_s5, col_s6, col_s7 = st.columns(3)
        col_s5.metric("Skills Tracked", len(trends_df))
        col_s6.metric("Avg Salary (Top-10)", f"{trends_df.head(10)['avg_salary'].mean():,.0f} KZT")
        col_s7.metric("Total Skill Mentions", int(trends_df['total_vacancies'].sum()))
    st.markdown("---")

# ========================
# PAGE: ABOUT & METHODOLOGY
# ========================
if page == "About & Methodology":
    st.header("About the Project & Methodology")

    st.markdown("""
    > **Diploma Thesis** — *Forecasting the Demand for IT Professions and Skills Using Machine Learning*
    >
    > **University:** Astana IT University &nbsp;|&nbsp; **Year:** 2025
    > **Domain:** Labor market analytics, NLP, Time-series forecasting, Graph analysis
    """)

    st.markdown("---")

    # ── DATASET ──────────────────────────────────────────────────────────────
    st.subheader("1. Dataset")
    st.markdown("""
    | Property | Value |
    |---|---|
    | **Source** | HeadHunter Kazakhstan (hh.kz) — automated web scraping |
    | **Period** | ~6 months of weekly snapshots |
    | **Raw vacancies** | **5 262** job postings |
    | **Features collected** | Title, description, requirements, salary, city, employer, publication date, experience level, schedule, employment type |
    | **Target region** | Republic of Kazakhstan |

    **Preprocessing steps applied:**
    - HTML tag removal and Unicode normalization
    - Salary parsing: extracted numeric KZT values; anomalous salaries (`< 50 000` or `> 10 000 000 KZT`) set to `NaN` — **189 values** corrected
    - Skill alias normalization: `js → javascript`, `go → golang`, `ts → typescript`, `k8s → kubernetes`, `node / nestjs → nodejs`, `sklearn → scikit-learn`, `ml / deep learning → machine learning`, `postgres → postgresql`, `ci/cd → cicd`
    - Weekly aggregation: each `(skill, ISO-week)` pair becomes one row — **841 records**, **44 unique skills**
    """)

    st.markdown("---")

    # ── DEMAND INDEX ─────────────────────────────────────────────────────────
    st.subheader("2. Composite Demand Index (Target Variable)")
    st.markdown(r"""
    The **Demand Index** is a normalized composite score $\in [0, 1]$ computed for each `(skill, week)` pair.
    It combines six sub-scores, each min-max normalized to $[0, 1]$:

    $$
    \text{DemandIndex} = 0.30 \cdot V + 0.25 \cdot S + 0.15 \cdot E + 0.10 \cdot G + 0.10 \cdot T + 0.10 \cdot C
    $$

    | Symbol | Sub-score | Weight | Description |
    |---|---|---|---|
    | $V$ | **Vacancy Score** | 30% | Weekly vacancy count (normalized) |
    | $S$ | **Salary Score** | 25% | Average salary level |
    | $E$ | **Employer Score** | 15% | Number of unique employers hiring for this skill |
    | $G$ | **Geo Score** | 10% | Number of unique cities with vacancies |
    | $T$ | **Growth Score** | 10% | 4-week rolling growth trend (slope) |
    | $C$ | **Competition Score** | 10% | Ratio of vacancies to competing skills (inverted) |

    **Demand Level** (classification target): the index is binned into 5 ordinal classes —
    *Very Low (1), Low (2), Medium (3), High (4), Very High (5)* — using quintile thresholds.
    """)

    st.markdown("---")

    # ── METHODS ──────────────────────────────────────────────────────────────
    st.subheader("3. Methods & Models")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("""
**Supervised — Regression**
Predicts the continuous `demand_index` for the next observation.

| Model | Purpose |
|---|---|
| **Ridge Regression** | Linear baseline with L2 regularization |
| **Gradient Boosting** | Main model — ensemble of shallow trees |
| **LightGBM** | Fast gradient boosting, handles high-dimensional features |

*Validation:* chronological 80/20 split (last 4 weeks = test).
*CV:* `TimeSeriesSplit(n_splits=3)` — no data leakage.

---

**Supervised — Classification**
Predicts `demand_level` (5 classes).

| Model | Test Accuracy |
|---|---|
| Logistic Regression | 44.7% |
| Random Forest | 74.5% |
| **Gradient Boosting** | **77.3%** |

---

**Time Series Forecasting**
Automates 4-week ahead demand prediction.

| Model | Description |
|---|---|
| **Persistence** | Last known value — naïve baseline |
| **MA-4** | Moving average over last 4 weeks |
| **ARIMA(p,d,q)** | Order selected by AIC grid search |
| **Gradient Boosting** | Feature-based ML — consistently best |
        """)

    with col_m2:
        st.markdown("""
**Unsupervised — Graph Clustering**
Discovers skill ecosystems from co-occurrence patterns.

| Step | Detail |
|---|---|
| **Co-occurrence matrix** | Count how often two skills appear in the same vacancy |
| **Graph construction** | NetworkX graph; node size = vacancy count, edge weight = co-occurrence |
| **Community detection** | Greedy Modularity Communities algorithm |
| **Result** | 7 skill ecosystems (Python, Frontend, Enterprise Java, DevOps, Databases, Mobile, Data/ML) |

---

**Unsupervised — K-Means Clustering**
Groups skills by demand statistics into behavioral clusters.

| Feature used | Description |
|---|---|
| `total_vacancies` | Overall market presence |
| `avg_weekly_vacancies` | Weekly activity level |
| `avg_salary` | Compensation level |
| `high_demand_weeks` | Consistency of demand |
| `avg_demand_index` | Composite score |

Optimal *k* selected via elbow method (inertia drop).

---

**Explainability — SHAP**
`TreeExplainer` computes Shapley values for the Gradient Boosting model,
showing which features drive each prediction up or down.

---

**Stationarity — ADF Test**
Augmented Dickey-Fuller test on each skill's weekly `vacancy_count`.
18 of 44 skills have stationary series (p < 0.05).
        """)

    st.markdown("---")

    # ── FEATURE ENGINEERING ──────────────────────────────────────────────────
    st.subheader("4. Feature Engineering (35 features)")
    st.markdown("""
    Features are computed per `(skill, week)` row:

    | Group | Features |
    |---|---|
    | **Raw demand** | `vacancy_count`, `avg_salary`, `unique_cities`, `unique_employers` |
    | **Sub-scores** | `vacancy_score`, `salary_score`, `employer_score`, `geo_score`, `growth_score`, `competition_score` |
    | **Rolling stats** | 2-week and 4-week rolling mean & std of `vacancy_count` and `demand_index` |
    | **Lag features** | Lag-1 and Lag-2 of `demand_index` |
    | **Temporal** | `week` (ISO week number), `week_sin`, `week_cos` (cyclical encoding) |
    | **Skill identity** | `skill_encoded` (LabelEncoder) |
    | **TF-IDF** | Top-10 TF-IDF components of skill descriptions |

    Cyclical encoding (`sin`/`cos`) prevents the model from treating week 52 and week 1 as far apart.
    """)

    st.markdown("---")

    # ── TECH STACK ────────────────────────────────────────────────────────────
    st.subheader("5. Technology Stack")
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.markdown("""
**Data & ML**
- `pandas`, `numpy`
- `scikit-learn`
- `lightgbm`
- `statsmodels` (ARIMA, ADF)
- `shap`
        """)
    with col_t2:
        st.markdown("""
**Visualization**
- `plotly` (interactive charts)
- `streamlit` (dashboard)
- `matplotlib`, `seaborn` (static plots)
- `pyvis` (network graph)
        """)
    with col_t3:
        st.markdown("""
**Graph & NLP**
- `networkx` (graph construction)
- `re` (regex skill extraction)
- `sklearn.feature_extraction` (TF-IDF)
        """)

    st.markdown("---")

    # ── RESULTS SUMMARY ───────────────────────────────────────────────────────
    st.subheader("6. Key Results")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Best Regressor", "Gradient Boosting")
    r2.metric("Test R²", "0.977")
    r3.metric("Test MAPE", "2.39%")
    r4.metric("Classification Acc.", "77.3%")

    st.success("""
    **Main finding:** The Gradient Boosting model with composite Demand Index features achieves near-perfect
    forecasting accuracy (R² = 0.977, MAPE = 2.39%), significantly outperforming ARIMA (MAE 3–10× higher)
    and naive baselines. The top in-demand skills in Kazakhstan are **JavaScript, SQL, Python, React, and Java**.
    DevOps (Docker, Kubernetes, Linux) form a tightly coupled ecosystem consistently required together.
    """)

# ========================
# PAGE: PROJECT PIPELINE
# ========================
elif page == "Project Pipeline":
    st.header("Project Pipeline Architecture")
    st.markdown("This section visually details the end-to-end Machine Learning pipeline developed for this diploma thesis.")
    
    st.markdown('''
    <style>
    .timeline {
        position: relative;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px 0;
    }
    .timeline::after {
        content: '';
        position: absolute;
        width: 6px;
        background-color: rgba(255,255,255,0.1);
        top: 0;
        bottom: 0;
        left: 50%;
        margin-left: -3px;
    }
    .t-container {
        padding: 10px 40px;
        position: relative;
        background-color: inherit;
        width: 50%;
    }
    .left { left: 0; }
    .right { left: 50%; }
    .t-container::after {
        content: '';
        position: absolute;
        width: 25px;
        height: 25px;
        right: -12px;
        background-color: white;
        border: 4px solid #FF9F55;
        top: 25px;
        border-radius: 50%;
        z-index: 1;
    }
    .right::after {
        left: -12px;
    }
    .content {
        padding: 20px 30px;
        background-color: rgba(255, 255, 255, 0.05);
        position: relative;
        border-radius: 6px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .c1::after { border-color: #3498db; }
    .c2::after { border-color: #2ecc71; }
    .c3::after { border-color: #9b59b6; }
    .c4::after { border-color: #f1c40f; }
    .c5::after { border-color: #e74c3c; }
    .c6::after { border-color: #aaaaaa; }
    </style>
    
    <div class="timeline">
      <div class="t-container left c1">
        <div class="content" style="border-top: 5px solid #3498db;">
          <h3 style="margin-top: 0; color: #3498db;">1. Data Collection</h3>
          <p>Automated web scraping of real IT job vacancies from HeadHunter (hh.kz). Extracted raw HTML requirements, responsibilities, and salaries.</p>
        </div>
      </div>
      <div class="t-container right c2">
        <div class="content" style="border-top: 5px solid #2ecc71;">
          <h3 style="margin-top: 0; color: #2ecc71;">2. Preprocessing & NLP</h3>
          <p>Cleaning raw text and using advanced Natural Language Processing (Regex with boundaries) to accurately extract over 60+ specific IT skills.</p>
        </div>
      </div>
      <div class="t-container left c3">
        <div class="content" style="border-top: 5px solid #9b59b6;">
          <h3 style="margin-top: 0; color: #9b59b6;">3. Graph Analysis (Unsupervised)</h3>
          <p>Building a Co-occurrence Matrix to discover which skills are requested together. Constructing NetworkX graphs and using greedy modularity to cluster skills.</p>
        </div>
      </div>
      <div class="t-container right c4">
        <div class="content" style="border-top: 5px solid #f1c40f;">
          <h3 style="margin-top: 0; color: #f1c40f;">4. Feature Engineering</h3>
          <p>Calculating a highly engineered <b>Composite Demand Index</b> based on vacancy volumes, salaries, and competition. Generating temporal features.</p>
        </div>
      </div>
      <div class="t-container left c5">
        <div class="content" style="border-top: 5px solid #e74c3c;">
          <h3 style="margin-top: 0; color: #e74c3c;">5. Machine Learning</h3>
          <p>Training multiple predictive models (Ridge Regression, Random Forest, Gradient Boosting) to forecast the Demand Index. Validated via Cross-Validation.</p>
        </div>
      </div>
      <div class="t-container right c6">
        <div class="content" style="border-top: 5px solid #aaaaaa;">
          <h3 style="margin-top: 0; color: #aaaaaa;">6. Evaluation & Dashboard</h3>
          <p>Presenting findings, visual charts (Actual vs Predicted, Residuals), and interactive graph visualizations inside this Streamlit application.</p>
        </div>
      </div>
    </div>
    ''', unsafe_allow_html=True)

# ========================
# PAGE: EDA & STATISTICS
# ========================
elif page == "EDA & Statistics":
    st.header("Exploratory Data Analysis")
    st.markdown("Statistical analysis of the dataset: distributions, correlations, stationarity tests.")

    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.eda_analysis import (
        plot_weekly_trend, plot_salary_boxplot, plot_correlation_heatmap,
        plot_vacancy_count_distribution, plot_missing_values, plot_adf_results,
    )

    if weekly_demand_df is not None:
        st.subheader("Weekly Demand Trend (Top Skills)")
        fig_wt = plot_weekly_trend(weekly_demand_df)
        st.plotly_chart(fig_wt, use_container_width=True)

        st.markdown("---")
        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            fig_vdist = plot_vacancy_count_distribution(weekly_demand_df)
            st.plotly_chart(fig_vdist, use_container_width=True)
        with col_eda2:
            fig_corr = plot_correlation_heatmap(weekly_demand_df)
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Run the pipeline first to generate `reports/weekly_demand.csv`.")

    if raw_df is not None:
        st.markdown("---")
        st.subheader("Salary Distribution by Job Category")
        if 'Job' not in raw_df.columns and 'job' in raw_df.columns:
            raw_df = raw_df.copy()
            raw_df['Job'] = raw_df['job']
        if 'salary_numeric' not in raw_df.columns and 'salary' in raw_df.columns:
            import re as _re
            def _p(s):
                if not s or str(s).strip() == '':
                    return float('nan')
                ns = _re.findall(r'\d+', str(s))
                return float(sum(int(n) for n in ns) / len(ns)) if ns else float('nan')
            raw_df = raw_df.copy()
            raw_df['salary_numeric'] = raw_df['salary'].apply(_p)
        fig_box = plot_salary_boxplot(raw_df)
        if fig_box.data:
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")
        st.subheader("Missing Values Analysis")
        fig_miss = plot_missing_values(raw_df)
        if fig_miss.data:
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.success("No missing values in the dataset!")

    if adf_df is not None:
        st.markdown("---")
        st.subheader("ADF Stationarity Test")
        st.markdown(
            "The **Augmented Dickey-Fuller test** checks whether each skill's demand time series is "
            "stationary. Green bars (p < 0.05) indicate stationarity — required for ARIMA modeling."
        )
        fig_adf = plot_adf_results(adf_df)
        st.plotly_chart(fig_adf, use_container_width=True)
        n_stat = adf_df['is_stationary'].sum()
        st.info(f"Stationary series: **{n_stat}** out of **{len(adf_df)}** skills")
    else:
        st.info("ADF results not found. Run the pipeline to generate `reports/adf_results.csv`.")

# ========================
# PAGE: DEMAND ANALYTICS
# ========================
elif page == "Demand Analytics":
    st.header("Top In-Demand IT Skills")
    
    if trends_df is not None:
        # --- TOP SKILLS TABLE ---
        st.dataframe(
            trends_df.style.format({"avg_weekly_vacancies": "{:.1f}", "avg_salary": "{:,.0f} KZT"}),
            use_container_width=True,
            height=300
        )
        
        # --- TOP SKILLS BAR CHART ---
        st.markdown("---")
        st.subheader("📊 Top-10 Skills by Total Vacancies")
        top10 = trends_df.head(10)
        fig_bar = px.bar(
            top10, x='skill', y='total_vacancies',
            color='total_vacancies', color_continuous_scale='viridis',
            text='total_vacancies',
            labels={'skill': 'Skill', 'total_vacancies': 'Total Vacancies'}
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(showlegend=False, height=450)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # --- SALARY DISTRIBUTION ---
        st.subheader("💰 Salary Distribution by Skill")
        fig_salary = px.bar(
            top10, x='skill', y='avg_salary',
            color='avg_salary', color_continuous_scale='greens',
            text=top10['avg_salary'].apply(lambda x: f"{x:,.0f} KZT"),
            labels={'skill': 'Skill', 'avg_salary': 'Average Salary (KZT)'}
        )
        fig_salary.update_traces(textposition='outside')
        fig_salary.update_layout(showlegend=False, height=450)
        st.plotly_chart(fig_salary, use_container_width=True)
        
        # --- SCATTER: VACANCIES vs SALARY ---
        st.subheader("🎯 Vacancies vs Salary (Bubble Chart)")
        fig_scatter = px.scatter(
            trends_df, x='total_vacancies', y='avg_salary',
            size='high_demand_weeks', color='high_demand_weeks',
            hover_name='skill', text='skill',
            labels={
                'total_vacancies': 'Total Vacancies',
                'avg_salary': 'Average Salary (KZT)',
                'high_demand_weeks': 'High Demand Weeks'
            },
            color_continuous_scale='plasma'
        )
        fig_scatter.update_traces(textposition='top center', textfont_size=10)
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # --- DETAILED SKILL ANALYSIS ---
        st.markdown("---")
        st.subheader("🔍 Detailed Skill Analysis")
        selected_skill = st.selectbox("Select a skill for details:", trends_df['skill'].unique())
        skill_info = trends_df[trends_df['skill'] == selected_skill].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Mentions (Vacancies)", int(skill_info['total_vacancies']))
        col2.metric("Avg. Vacancies per Week", f"{skill_info['avg_weekly_vacancies']:.1f}")
        col3.metric("Avg. Salary", f"{skill_info['avg_salary']:,.0f} KZT")
        col4.metric("Weeks of High Demand", int(skill_info['high_demand_weeks']))
        
        # --- RAW DATA STATS (if available) ---
        if raw_df is not None:
            st.markdown("---")
            st.subheader("📋 Dataset Overview")
            
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                # City distribution
                if 'city' in raw_df.columns:
                    city_df = raw_df['city'].value_counts().head(10).reset_index()
                    city_df.columns = ['City', 'Vacancies']
                    fig_city = px.bar(
                        city_df, x='Vacancies', y='City',
                        orientation='h', color='Vacancies',
                        color_continuous_scale='blues',
                        title='Top-10 Cities by Vacancies'
                    )
                    fig_city.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig_city, use_container_width=True)
            
            with col_r2:
                # Experience distribution
                if 'experience' in raw_df.columns:
                    exp_counts = raw_df['experience'].value_counts()
                    fig_exp = px.pie(
                        values=exp_counts.values, names=exp_counts.index,
                        title='Experience Level Distribution',
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    fig_exp.update_layout(height=400)
                    st.plotly_chart(fig_exp, use_container_width=True)
            
            # Employment type
            col_r3, col_r4 = st.columns(2)
            with col_r3:
                if 'schedule' in raw_df.columns:
                    sched_df = raw_df['schedule'].value_counts().reset_index()
                    sched_df.columns = ['Schedule', 'Count']
                    fig_sched = px.bar(
                        sched_df, x='Schedule', y='Count',
                        color='Count',
                        color_continuous_scale='teal',
                        title='Work Schedule Distribution'
                    )
                    fig_sched.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig_sched, use_container_width=True)
            
            with col_r4:
                if 'employment' in raw_df.columns:
                    emp_counts = raw_df['employment'].value_counts()
                    fig_emp = px.pie(
                        values=emp_counts.values, names=emp_counts.index,
                        title='Employment Type Distribution',
                        color_discrete_sequence=px.colors.sequential.Purples
                    )
                    fig_emp.update_layout(height=350)
                    st.plotly_chart(fig_emp, use_container_width=True)
            
    else:
        st.warning("Analytics data not found. Run the pipeline (main_skills.py) first.")

# ========================
# PAGE: KAZAKHSTAN MAP
# ========================
elif page == "Kazakhstan Map":
    st.header("IT Vacancy Geography: Kazakhstan")
    st.markdown(
        "Interactive bubble map showing the distribution of IT vacancies across Kazakhstani cities. "
        "Bubble **size** and **color** reflect the number of vacancies. Hover over a city for details."
    )

    if raw_df is not None:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from src.kazakhstan_map import build_kazakhstan_map, build_city_skills_chart

        if 'salary_numeric' not in raw_df.columns and 'salary' in raw_df.columns:
            import re as _re
            def _parse_sal(s):
                if not s or str(s).strip() == '':
                    return float('nan')
                nums = _re.findall(r'\d+', str(s))
                return float(sum(int(n) for n in nums) / len(nums)) if nums else float('nan')
            raw_df = raw_df.copy()
            raw_df['salary_numeric'] = raw_df['salary'].apply(_parse_sal)

        fig_map = build_kazakhstan_map(raw_df)
        st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("---")
        col_map1, col_map2 = st.columns([3, 2])

        with col_map1:
            fig_cities = build_city_skills_chart(raw_df, top_n_cities=10)
            st.plotly_chart(fig_cities, use_container_width=True)

        with col_map2:
            st.subheader("City Statistics")
            city_tbl = raw_df['city'].value_counts().reset_index()
            city_tbl.columns = ['City', 'Vacancies']
            city_tbl['Share (%)'] = (city_tbl['Vacancies'] / city_tbl['Vacancies'].sum() * 100).round(1)
            st.dataframe(city_tbl, use_container_width=True, height=380)

        st.markdown("---")
        st.subheader("Average Salary by City (KZT)")
        if 'salary_numeric' in raw_df.columns:
            sal_city = (
                raw_df[raw_df['salary_numeric'].notna() & (raw_df['salary_numeric'] > 50_000)]
                .groupby('city')['salary_numeric']
                .agg(['mean', 'count'])
                .reset_index()
                .rename(columns={'mean': 'avg_salary', 'count': 'n'})
            )
            sal_city = sal_city[sal_city['n'] >= 5].sort_values('avg_salary', ascending=False).head(10)
            fig_sal = px.bar(
                sal_city, x='city', y='avg_salary',
                color='avg_salary', color_continuous_scale='Teal',
                text=sal_city['avg_salary'].apply(lambda x: f"{x:,.0f}"),
                labels={'city': 'City', 'avg_salary': 'Avg Salary (KZT)'},
            )
            fig_sal.update_traces(textposition='outside')
            fig_sal.update_layout(showlegend=False, height=380, xaxis_tickangle=-30)
            st.plotly_chart(fig_sal, use_container_width=True)
    else:
        st.warning("Data not found. Place results.csv in the project root.")

# ========================
# PAGE: SKILL ECOSYSTEMS
# ========================
elif page == "Skill Ecosystems (Graph)":
    st.header("Interactive Skill Ecosystems Graph")
    st.markdown("""
    **Unsupervised Learning (Graph Clustering)**: Greedy Modularity algorithm automatically groups skills
    that are most often required together. Each **color** = one ecosystem.
    Node **size** = total vacancies mentioning the skill. Edge **thickness** = co-occurrence frequency.
    Zoom, drag, and hover over nodes for details.
    """)
    
    graph_html_path = 'reports/figures/skill_graph.html'
    if os.path.exists(graph_html_path):
        import streamlit.components.v1 as components
        with open(graph_html_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        components.html(source_code, height=740, scrolling=False)
    else:
        st.warning("Graph not found. Run: `python main_skills.py --mode train --data results.csv`")

    st.markdown("---")
    st.subheader("Ecosystem Clusters")

    # Цвета синхронизированы с graph_analysis.py CLUSTER_COLORS
    CLUSTER_COLORS = [
        '#4E9AF1', '#F4A261', '#2EC4B6', '#E63946',
        '#A8DADC', '#FFD166', '#C77DFF', '#06D6A0',
        '#EF476F', '#118AB2',
    ]

    if graph_data and 'nodes' in graph_data:
        clusters  = {}
        node_sizes = {}
        for node in graph_data['nodes']:
            group = node.get('group', 0)
            skill = node.get('id', '')
            clusters.setdefault(group, []).append(skill)
            node_sizes[skill] = node.get('size', 0)

        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
        cols = st.columns(3)

        for idx, (group, skills) in enumerate(sorted_clusters):
            if len(skills) <= 1:
                continue
            cluster_name  = get_cluster_name(skills)
            color         = CLUSTER_COLORS[int(group) % len(CLUSTER_COLORS)]
            col_idx       = idx % 3
            skills_sorted = sorted(skills, key=lambda s: node_sizes.get(s, 0), reverse=True)

            badges_html = "".join([
                f'<span style="background:{color}22; border:1px solid {color}66; '
                f'padding:3px 10px; border-radius:12px; font-size:0.82em; '
                f'display:inline-block; margin-bottom:5px; color:white;">'
                f'{s} <span style="opacity:0.55; font-size:0.85em">({node_sizes.get(s,0)})</span>'
                f'</span>'
                for s in skills_sorted
            ])

            cols[col_idx].markdown(f'''
            <div style="border-left:5px solid {color}; padding:14px 16px;
                        margin-bottom:16px; border-radius:8px;
                        background:linear-gradient(135deg,{color}18,{color}05);">
                <div style="font-weight:700; font-size:1.05em; color:{color}; margin-bottom:4px;">
                    {cluster_name}
                </div>
                <div style="font-size:0.78em; color:#999; margin-bottom:10px;">
                    {len(skills)} skills
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:5px;">
                    {badges_html}
                </div>
            </div>''', unsafe_allow_html=True)

        # Graph statistics
        st.markdown("---")
        st.subheader("Graph Statistics")
        links = graph_data.get('links', [])
        nodes = graph_data.get('nodes', [])
        real_clusters = [c for c, s in clusters.items() if len(s) > 1]

        cg1, cg2, cg3, cg4 = st.columns(4)
        cg1.metric("Skills (Nodes)",       len(nodes))
        cg2.metric("Connections (Edges)",  len(links))
        cg3.metric("Ecosystems",           len(real_clusters))
        avg_sz = np.mean([len(s) for _, s in sorted_clusters if len(s) > 1]) if real_clusters else 0
        cg4.metric("Avg Ecosystem Size",   f"{avg_sz:.1f}")

        if links:
            st.subheader("Top-20 Skill Co-occurrences")
            link_df  = pd.DataFrame(links)
            top_links = link_df.nlargest(20, 'weight').copy()
            top_links['pair'] = top_links.apply(
                lambda r: f"{r['source']} ↔ {r['target']}", axis=1
            )
            fig_links = px.bar(
                top_links, y='pair', x='weight', orientation='h',
                color='weight', color_continuous_scale='Plasma',
                labels={'weight': 'Co-occurrence count', 'pair': 'Skill pair'},
            )
            fig_links.update_layout(height=540, showlegend=False)
            st.plotly_chart(fig_links, use_container_width=True)
    else:
        st.info("Graph data not found. Run the pipeline to generate it.")

# ========================
# PAGE: ML MODEL EVALUATION
# ========================
elif page == "ML Model Evaluation":
    st.header("ML Model Evaluation (Regression)")

    if model_metrics:
        model_name   = model_metrics.get('model', 'Best Model')
        test_weeks   = int(model_metrics.get('test_weeks', 4))
        train_r2     = model_metrics.get('train_r2', float('nan'))
        test_r2      = model_metrics.get('test_r2',  float('nan'))
        test_mae     = model_metrics.get('test_mae',  float('nan'))
        test_rmse    = model_metrics.get('test_rmse', float('nan'))
        test_mape    = model_metrics.get('test_mape', float('nan'))

        st.markdown(f"""
The best model is **{model_name}**, evaluated on a **chronological test set** (last **{test_weeks} weeks** of data).
This split prevents data leakage — the model never sees future information during training.
        """)

        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Train R²",    f"{train_r2:.3f}")
        col_m2.metric("Test R²",     f"{test_r2:.3f}")
        col_m3.metric("Test MAE",    f"{test_mae:.4f}")
        col_m4.metric("Test RMSE",   f"{test_rmse:.4f}")
        col_m5.metric("Test MAPE",   f"{test_mape:.1f}%")
    else:
        st.info("Метрики не найдены. Запустите пайплайн: `python main_skills.py --mode train --data results.csv`")
    
    col1, col2 = st.columns(2)
    
    actual_pred_path = 'reports/figures/skill_actual_vs_predicted.png'
    if os.path.exists(actual_pred_path):
        col1.image(actual_pred_path, caption="Actual vs Predicted Index", use_column_width=True)
        
    residuals_path = 'reports/figures/skill_residuals.png'
    if os.path.exists(residuals_path):
        col2.image(residuals_path, caption="Residuals Plot", use_column_width=True)

    st.markdown("---")
    
    # Additional model comparison chart
    model_comp_path = 'reports/figures/skill_model_comparison.png'
    if os.path.exists(model_comp_path):
        st.subheader("🤖 ML Models Comparison")
        st.image(model_comp_path, caption="Ridge vs Random Forest vs Gradient Boosting (R² Score)", use_column_width=True)
    
    # Demand distribution
    dist_path = 'reports/figures/skill_demand_distribution.png'
    if os.path.exists(dist_path):
        st.subheader("📊 Demand Distribution")
        st.image(dist_path, caption="Overall Demand Distribution and Top Skills", use_column_width=True)
    
    # Feature Importance
    feat_path = 'reports/figures/skill_feature_importance.png'
    if os.path.exists(feat_path):
        st.subheader("Feature Importance")
        st.image(feat_path, caption="Top Feature Importances", use_column_width=True)

    st.markdown("---")
    st.subheader("Learning Curves")
    lc_path = 'reports/figures/learning_curves.png'
    if os.path.exists(lc_path):
        st.image(lc_path, caption="Learning Curves (Train vs CV R²)", use_column_width=True)
    else:
        st.info("Learning curves not found. Run the pipeline first.")

    shap_path = 'reports/figures/shap_summary.png'
    if os.path.exists(shap_path):
        st.subheader("SHAP Feature Explanations")
        st.markdown("Each dot is one sample. **Color** = feature value (red=high, blue=low). "
                    "**X position** = SHAP value (impact on prediction).")
        st.image(shap_path, caption="SHAP Summary Plot", use_column_width=True)

    st.markdown("---")
    st.subheader("Classification: Demand Level (5 classes)")
    st.markdown("Classifiers predict `demand_level` (1=Very Low … 5=Very High) from skill features.")
    if cls_df is not None:
        st.dataframe(cls_df.style.format({'accuracy': '{:.4f}', 'f1_weighted': '{:.4f}'}),
                     use_container_width=True)
        cm_path = 'reports/figures/confusion_matrix.png'
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix (best classifier)", use_column_width=True)
    else:
        st.info("Classification report not found. Run the pipeline first.")

    st.markdown("---")
    st.subheader("ARIMA vs Naive Baselines")
    if arima_df is not None:
        st.dataframe(arima_df, use_container_width=True)
        bl_path = 'reports/figures/baseline_comparison.png'
        if os.path.exists(bl_path):
            st.image(bl_path, caption="MAE: Persistence vs MA-4 vs ARIMA vs Gradient Boosting",
                     use_column_width=True)
    else:
        st.info("ARIMA comparison not found. Run the pipeline first.")

    st.markdown("---")
    st.subheader("K-Means Skill Clusters")
    if kmeans_df is not None:
        st.dataframe(
            kmeans_df[['skill', 'cluster', 'cluster_label', 'total_vacancies',
                        'avg_weekly_vacancies', 'avg_demand_index']]
            .sort_values(['cluster', 'total_vacancies'], ascending=[True, False])
            .style.format({'total_vacancies': '{:.0f}', 'avg_weekly_vacancies': '{:.2f}',
                           'avg_demand_index': '{:.4f}'}),
            use_container_width=True,
        )
    else:
        st.info("K-Means results not found. Run the pipeline first.")

# ========================
# PAGE: FORECAST
# ========================
elif page == "Forecast":
    st.header("Demand Forecast (ARIMA)")
    st.markdown(
        "Select a skill to see its historical **Demand Index** and a 4-week ARIMA forecast "
        "with an 80% confidence interval."
    )

    if weekly_demand_df is None:
        st.warning("weekly_demand.csv not found. Run the pipeline first.")
    else:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from src.arima_forecast import get_skill_forecast

        available_skills = sorted(weekly_demand_df['skill'].unique().tolist())
        selected = st.selectbox("Choose a skill:", available_skills, index=0)

        with st.spinner("Fitting ARIMA model..."):
            fc = get_skill_forecast(weekly_demand_df, selected, n_forecast=4)

        hist_y      = fc['history_y']
        hist_weeks  = fc['history_weeks']
        fc_mean     = fc['forecast_mean']
        ci_lo       = fc['ci_lower']
        ci_hi       = fc['ci_upper']
        future_lbl  = fc['future_labels']
        arima_order = fc['arima_order']

        # Build figure
        fig_fc = go.Figure()

        # Historical
        fig_fc.add_trace(go.Scatter(
            x=list(hist_weeks), y=list(hist_y),
            mode='lines+markers',
            name='Historical',
            line=dict(color='#4E9AF1', width=2),
            marker=dict(size=5),
        ))

        # CI band
        all_fc_x = future_lbl
        fig_fc.add_trace(go.Scatter(
            x=all_fc_x + all_fc_x[::-1],
            y=list(ci_hi) + list(ci_lo)[::-1],
            fill='toself',
            fillcolor='rgba(6,214,160,0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='80% CI',
            showlegend=True,
        ))

        # Forecast line
        fig_fc.add_trace(go.Scatter(
            x=all_fc_x, y=list(fc_mean),
            mode='lines+markers',
            name=f'ARIMA{arima_order} Forecast',
            line=dict(color='#06D6A0', width=2, dash='dash'),
            marker=dict(size=8, symbol='diamond'),
        ))

        fig_fc.update_layout(
            title=f'Demand Index Forecast: {selected}  |  ARIMA{arima_order}',
            xaxis_title='Week',
            yaxis_title='Demand Index',
            hovermode='x unified',
            height=460,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(tickangle=-45, gridcolor='rgba(255,255,255,0.08)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        col_fc1, col_fc2, col_fc3 = st.columns(3)
        col_fc1.metric("Last Demand Index", f"{hist_y[-1]:.4f}")
        col_fc2.metric("Forecasted (W+1)",  f"{fc_mean[0]:.4f}")
        col_fc3.metric("ARIMA Order",       arima_order)

        if arima_df is not None and selected in arima_df['skill'].values:
            row = arima_df[arima_df['skill'] == selected].iloc[0]
            st.markdown("---")
            st.subheader("Model Comparison for this Skill")
            comp_cols = ['mae_persistence', 'mae_ma4', 'mae_arima']
            comp_data = {c.replace('mae_', '').upper(): row.get(c, float('nan'))
                         for c in comp_cols if c in row.index}
            if comp_data:
                fig_cmp = go.Figure(go.Bar(
                    x=list(comp_data.keys()),
                    y=list(comp_data.values()),
                    marker_color=['#F4A261', '#4E9AF1', '#2EC4B6'],
                    text=[f"{v:.5f}" for v in comp_data.values()],
                    textposition='outside',
                ))
                fig_cmp.update_layout(
                    title=f'MAE Comparison — {selected}',
                    yaxis_title='MAE',
                    height=320,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
                )
                st.plotly_chart(fig_cmp, use_container_width=True)