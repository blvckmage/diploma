import streamlit as st
import pandas as pd
import json
import os

# Set page config
st.set_page_config(page_title="IT Skills: Analytics & Graphs", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Project Pipeline", "Demand Analytics", "Skill Ecosystems (Graph)", "ML Model Evaluation"]
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

trends_df = load_data()
graph_data = load_graph_data()

if page == "Project Pipeline":
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

elif page == "Demand Analytics":
    st.header("Top In-Demand IT Skills")
    if trends_df is not None:
        st.dataframe(
            trends_df.style.format({"avg_weekly_vacancies": "{:.1f}", "avg_salary": "{:,.0f} KZT"}),
            use_container_width=True,
            height=300
        )
        
        st.markdown("---")
        st.subheader("Detailed Skill Analysis")
        selected_skill = st.selectbox("Select a skill for details:", trends_df['skill'].unique())
        skill_info = trends_df[trends_df['skill'] == selected_skill].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Mentions (Vacancies)", int(skill_info['total_vacancies']))
        col2.metric("Avg. Vacancies per Week", f"{skill_info['avg_weekly_vacancies']:.1f}")
        col3.metric("Avg. Salary", f"{skill_info['avg_salary']:,.0f} KZT")
        col4.metric("Weeks of High Demand", int(skill_info['high_demand_weeks']))
        
    else:
        st.warning("Analytics data not found. Run the pipeline (main_skills.py) first.")

elif page == "Skill Ecosystems (Graph)":
    st.header("Interactive Skill Ecosystems Graph")
    st.markdown("""
    **Unsupervised Learning (Graph Clustering)**: The algorithm automatically determined which skills are most often required together in the same vacancies.
    Colors indicate "ecosystems" (e.g., Frontend stack, Data stack, DevOps stack). You can **zoom in, zoom out, and move** the nodes.
    """)
    
    graph_html_path = 'reports/figures/skill_graph.html'
    if os.path.exists(graph_html_path):
        import streamlit.components.v1 as components
        with open(graph_html_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        components.html(source_code, height=650, scrolling=True)
    else:
        st.warning("Graph file not found. Run the pipeline (main_skills.py) to generate the graph.")
        
    st.markdown("---")
    st.header("Cluster Details")
    st.markdown("Dynamic list of skills belonging to each automatically generated ecosystem.")
    
    if graph_data and 'nodes' in graph_data:
        clusters = {}
        for node in graph_data['nodes']:
            group = node.get('group', 0)
            if group not in clusters:
                clusters[group] = []
            clusters[group].append(node['id'])
            
        # Default Vis.js colors used by Pyvis
        color_palette = [
            "#97C2FC", # Blue
            "#FFFF00", # Yellow
            "#FB7E81", # Red
            "#7BE141", # Green
            "#EB7DF4", # Magenta
            "#FFA807", # Orange
            "#C2FABC", # Light Green
            "#F0A30A", # Dark Orange
            "#800080", # Purple
            "#008080"  # Teal
        ]
            
        # Display clusters in columns
        cols = st.columns(3)
        
        # Sort clusters by number of skills (largest first)
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
        
        for idx, (group, skills) in enumerate(sorted_clusters):
            # Dynamic Name
            cluster_name = get_cluster_name(skills)
            
            # Color
            color = color_palette[int(group) % len(color_palette)]
            col_idx = idx % 3
            
            badges_html = "".join([f'<span style="background-color: rgba(128,128,128,0.2); padding: 3px 10px; border-radius: 12px; font-size: 0.85em; display: inline-block; margin-bottom: 5px;">{skill}</span>' for skill in skills])
            
            cols[col_idx].markdown(f'''
            <div style="
                border-left: 6px solid {color};
                padding: 12px 15px;
                margin-bottom: 15px;
                border-radius: 6px;
                background-color: rgba(128, 128, 128, 0.1);
                min-height: 150px;
            ">
                <div style="font-weight: bold; font-size: 1.1em; margin-bottom: 5px; color: {color};">
                    {cluster_name}
                </div>
                <div style="font-size: 0.8em; color: #888; margin-bottom: 10px;">
                    Cluster ID: {group} | {len(skills)} skills
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 6px;">
                    {badges_html}
                </div>
            </div>
            ''', unsafe_allow_html=True)
    else:
        st.info("No cluster data available.")

elif page == "ML Model Evaluation":
    st.header("ML Model Evaluation (Regression)")
    st.markdown("""
    The machine learning model (**Ridge Regression**) predicts the composite `Demand Index` based on historical data, salaries, and competition.
    
    **Model Metrics on Test Set:**
    * **R-squared Score:** ~0.93 (The model explains 93% of the variance)
    * **MAE (Mean Absolute Error):** ~0.02 (Average prediction error)
    """)
    
    col1, col2 = st.columns(2)
    
    actual_pred_path = 'reports/figures/skill_actual_vs_predicted.png'
    if os.path.exists(actual_pred_path):
        col1.image(actual_pred_path, caption="Actual vs Predicted Index", use_container_width=True)
        
    residuals_path = 'reports/figures/skill_residuals.png'
    if os.path.exists(residuals_path):
        col2.image(residuals_path, caption="Residuals Plot", use_container_width=True)

    st.markdown("---")
    dist_path = 'reports/figures/skill_demand_distribution.png'
    if os.path.exists(dist_path):
        st.image(dist_path, caption="Overall Demand Distribution and Top Skills", use_container_width=True)
