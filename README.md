# 📈 Forecasting IT Skills Demand (Diploma Thesis)

## 📌 Overview
This repository contains the full source code for a diploma thesis focused on **Forecasting the demand for IT professions and skills using Machine Learning methods**. 

The project analyzes real-world job market data, extracts relevant IT skills using Natural Language Processing (NLP), identifies "skill ecosystems" using Graph Theory, and predicts the exact demand index for each skill using Regression models. The entire pipeline is wrapped in a modern, interactive **Streamlit** dashboard.

## 🚀 Key Features
* **Advanced Data Pipeline**: End-to-end processing from raw HTML vacancy texts to structured ML-ready datasets.
* **NLP Skill Extraction**: Precise extraction of over 60+ IT skills using custom Regex with word boundaries.
* **Graph Analysis (Unsupervised Learning)**: Co-occurrence matrix calculation to automatically detect and cluster skills into domains (e.g., *Frontend & Web Ecosystem*, *Python Ecosystem*).
* **Demand Indexing**: A robust mathematical feature engineering process that combines vacancy volume, salary levels, and employer competition into a continuous target variable (`Demand Index`).
* **Machine Learning (Supervised Learning)**: Training **Ridge Regression**, **Random Forest**, and **Gradient Boosting** to predict future skill demand with high accuracy ($R^2 \approx 0.93$).
* **Interactive Dashboard**: A dynamic Streamlit web application providing data visualization, Interactive NetworkX/PyVis graphs, and ML model evaluation metrics.

## 🏗️ Project Architecture (Pipeline)

1. **Data Collection**: Scraping real IT job vacancies from job boards (hh.kz).
2. **Preprocessing & NLP**: `src/preprocessing.py` and `src/features.py`. Cleaning raw text, handling missing values, extracting exact IT skills.
3. **Graph Analysis (Unsupervised)**: `src/graph_analysis.py`. Building a network graph of skills that are frequently required together and applying greedy modularity to group them.
4. **Feature Engineering**: Expanding data temporally and calculating the Composite Demand Index.
5. **Machine Learning (Regression)**: `src/models.py`. Training regressors to forecast the index. Cross-validation is used to prevent overfitting.
6. **Evaluation & Dashboard**: `src/evaluation.py` and `dashboard.py`. Generating metrics (MAE, RMSE, $R^2$), visual plots (Actual vs Predicted, Residuals), and rendering the UI.

## 🛠️ Tech Stack
* **Language**: Python 3.10+
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn (Ridge, Random Forest, Gradient Boosting)
* **NLP**: Regular Expressions, NLTK/Spacy patterns
* **Graph Theory**: NetworkX, Community Detection (Greedy Modularity)
* **Visualization**: Matplotlib, Seaborn, PyVis
* **Web App**: Streamlit

## 💻 Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/blvckmage/diploma.git
   cd diploma
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Machine Learning Pipeline**:
   *This step processes the raw data, trains the models, and generates the graph JSON/HTML files.*
   ```bash
   python main_skills.py --mode train --data data/processed/results.csv
   ```

5. **Launch the Streamlit Dashboard**:
   ```bash
   streamlit run dashboard.py
   ```
   *The dashboard will automatically open in your browser at `http://localhost:8501`.*

## 📁 Repository Structure
```text
DIPLOMA/
├── data/
│   └── processed/          # Parsed and cleaned datasets
├── models/                 # Saved models (.joblib) and graph data (JSON)
├── reports/
│   └── figures/            # Generated plots and PyVis HTML graphs
├── src/                    # Source code modules
│   ├── preprocessing.py    # Data cleaning and formatting
│   ├── features.py         # NLP and feature engineering
│   ├── graph_analysis.py   # Unsupervised learning (NetworkX)
│   ├── skill_analysis.py   # Demand calculation and aggregation
│   ├── models.py           # ML Model wrappers and comparators
│   └── evaluation.py       # Evaluation metrics and plotting
├── dashboard.py            # Streamlit interactive application
├── main_skills.py          # Main execution script for the pipeline
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 📝 License
This project was developed as a university diploma thesis.
