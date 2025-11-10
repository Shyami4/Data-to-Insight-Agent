# Data-to-Insight Agent  
*A Lightweight AI Prototype for Automated Business Insights*

---

## Overview
The **Data-to-Insight Agent** is a **Streamlit-based AI prototype** that transforms raw CSV uploads into executive-ready summaries, business KPIs, and strategic recommendations, along with an intelligent AI Chat bot that responds to natural language queries with answers set in the data and context.

This project demonstrates:
- Real-time insights from structured data (no analyst wait times)
- Trend, benchmark, and anomaly detection
- Responsible AI with local data processing and anonymized context

---

## Live Demo

**Public App:** [https://data2insights.streamlit.app/](https://data2insights.streamlit.app/)  

Upload the **sample dataset** from `sample_data/sales_demo.csv` to explore full functionality — including sales trends, store benchmarks, and AI-generated summaries.

---

## System Architecture
📂 Data-to-Insight-Agent
│
├── app.py # Main Streamlit app
├── analytics_pipeline.py # KPI and trend calculations
├── narrative.py # AI-driven summary generation
├── plots.py # Plotly-based charts
├── chart_style.py # Visualization styles
│
├── config/
│ └── rules.yaml # Performance thresholds and AI parameters
│
├── images/
│ ├── app_icon.png
│ └── app_banner.png
│
├── sample_data/
│ └── sales_demo.csv # Example dataset
│
├── requirements.txt # Dependencies
├── LICENSE # MIT License
└── README.md # This documentation


### Workflow Summary

1. **Data Ingestion Layer**
   - Upload CSV → Schema validation → Handle missing values  
2. **Analytics Pipeline**
   - Compute weekly KPIs, identify growth/decline, and benchmark stores/departments  
3. **AI Intelligence Layer**
   - Generates executive summaries, risk alerts, and recommended actions  
4. **Visualization Layer**
   - Interactive Streamlit dashboard with charts, trend tables, and AI insights  

---

## Setup Instructions

### Clone the repository
```bash
git clone https://github.com/Shyami4/data-to-insight-agent.git
cd data-to-insight-agent
```

### Create a Virtual environment
```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Add your OpenAI API key
Create a .env file in the project root directory with the following content:
```bash
OPENAI_API_KEY=your_api_key_here
```

### Run the app locally
```bash
streamlit run app.py
```

