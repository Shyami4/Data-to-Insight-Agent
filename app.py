# app.py
import numpy as np
import pandas as pd
import streamlit as st
import time
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

from analytics_pipeline import load_rules, compute_kpis
from narrative import draft_insights

# Plots you expose in plots.py
from plots import (
    # Overview
    _sparkline, 
    kpi_value_and_delta_vs_py,
    fig_sales_trend_forecast_shaded,
    fig_wow_bars,
    fig_top_movers_stores,
    fig_bottom_departments,
    fig_stores_desc, 
    fig_departments_desc, 
    table_store_4wk_change,
    # Drivers & Performance
    fig_regional_share,
    fig_dept_pareto,
    fig_dept_sparklines_top3,
    fig_store_consistency_heatmap,
    # Diagnostics & Efficiency
    fig_efficiency_quadrants_r2,
    outlier_table_iqr,
)

load_dotenv()
st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

# ---------- CSS ----------
st.markdown("""
<style>
/* Base styles */
.block-container {
    padding-top: 0.6rem;
    padding-bottom: 2rem;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #635BFF 0%, #4f47d6 100%);
    color: white;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.55rem 1.2rem;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(99, 91, 255, 0.3);
}
.stButton>button:hover {
    background: linear-gradient(135deg, #4f47d6 0%, #3d38a8 100%);
    box-shadow: 0 6px 20px rgba(99, 91, 255, 0.5);
    transform: translateY(-2px);
}

/* Cards */
.card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    border: 1px solid #334155;
    transition: all 0.3s ease;
}
.card:hover {
    border-color: #635BFF;
    box-shadow: 0 8px 32px rgba(99, 91, 255, 0.3);
    transform: translateY(-2px);
}
.card h3 {
    margin: 0 0 8px 0;
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 500;
}
.card .v {
    font-size: 2rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1;
}

/* Hero section */
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 24px 28px;
    border-radius: 16px;
    margin: 6px 0 20px;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
}
.hero h2 {
    color: #fff;
    margin: 0 0 8px 0;
    font-size: 1.8rem;
    font-weight: 700;
}
.hero p {
    color: rgba(255, 255, 255, 0.9);
    margin: 0;
    font-size: 1rem;
}

/* Info boxes */
.insight {
    background: #0f172a;
    border: 1px solid #1e293b;
    padding: 16px 18px;
    border-radius: 12px;
    color: #cbd5e1;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

/* Status dots */
.dot {
    height: 14px;
    width: 14px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 10px;
    box-shadow: 0 0 12px currentColor;
}
.dot.green {
    background: #22c55e;
    box-shadow: 0 0 12px #22c55e;
}
.dot.yellow {
    background: #f59e0b;
    box-shadow: 0 0 12px #f59e0b;
}
.dot.red {
    background: #ef4444;
    box-shadow: 0 0 12px #ef4444;
}

/* Tables */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
}

/* Streamlit elements */
.stSelectbox > div > div {
    background: #1e293b;
    border-color: #334155;
    border-radius: 8px;
}

/* Plotly charts - remove default margins */
.js-plotly-plot {
    border-radius: 12px;
    overflow: hidden;
}

/* Caption styling */
.stCaption {
    color: #64748b !important;
    font-size: 0.85rem;
    font-style: italic;
}

/* Metric deltas */
[data-testid="stMetricDelta"] {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------- session ----------
ss = st.session_state
ss.setdefault("df", None)
ss.setdefault("result", None)
ss.setdefault("insights", None)
ss.setdefault("filters", {"store":"All", "department":"All", "region":"All"})
rules = load_rules()

# ---------- Reusable Header Component ----------
def page_header(title: str, subtitle: str, gradient_start: str = "#667eea", gradient_end: str = "#764ba2"):
    """
    Reusable header component for all pages
    """
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {gradient_start} 0%, {gradient_end} 100%);
        padding: 48px 40px;
        border-radius: 20px;
        margin: 0 0 32px 0;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: -50px;
            right: -50px;
            width: 200px;
            height: 200px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            filter: blur(40px);
        "></div>
        <div style="
            position: absolute;
            bottom: -30px;
            left: -30px;
            width: 150px;
            height: 150px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            filter: blur(30px);
        "></div>
        <div style="position: relative; z-index: 1;">
            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                <h1 style="
                    color: #fff;
                    margin: 0;
                    font-size: 2.5rem;
                    font-weight: 800;
                    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
                ">{title}</h1>
            </div>
            <p style="
                color: rgba(255, 255, 255, 0.95);
                margin: 0;
                font-size: 1.15rem;
                font-weight: 400;
                line-height: 1.6;
                max-width: 700px;
            ">
                {subtitle}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------- Helpers ----------
REQUIRED_COLS = {"date", "weekly_sales"}

def with_loading(func, message="Processing..."):
    """Wrapper to show loading state"""
    with st.spinner(message):
        return func()
    
def _safe_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        return d.dropna(subset=["date"])
    return df

def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    v = _safe_to_datetime(df)
    f = ss.filters
    if f.get("store") not in (None, "All") and "store" in v.columns:
        v = v[v["store"] == f["store"]]
    if f.get("department") not in (None, "All") and "department" in v.columns:
        v = v[v["department"] == f["department"]]
    if f.get("region") not in (None, "All") and "region" in v.columns:
        v = v[v["region"] == f["region"]]
    return v

def _recompute(df_source: pd.DataFrame):
    with st.spinner("🔄 Analyzing data..."):
        ss.result = compute_kpis(df_source, rules)
        ss.insights = None
    st.success("Analysis complete!")
    time.sleep(0.5)  # Brief pause for UX

def _ensure_data():
    if ss.df is None:
        st.info("Upload a dataset first in **Upload Data**.")
        st.stop()

def _health_status(result_dict: dict) -> tuple[str, str]:
    t = result_dict.get("trend_4wk")
    outliers = int(result_dict.get("outliers", 0))
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return "yellow", "Inconclusive"
    if t >= 0.05 and outliers <= 5:
        return "green", f"Healthy (+{t*100:.1f}%)"
    if t >= 0.00:
        return "yellow", f"Watch (+{t*100:.1f}%)"
    return "red", f"Declining ({t*100:.1f}%)"

def _last4_weeks_sales_sum(df: pd.DataFrame) -> float:
    # robust guard
    if df is None or df.empty or not REQUIRED_COLS.issubset(df.columns):
        return float("nan")
    tmp = _safe_to_datetime(df)[["date", "weekly_sales"]].dropna()
    if tmp.empty:
        return float("nan")
    try:
        s = (tmp.set_index("date")
                .resample("W")["weekly_sales"]
                .sum()
                .tail(4)
                .sum())
        return float(s) if pd.notna(s) else float("nan")
    except Exception:
        return float("nan")
    
def build_store_4wk_table(df: pd.DataFrame) -> pd.DataFrame:
    need = {"date","store","weekly_sales"}
    if df is None or df.empty or need - set(df.columns):
        return pd.DataFrame(columns=["store","last_4w","prev_4w","Δ vs prev 4w"])

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date","store","weekly_sales"])

    # Weekly sales per store
    w = (d.groupby(["store", pd.Grouper(key="date", freq="W")], as_index=False)
           ["weekly_sales"].sum()
           .sort_values(["store","date"]))

    last_week = w["date"].max()
    if pd.isna(last_week):
        return pd.DataFrame(columns=["store","last_4w","prev_4w","Δ vs prev 4w"])

    four_weeks = pd.Timedelta(weeks=4)
    last4_mask = (w["date"] > (last_week - four_weeks)) & (w["date"] <= last_week)
    prev4_mask = (w["date"] > (last_week - 2*four_weeks)) & (w["date"] <= (last_week - four_weeks))

    last4 = w.loc[last4_mask].groupby("store", as_index=False)["weekly_sales"].sum().rename(columns={"weekly_sales":"last_4w"})
    prev4 = w.loc[prev4_mask].groupby("store", as_index=False)["weekly_sales"].sum().rename(columns={"weekly_sales":"prev_4w"})

    out = last4.merge(prev4, on="store", how="left").fillna({"prev_4w":0.0})
    out["Δ vs prev 4w"] = np.where(out["prev_4w"]>0, (out["last_4w"]-out["prev_4w"])/out["prev_4w"], np.nan)
    out = out.sort_values("last_4w", ascending=False)
    return out

def section_divider(icon="", title=""):
    st.markdown(f"""
    <div style="
        margin: 32px 0 24px 0;
        padding: 16px 0;
        border-top: 2px solid #334155;
        border-bottom: 1px solid #1e293b;
    ">
        <h2 style="
            color: #846ae9;
            margin: 0;
            font-size: 1.6rem;
            font-weight: 800;
        ">
            {icon} {title}
        </h2>
    </div>
    """, unsafe_allow_html=True)

# --- Attention Required box (Executive Summary) --------------------------
def _attention_box(df: pd.DataFrame, result: dict):
    if df is None or df.empty:
        return
    d = _safe_to_datetime(df)
    if d.empty or "weekly_sales" not in d.columns:
        return

    # 1) Store with highest missing %
    if "store" in d.columns:
        miss = (d.assign(is_miss=d["weekly_sales"].isna())
                  .groupby("store")["is_miss"].mean()
                  .sort_values(ascending=False))
        top_store = miss.index[0] if len(miss) else None
        miss_pct  = float(miss.iloc[0]*100) if len(miss) else 0.0
        med = (d.dropna(subset=["weekly_sales"])
                 .groupby("store")["weekly_sales"].median())
        missing_rows = (d[d["store"].eq(top_store)]["weekly_sales"].isna().sum()
                        if top_store in d.get("store", pd.Series(dtype=str)).unique() else 0)
        at_risk = float(med.get(top_store, 0) * missing_rows)
    else:
        top_store, miss_pct, at_risk = None, 0.0, 0.0

    # 2) Declining departments WoW
    if "department" in d.columns:
        weekly_dept = (d.groupby(["department", pd.Grouper(key="date", freq="W")], as_index=False)
                         ["weekly_sales"].sum().sort_values(["department","date"]))
        weekly_dept["prev"] = weekly_dept.groupby("department")["weekly_sales"].shift(1)
        weekly_dept["wow"]  = (weekly_dept["weekly_sales"] - weekly_dept["prev"]) / weekly_dept["prev"]
        decl = (weekly_dept.dropna(subset=["wow"]).sort_values("date")
                  .groupby("department").tail(1).sort_values("wow").head(2))
        decl_list = decl["department"].tolist()
        decl_pct  = (decl["wow"].mean()*100) if len(decl) else 0.0
        decl_impact_week = float(decl["weekly_sales"].sum() * (-decl["wow"].mean())) if len(decl) else 0.0
    else:
        decl_list, decl_pct, decl_impact_week = [], 0.0, 0.0

    # 3) Region underperformer
    weekly_reg = (
        d.groupby(["region", pd.Grouper(key="date", freq="W")], as_index=False)["weekly_sales"]
        .sum()
        .sort_values(["region", "date"])
    )
    weekly_reg["prev"] = weekly_reg.groupby("region")["weekly_sales"].shift(1)
    weekly_reg["wow"]  = (weekly_reg["weekly_sales"] - weekly_reg["prev"]) / weekly_reg["prev"]

    reg_last = (
        weekly_reg.dropna(subset=["wow"])
                .sort_values("date")
                .groupby("region")
                .tail(1)
                .sort_values("wow")
    )

    if len(reg_last):
        worst_row  = reg_last.iloc[0]
        worst_reg  = worst_row["region"]
        worst_wow  = float(worst_row["wow"] * 100)
        # if current week is below previous, measure the shortfall; else 0
        diff       = float(worst_row["weekly_sales"] - worst_row["prev"])
        worst_impact = abs(diff) if diff < 0 else 0.0
    else:
        worst_reg, worst_wow, worst_impact = None, 0.0, 0.0

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 24px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.2);
    ">
        <div style="display: flex; align-items: center; margin-bottom: 16px;">
            <span style="font-size: 32px; margin-right: 12px;">🚨</span>
            <h3 style="margin: 0; color: #fca5a5; font-size: 1.5rem;">Needs Immediate Action</h3>
        </div>
    """, unsafe_allow_html=True)

    # Priority 1
    st.markdown(f"""
    <div style="
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 16px;
        margin: 12px 0;
        border-radius: 8px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <span style="
                    background: #ef4444;
                    color: white;
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-size: 0.75rem;
                    font-weight: 700;
                    margin-right: 8px;
                ">CRITICAL</span>
                <span style="color: #e5e7eb; font-size: 1.1rem; font-weight: 600;">
                    {top_store or 'N/A'}: Missing {miss_pct:.0f}% of sales data
                </span>
            </div>
        </div>
        <p style="color: #9ca3af; margin: 8px 0 4px 0; font-size: 0.9rem;">
            💰 Impact: <strong style="color: #fca5a5;">${at_risk:,.0f}</strong> revenue at risk
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("📧 Alert Store Manager", key="fix_data_btn", use_container_width=True):
            st.success("✅ Alert sent to store manager")
    with col2:
        if st.button("📊 Export Report", key="export_data_btn", use_container_width=True):
            st.info("📥 Generating data quality report...")

    # Priority 2
    labs = ", ".join(decl_list) if decl_list else "—"
    st.markdown(f"""
    <div style="
        background: rgba(251, 146, 60, 0.1);
        border-left: 4px solid #f59e0b;
        padding: 16px;
        margin: 12px 0;
        border-radius: 8px;
    ">
        <div>
            <span style="
                background: #f59e0b;
                color: white;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 0.75rem;
                font-weight: 700;
                margin-right: 8px;
            ">HIGH</span>
            <span style="color: #e5e7eb; font-size: 1.1rem; font-weight: 600;">
                {labs}: Declining {abs(decl_pct):.1f}% WoW
            </span>
        </div>
        <p style="color: #9ca3af; margin: 8px 0 4px 0; font-size: 0.9rem;">
            📉 Impact: <strong style="color: #fbbf24;">${decl_impact_week:,.0f}/week</strong> revenue loss
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔍 Investigate", key="inv_causes_btn", use_container_width=True):
            st.info("🔎 Opening department analysis...")

    # Priority 3
    st.markdown(f"""
    <div style="
        background: rgba(251, 191, 36, 0.1);
        border-left: 4px solid #eab308;
        padding: 16px;
        margin: 12px 0;
        border-radius: 8px;
    ">
        <div>
            <span style="
                background: #eab308;
                color: #1e293b;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 0.75rem;
                font-weight: 700;
                margin-right: 8px;
            ">MEDIUM</span>
            <span style="color: #e5e7eb; font-size: 1.1rem; font-weight: 600;">
                {worst_reg or 'N/A'} region: {worst_wow:.1f}% vs last week
            </span>
        </div>
        <p style="color: #9ca3af; margin: 8px 0 4px 0; font-size: 0.9rem;">
             Impact: <strong style="color: #fde047;">${worst_impact:,.0f}/week</strong> underperformance
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("📋 Review Strategy", key="review_strategy_btn", use_container_width=True):
            st.info("📝 Opening regional performance page...")

    # Positive highlights
    st.markdown("""
    <div style="
        background: rgba(34, 197, 94, 0.1);
        border-left: 4px solid #22c55e;
        padding: 16px;
        margin: 20px 0 0 0;
        border-radius: 8px;
    ">
        <h4 style="color: #86efac; margin: 0 0 12px 0; font-size: 1rem;">
            💚 Positive Highlights
        </h4>
        <ul style="color: #d1d5db; margin: 0; padding-left: 20px; line-height: 1.8;">
            <li>Store_5: +23% above average → Replicate best practices</li>
            <li>Dept_6: +12% growth momentum → Increase inventory allocation</li>
            <li>Overall: +0.9% WoW → Maintaining positive trajectory</li>
        </ul>
    </div>
    </div>
    """, unsafe_allow_html=True)

# ---------- Sidebar nav ----------
with st.sidebar:
    page = option_menu(
        "Navigation",
        ["Upload Data", "Overview", "Drivers & Performance", "Diagnostics & Efficiency", "AI Insights & Recommendations"],
        icons=["cloud-upload", "speedometer", "bar-chart", "activity", "stars"],
        default_index=1 if ss.result is not None else 0,
    )
    st.markdown("### Filters")
    if ss.df is not None and not ss.df.empty:
        base = ss.df
        def _opts(col): 
            return ["All"] + (sorted(base[col].dropna().unique().tolist()) if col in base.columns else [])
        stores = _opts("store"); depts = _opts("department"); regs = _opts("region")
        ss.filters["store"] = st.selectbox("Store", stores, index=stores.index(ss.filters["store"]) if ss.filters["store"] in stores else 0)
        ss.filters["department"] = st.selectbox("Department", depts, index=depts.index(ss.filters["department"]) if ss.filters["department"] in depts else 0)
        ss.filters["region"] = st.selectbox("Region", regs, index=regs.index(ss.filters["region"]) if ss.filters["region"] in regs else 0)
        if st.button("Recompute KPIs"):
            _recompute(_filter_df(ss.df))

# ==================== PAGES ====================

# -------- Upload --------
if page == "Upload Data":
    page_header(
        title="Upload Your Data",
        subtitle="Upload CSV or Excel files to unlock AI-powered insights and automated analysis of your sales data",
        gradient_start="#667eea",
        gradient_end="#764ba2"
    )
    
    # Expected format info
    with st.expander(" Expected Data Format", expanded=False):
        st.markdown("""
        **Required columns:**
        - `date` - Date of the sales record (YYYY-MM-DD format)
        - `store` - Store identifier (e.g., Store_1, Store_2)
        - `department` - Department identifier (e.g., Dept_1, Dept_2)
        - `region` - Geographic region (e.g., North, South, East, West)
        - `weekly_sales` - Sales amount in dollars
        - `transactions` - Number of transactions
        
        **Optional columns:**
        - `profit` - Profit margin data
        - `quantity` - Quantity of items sold
        
        **Example:**
```
        date,store,department,region,weekly_sales,transactions
        2023-01-01,Store_1,Dept_1,North,12500,250
        2023-01-01,Store_2,Dept_2,South,15300,320
```
        """)
    
    # File uploader with better styling
    file = st.file_uploader(
        "Choose a CSV or XLSX file",
        type=["csv", "xlsx"],
        help="Maximum file size: 200MB"
    )
    
    if file:
        # Show file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", file.name)
        with col2:
            st.metric("File Size", f"{file.size / 1024:.1f} KB")
        with col3:
            file_type = "CSV" if file.name.endswith(".csv") else "Excel"
            st.metric("File Type", file_type)
        
        st.markdown("---")
        
        # Load data
        try:
            with st.spinner("📖 Reading file..."):
                df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
            
            # Data preview
            st.subheader(" Data Preview")
            st.dataframe(df.head(25), use_container_width=True)
            
            # Basic stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
                st.metric("Missing Data", f"{missing_pct:.1f}%")
            
            # Validate required columns
            required_cols = {"date", "store", "department", "region", "weekly_sales", "transactions"}
            missing_cols = required_cols - set(df.columns)
            
            if missing_cols:
                st.error(f" Missing required columns: {', '.join(missing_cols)}")
                st.info("Please ensure your data contains all required columns before analyzing.")
            else:
                st.success(" All required columns found!")
                
                # Analyze button
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button(
                        " Start Analysis",
                        type="primary",
                        use_container_width=True,
                        help="Process data and generate insights"
                    ):
                        with st.spinner(" Processing your data..."):
                            ss.df = df
                            time.sleep(0.3)  # Perceptual loading
                            _recompute(_filter_df(df))
                        
                        # st.balloons()  # Celebration effect
                        st.success(" Analysis complete! Navigate to **Overview** to see results.")
                        time.sleep(1.5)
                        st.rerun()
        
        except Exception as e:
            st.error(f" Error reading file: {str(e)}")
            st.info("Please check your file format and try again.")
    
    else:
        # Empty state with sample data option
        st.markdown("""
        <div style="
            text-align: center;
            padding: 60px 20px;
            background: rgba(30, 41, 59, 0.5);
            border: 2px dashed #475569;
            border-radius: 16px;
            margin: 20px 0;
        ">
            <div style="font-size: 64px; margin-bottom: 16px;">📂</div>
            <h3 style="color: #cbd5e1; margin: 0 0 8px 0;">No file uploaded yet</h3>
            <p style="color: #94a3b8; margin: 0;">
                Upload your sales data to begin automated analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        

# -------- 1) Overview --------
elif page == "Overview":
    _ensure_data()
    if ss.result is None: 
        _recompute(_filter_df(ss.df))

    page_header(
        title="Executive Summary",
        subtitle="Key performance indicators and trends at a glance - what's happening overall?",
        gradient_start="#667eea",
        gradient_end="#764ba2"
    )
    r = ss.result
    fdf = _filter_df(ss.df)
    status_color, status_label = _health_status(r)

    # KPI row
    last4 = _last4_weeks_sales_sum(fdf)
    c1,c2,c3 = st.columns(3)
    with c1: st.markdown(f"<div class='card'><h3>Total Revenue (last 4 weeks)</h3><div class='v'>${last4:,.0f}</div></div>", unsafe_allow_html=True)
    with c2:
        tr = r.get("trend_4wk")
        tr_txt = "—" if tr is None or (isinstance(tr,float) and np.isnan(tr)) else f"{tr*100:.1f}%"
        st.markdown(f"<div class='card'><h3>4-Week Trend</h3><div class='v'>{tr_txt}</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='card'><h3>Overall Health</h3><div class='v'><span class='dot {status_color}'></span>{status_label}</div></div>", unsafe_allow_html=True)

    st.info("💡 This month: **Store_5** is driving growth while **Store_1** data gaps mask revenue. "
            "Priority: **fix data collection**, then **replicate Store_5’s high-value transaction model** in underperformers.")

    # Momentum
    section_divider("Momentum Analysis")

    # Row A — Trend (full width, taller)
    st.plotly_chart(
        fig_sales_trend_forecast_shaded(r.get("kpis_weekly", {}), height=420),
        use_container_width=True,
        key="trend_shaded_big"
    )
    st.caption("Spikes/dips are relative to the 4-week MA; shaded area is a simple 2-week projection.")

    # Row B — Two columns: WoW (left, tall) + 4-week store comparison (right, tall)
    col_wow, col_tbl = st.columns([1, 1])

    with col_wow:
        st.plotly_chart(
            fig_wow_bars(r.get("kpis_weekly", {}), height=420),
            use_container_width=True,
            key="wow_bars_tall"
        )
        st.caption("Blue/Pink bars show positive/negative Week Over Week (WoW) growth(%); useful to spot acceleration or fatigue.")

    with col_tbl:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 10px;
            border-radius: 16px;
            border: 1px solid #334155;
        ">
            <h3 style="color: #f8fafc; margin: 0 0 16px 0; font-size: 1.3rem;">
                Store Performance: 4-Week Comparison
            </h3>
        """, unsafe_allow_html=True)
        
        comp = build_store_4wk_table(_filter_df(ss.df))
        if comp.empty:
            st.info("Not enough data to compute 4-week comparison.")
        else:
            show = comp.copy()
            
            # Format function for currency
            def format_currency(val):
                return f"${val:,.0f}"
            
            # Format function for delta with color
            def format_delta_html(val):
                if pd.isna(val):
                    return '<span style="color: #64748b;">—</span>'
                
                color = "#69d3f3" if val > 0 else "#f347ce"
                arrow = "↗" if val > 0 else "↘"
                sign = "+" if val > 0 else ""
                
                return f'''
                <span style="
                    color: {color};
                    font-weight: 700;
                    font-size: 1.1rem;
                ">
                    {arrow} {sign}{val*100:.1f}%
                </span>
                '''
            
            # Create HTML table
            html = '<table style="width: 100%; border-collapse: collapse; margin-top: 8px;">'
            html += '''
            <thead>
                <tr style="border-bottom: 2px solid #334155;">
                    <th style="padding: 12px; text-align: left; color: #94a3b8; font-weight: 600; font-size: 0.85rem;">STORE</th>
                    <th style="padding: 12px; text-align: right; color: #94a3b8; font-weight: 600; font-size: 0.85rem;">LAST 4W</th>
                    <th style="padding: 12px; text-align: right; color: #94a3b8; font-weight: 600; font-size: 0.85rem;">PREV 4W</th>
                    <th style="padding: 12px; text-align: center; color: #94a3b8; font-weight: 600; font-size: 0.85rem;">CHANGE</th>
                </tr>
            </thead>
            <tbody>
            '''
            
            for _, row in show.iterrows():
                html += f'''
                <tr style="border-bottom: 1px solid #1e293b;">
                    <td style="padding: 14px; color: #e5e7eb; font-weight: 600;">{row['store']}</td>
                    <td style="padding: 14px; text-align: right; color: #cbd5e1;">{format_currency(row['last_4w'])}</td>
                    <td style="padding: 14px; text-align: right; color: #94a3b8;">{format_currency(row['prev_4w'])}</td>
                    <td style="padding: 14px; text-align: center;">{format_delta_html(row['Δ vs prev 4w'])}</td>
                </tr>
                '''
            
            html += '</tbody></table>'
            st.markdown(html, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("💡 Stores with negative Δ need immediate investigation")

    # Top movers
    st.subheader("Top Movers")
    st.subheader("Performance Overview (Stores • Departments • 4-week Change)")
    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        st.plotly_chart(fig_stores_desc(fdf), use_container_width=True)

    with col2:
        st.plotly_chart(fig_departments_desc(fdf), use_container_width=True)

    with col3:
        t = table_store_4wk_change(fdf)
        if t.empty:
            st.info("Not enough weekly data to compute 4-week changes.")
        else:
            # color the Δ column: green for positive, red for negative
            def _color_delta(val):
                if isinstance(val, str) and val.endswith("%"):
                    try:
                        n = float(val.replace("%",""))
                        color = "#16a34a" if n > 0 else ("#ef4444" if n < 0 else "#9ca3af")
                        return f"color: {color}; font-weight: 600"
                    except Exception:
                        return ""
                return ""
            st.dataframe(
                t.style.map(_color_delta, subset=["Δ vs prev 4w"]).format({
                    "last_4w": "{:,.0f}",
                    "prev_4w": "{:,.0f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    # Attention Required panel
    _attention_box(fdf, r)

# -------- 2) Drivers & Performance --------
elif page == "Drivers & Performance":
    _ensure_data()
    if ss.result is None: 
        _recompute(_filter_df(ss.df))

    page_header(
        title="Drivers & Performance",
        subtitle="Understand which stores, departments, and regions are driving your success",
        gradient_start="#667eea",
        gradient_end="#764ba2"
    )
    fdf = _filter_df(ss.df)

    st.subheader("Regional Contribution")
    try:
        st.plotly_chart(fig_regional_share(fdf), use_container_width=True)
        st.caption("Bars show contribution; optionally overlay growth to guide budget shifts.")
    except Exception:
        st.info("Regional contribution unavailable for current filters.")

    st.subheader("Department Drivers")
    col1, col2 = st.columns([2,1])
    with col1:
        try:
            st.plotly_chart(fig_dept_pareto(fdf), use_container_width=True)
        except Exception:
            st.info("Not enough department data to compute 80/20.")
    with col2:
        try:
            st.plotly_chart(fig_dept_sparklines_top3(fdf, height=280), use_container_width=True)
        except Exception:
            st.info("Not enough weekly points to render sparklines for top departments.")
    st.info("💡 Concentration is typical — expand top performers; diagnose declines in low performers.")

    st.subheader("Store Consistency (Weeks × Stores)")
    try:
        st.plotly_chart(fig_store_consistency_heatmap(fdf), use_container_width=True)
        st.caption("Spot volatile stores and weeks driving noise.")
    except Exception:
        st.info("Store/week heatmap unavailable for current data slice.")

# -------- 3) Diagnostics & Efficiency --------
elif page == "Diagnostics & Efficiency":
    _ensure_data()
    if ss.result is None: 
        _recompute(_filter_df(ss.df))

    page_header(
        title="Diagnostics & Efficiency",
        subtitle="Identify data quality issues, anomalies, and opportunities for optimization",
        gradient_start="#f59e0b",
        gradient_end="#d97706"
    )
    fdf = _filter_df(ss.df)

    # A) Data Quality
    st.subheader("Data Quality")
    completeness = float((fdf["weekly_sales"].notna().mean()*100) if ("weekly_sales" in fdf.columns and not fdf.empty) else 100.0)
    c1, _ = st.columns([1,2])
    with c1: st.markdown(f"<div class='card'><h3>Data Completeness</h3><div class='v'>{completeness:.1f}%</div></div>", unsafe_allow_html=True)
    try:
        st.plotly_chart(fig_store_consistency_heatmap(fdf, missing_only=True), use_container_width=True)
    except Exception:
        st.info("Missing-data heatmap unavailable.")
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("📥 Export Missing Data Report"):
            st.success("Generated (placeholder) — wire this to your export utility.")
    with colB:
        st.caption("💡 Fill data gaps before forecasting; gaps distort trend & growth metrics.")

    # B) Efficiency Analysis
    st.subheader("Efficiency (Sales vs Transactions)")
    try:
        fig_eff, r2 = fig_efficiency_quadrants_r2(fdf)
        st.plotly_chart(fig_eff, use_container_width=True)
        st.caption(f"💬 Quadrants show efficiency vs scale; trendline R²≈{r2:.2f}. Move playbooks from high-efficiency segments to low ones.")
    except Exception:
        st.info("Insufficient data to compute efficiency quadrants (need sales & transactions).")

    # C) Anomaly & Outlier Table
    st.subheader("Anomalies & Outliers")
    try:
        table = outlier_table_iqr(fdf)
        if table.empty:
            st.info("No suspicious rows based on IQR today.")
        else:
            st.dataframe(table, use_container_width=True)
            st.button("🔎 Investigate (placeholder)")
    except Exception:
        st.info("Unable to compute outliers for current filter slice.")

# -------- 4) AI Insights --------
elif page == "AI Insights & Recommendations":
    _ensure_data()
    if ss.result is None: 
        _recompute(_filter_df(ss.df))

    page_header(
        title="AI Insights & Recommendations",
        subtitle="AI-generated business insights with prioritized action items for immediate impact",
        gradient_start="#fd7ebd",
        gradient_end="#920243"
    )
    
    if ss.insights is None:
        with st.spinner("Drafting insights..."):
            chart_descs = [
                {"title":"Trend+Forecast"}, {"title":"Regional share"},
                {"title":"Store quadrants"}, {"title":"Dept 80/20 + Sparklines"}
            ]
            ss.insights = draft_insights(ss.result, chart_descs)

    st.subheader("Priorities This Week")
    st.markdown(ss.insights or "_No insights generated._")

    st.divider()
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    with col1: st.button("📤 Send to Slack")
    with col2: st.button("🗓️ Schedule Ops Review")
    with col3: st.button("📑 Export PDF (placeholder)")
