# app.py
import numpy as np
import pandas as pd
import streamlit as st
import time
import base64
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

from textwrap import dedent
from analytics_pipeline import load_rules, compute_kpis, validate_dataframe, flag_outliers_iqr
from narrative import draft_page_summary , draft_drivers_summary
import json
import hashlib
import re
from narrative import micro_insight

# Plots you expose in plots.py
from plots import (
    # Overview
    fig_sales_trend_with_stores,
    fig_wow_bars,
    fig_store_benchmark,
    fig_department_benchmark,
    table_store_4wk_change,
    # Drivers & Performance
    fig_regional_donut,
    fig_department_pareto,
    # Diagnostics & Efficiency
    fig_efficiency_quadrants_r2,
    outlier_table_iqr,
)

# ---------- ENHANCED CSS ----------
st.markdown("""
<style>
/* Base styles */
.block-container {
    padding-top: 0.6rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #635BFF 0%, #4f47d6 100%);
    color: white;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 1.4rem;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(99, 91, 255, 0.3);
    font-size: 0.95rem;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #4f47d6 0%, #3d38a8 100%);
    box-shadow: 0 6px 20px rgba(99, 91, 255, 0.5);
    transform: translateY(-2px);
}

/* Primary button variant */
.stButton>button[kind="primary"] {
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
    box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3);
}
.stButton>button[kind="primary"]:hover {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
    box-shadow: 0 6px 20px rgba(34, 197, 94, 0.5);
}

/* Cards */
.card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    border: 1px solid #334155;
    transition: all 0.3s ease;
    min-height: 210px;           
    display: flex;                
    flex-direction: column;     
    justify-content: flex-start;
}
.card:hover {
    border-color: #635BFF;
    box-shadow: 0 8px 32px rgba(99, 91, 255, 0.3);
    transform: translateY(-4px);
}
.card h3 {
    margin: 0 0 12px 0;
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
}
.card .v {
    font-size: 2.2rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1;
}

/* Status dots with animation */
.dot {
    height: 14px;
    width: 14px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 10px;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}
.dot.green {
    background: #22c55e;
    box-shadow: 0 0 16px #22c55e;
}
.dot.yellow {
    background: #f59e0b;
    box-shadow: 0 0 16px #f59e0b;
}
.dot.red {
    background: #ef4444;
    box-shadow: 0 0 16px #ef4444;
}

/* Tables */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
}
.stDataFrame [data-testid="stDataFrameResizable"] {
    border-radius: 12px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
}
[data-testid="stSidebar"] .stSelectbox label {
    color: #cbd5e1;
    font-weight: 500;
}

/* Selectbox styling */
.stSelectbox > div > div {
    background: #1e293b;
    border-color: #334155;
    border-radius: 8px;
    transition: all 0.2s ease;
}
.stSelectbox > div > div:hover {
    border-color: #635BFF;
}

/* Plotly charts */
.js-plotly-plot {
    border-radius: 12px;
    overflow: hidden;
}

/* Caption styling */
.stCaption {
    color: #64748b !important;
    font-size: 0.85rem;
    font-style: italic;
    margin-top: 8px;
}

/* Info/Success/Warning boxes */
.stAlert {
    border-radius: 12px;
    border-left-width: 4px;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(30, 41, 59, 0.5);
    border-radius: 8px;
    font-weight: 600;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(30, 41, 59, 0.3);
    border: 2px dashed #475569;
    border-radius: 12px;
    padding: 20px;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #635BFF;
    background: rgba(99, 91, 255, 0.05);
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    font-weight: 700;
}
[data-testid="stMetricLabel"] {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #635BFF !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #0f172a;
}
::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #475569;
}
            
/* spacing before the page-summary toggle */
.section-spacer { height: 12px; }

/* AI page summary card */
.ai-summary-card{
  background:#0b1220;
  border:1px solid #1f2937;
  border-radius:14px;
  padding:18px 18px 12px 18px;
  box-shadow:0 6px 18px rgba(0,0,0,.25);
  margin-top:10px;
}
            
/* Better list formatting */
.ai-summary-card ol { margin:0 0 8px 18px; padding-left: 0; }
.ai-summary-card ol li { margin:6px 0; color:#cbd5e1; line-height: 1.7; }

/* Prevent markdown rendering issues */
.ai-summary-card p { margin: 8px 0; line-height: 1.7; color: #cbd5e1; }
.ai-summary-card strong { font-weight: 700; color: #e5e7eb; }
.ai-summary-card em { font-style: italic; }
            
/* Uniform subheading style inside AI cards */
.ai-summary-card .ai-subheading{
  margin: 16px 0 8px 0;
  font-size: 1.05rem;
  font-weight: 800;
  color: #e5e7eb;
  letter-spacing: .2px;
}

/* Ensure bullet points have consistent styling */
.ai-summary-card div {
  color: #cbd5e1;
}

/* card header */
.ai-summary-header{
  display:flex; align-items:center; gap:10px;
  margin-bottom:6px;
}
.ai-summary-title{
  margin:0; color:#e5e7eb; font-size:1.15rem; font-weight:700;
}

/* subtle rule */
.ai-hr{ border:0; border-top:1px solid #1f2937; margin:10px 0 14px 0; }

/* bullets look tighter */
.ai-summary-card ul{ margin:0 0 8px 18px; }
.ai-summary-card li{ margin:4px 0; color:#cbd5e1; }

/* action buttons row */
.ai-actions{ margin-top:10px; }
</style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
ss = st.session_state
ss.setdefault("df", None)
ss.setdefault("result", None)
ss.setdefault("insights", None)
ss.setdefault("filters", {"store":"All", "department":"All", "region":"All"})
ss.setdefault("page_summary_cache", {})
rules = load_rules()
st.set_page_config(
    page_title="Data-to-Insight Agent",
    page_icon="images/agent_icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------- REUSABLE HEADER COMPONENT ----------
def page_header(title: str, subtitle: str, gradient_start: str = "#667eea", gradient_end: str = "#764ba2"):
    """Reusable header component for all pages"""
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
        <div style="position: absolute; top: -50px; right: -50px; width: 200px; height: 200px; background: rgba(255, 255, 255, 0.1); border-radius: 50%; filter: blur(40px);"></div>
        <div style="position: absolute; bottom: -30px; left: -30px; width: 150px; height: 150px; background: rgba(255, 255, 255, 0.1); border-radius: 50%; filter: blur(30px);"></div>
        <div style="position: relative; z-index: 1;">
            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                <h1 style="color: #fff; margin: 0; font-size: 2.5rem; font-weight: 800; text-shadow: 0 2px 10px rgba(0,0,0,0.2);">{title}</h1>
            </div>
            <p style="color: rgba(255, 255, 255, 0.95); margin: 0; font-size: 1.15rem; font-weight: 400; line-height: 1.6; max-width: 700px;">{subtitle}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------- HELPER FUNCTIONS ----------
REQUIRED_COLS = {"date", "weekly_sales"}

def _safe_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date column to datetime safely"""
    if "date" in df.columns:
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        return d.dropna(subset=["date"])
    return df

def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar filters to dataframe"""
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
    """Recompute KPIs with loading state"""
    with st.spinner("AI analyzing your data..."):
        ss.result = compute_kpis(df_source, rules)
        ss.insights = None
        ss.ai_cache = {}
        time.sleep(0.3)
    st.success("Analysis complete!")
    time.sleep(0.3)

def _ensure_data():
    """Check if data is loaded"""
    if ss.df is None:
        st.warning("📁 Please upload a dataset first in **Upload Data** page.", icon="⚠️")
        st.stop()

def validate_data_with_pipeline(df: pd.DataFrame, rules: dict) -> dict:
    
    quality_report = {
        "status": "excellent",
        "completeness": 100.0,
        "missing_count": 0,
        "outliers": pd.DataFrame(),
        "issues": [],
        "warnings": [],
        "stats": {},
        "validation_passed": True,
        "pipeline_report": None
    }
    
    # Use analytics_pipeline validation
    try:
        df_validated, pipeline_report = validate_dataframe(df, rules)
        quality_report["pipeline_report"] = pipeline_report
        
        # Check for errors from pipeline
        if pipeline_report.get("errors"):
            quality_report["validation_passed"] = False
            quality_report["issues"].extend(pipeline_report["errors"])
        
        # Add warnings from pipeline
        if pipeline_report.get("warnings"):
            quality_report["warnings"].extend(pipeline_report["warnings"])
        
        # Use validated dataframe for further checks
        df = df_validated
        
    except Exception as e:
        quality_report["issues"].append(f"Validation error: {str(e)}")
        quality_report["validation_passed"] = False
        return quality_report
    
    # Calculate completeness
    if not df.empty:
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        quality_report["completeness"] = ((total_cells - missing_cells) / total_cells * 100)
        quality_report["missing_count"] = int(missing_cells)
    
    # Outlier detection using pipeline function
    if not df.empty and "weekly_sales" in df.columns:
        try:
            outlier_config = rules.get("outliers", {})
            df_with_outliers = flag_outliers_iqr(
                df,
                col="weekly_sales",
                group_by=outlier_config.get("group_by", "store"),
                k=outlier_config.get("k", 1.5)
            )
            
            if "is_outlier" in df_with_outliers.columns:
                outlier_df = df_with_outliers[df_with_outliers["is_outlier"]]
                if not outlier_df.empty:
                    # Select relevant columns for display
                    display_cols = ["date", "store", "weekly_sales"]
                    display_cols = [c for c in display_cols if c in outlier_df.columns]
                    quality_report["outliers"] = outlier_df[display_cols].head(20)
                    
                    outlier_count = len(outlier_df)
                    if outlier_count > 5:
                        quality_report["warnings"].append(
                            f"🔍 {outlier_count} outliers detected (review for data quality issues)"
                        )
        except Exception as e:
            quality_report["warnings"].append(f"Outlier detection skipped: {str(e)}")
    
    # Business validation from rules
    validation_rules = rules.get("cleaning", {}).get("validation", {})
    
    # Check minimum stores
    if "store" in df.columns:
        store_count = df["store"].nunique()
        min_stores = validation_rules.get("min_stores", 2)
        quality_report["stats"]["store_count"] = store_count
        
        if store_count < min_stores:
            quality_report["issues"].append(
                f"⚠️ Only {store_count} stores (minimum {min_stores} required)"
            )
            quality_report["validation_passed"] = False
    
    # Check minimum weeks
    if "date" in df.columns:
        try:
            dates = pd.to_datetime(df["date"])
            week_count = dates.dt.to_period('W').nunique()
            min_weeks = validation_rules.get("min_weeks", 4)
            quality_report["stats"]["weeks_of_data"] = week_count
            
            if week_count < min_weeks:
                quality_report["issues"].append(
                    f"⚠️ Only {week_count} weeks (minimum {min_weeks} required for trends)"
                )
                quality_report["validation_passed"] = False
        except:
            pass
    
    # Transaction validation
    if "transactions" in df.columns and "weekly_sales" in df.columns:
        # Sales with zero transactions
        zero_txn_with_sales = ((df["transactions"] == 0) & (df["weekly_sales"] > 0)).sum()
        if zero_txn_with_sales > 0:
            quality_report["warnings"].append(
                f"💡 {zero_txn_with_sales} records have sales but zero transactions"
            )
        
        # Unusually high average transaction
        avg_txn = df["weekly_sales"] / df["transactions"].replace(0, np.nan)
        max_avg = validation_rules.get("max_avg_transaction", 50000)
        high_avg_count = (avg_txn > max_avg).sum()
        if high_avg_count > 0:
            quality_report["warnings"].append(
                f"💡 {high_avg_count} records have avg transaction > ${max_avg:,}"
            )
    
    # Basic stats
    quality_report["stats"]["total_rows"] = len(df)
    quality_report["stats"]["total_columns"] = len(df.columns)
    
    if "weekly_sales" in df.columns:
        quality_report["stats"]["total_revenue"] = float(df["weekly_sales"].sum())
        quality_report["stats"]["avg_weekly_sales"] = float(df["weekly_sales"].mean())
    
    if "date" in df.columns:
        try:
            dates = pd.to_datetime(df["date"])
            quality_report["stats"]["date_range_start"] = dates.min().strftime("%Y-%m-%d")
            quality_report["stats"]["date_range_end"] = dates.max().strftime("%Y-%m-%d")
        except:
            pass
    
    # Determine overall status
    completeness = quality_report["completeness"]
    issues_count = len(quality_report["issues"])
    warnings_count = len(quality_report["warnings"])
    
    if completeness >= 95 and issues_count == 0 and warnings_count <= 2:
        quality_report["status"] = "excellent"
    elif completeness >= 85 and issues_count == 0:
        quality_report["status"] = "good"
    else:
        quality_report["status"] = "needs_attention"
    
    return quality_report

def _health_status(result_dict: dict) -> tuple[str, str]:
    """Determine health status from trend"""
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
    """Calculate last 4 weeks sales sum"""
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
    """Build 4-week comparison table for stores"""
    need = {"date","store","weekly_sales"}
    if df is None or df.empty or need - set(df.columns):
        return pd.DataFrame(columns=["store","last_4w","prev_4w","Δ vs prev 4w"])

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date","store","weekly_sales"])

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

def insight_box(message: str, icon: str = "💡", color: str = "#635BFF"):
    """Create a styled insight box"""
    st.markdown(f"""
    <div style="
        background: rgba(99, 91, 255, 0.08);
        border-left: 4px solid {color};
        padding: 16px 20px;
        border-radius: 12px;
        margin: 20px 0;
    ">
        <span style="color: #a5b4fc; font-weight: 600; font-size: 1.05rem;">{icon} Key Insight:</span>
        <span style="color: #cbd5e1; margin-left: 8px; line-height: 1.7;">
            {message}
        </span>
    </div>
    """, unsafe_allow_html=True)

def _hash_ctx(ctx: dict) -> str:
    """Create cache key from context"""
    return hashlib.md5(json.dumps(ctx, sort_keys=True, default=str).encode()).hexdigest()

def _clean_markdown_for_display(md: str) -> str:
    """
    Clean markdown text and convert to HTML for display inside HTML containers.
    ENHANCED: Properly converts markdown formatting to HTML tags.
    """
    if not md:
        return ""
    
    # First, do the existing text cleaning
    # CRITICAL: Handle italic markers causing concatenation
    # Pattern: *text*nextword → *text* nextword
    md = re.sub(r'\*([^*]+?)\*([a-zA-Z0-9])', r'*\1* \2', md)
    
    # Fix specific word concatenation patterns
    md = re.sub(r'underperformat', 'underperforms at', md)
    md = re.sub(r'dominat(\s|$)', r'dominates\1', md)
    
    # Fix bullet (•) spacing
    md = re.sub(r'([KM])•', r'\1 • ', md)      # After numbers: 570K•
    md = re.sub(r'•([A-Z])', r'• \1', md)      # Before capitals: •Dept
    
    # CRITICAL: Add $ before numbers missing it
    # Matches: 570K, 426K but not $570K (already has $)
    md = re.sub(r'(?<!\$)(?<!\w)(\d+(?:,\d{3})*[KM])\b', r'$\1', md)
    
    # Fix spacing around "vs"
    md = re.sub(r'([KM\)])\s*vs\s*', r'\1 vs ', md)  # Before vs
    md = re.sub(r'vs\s*([A-Z(])', r'vs \1', md)       # After vs
    
    # Fix spacing around parentheses
    md = re.sub(r'\)([a-zA-Z])', r') \1', md)    # After )
    md = re.sub(r'([a-zA-Z])\(', r'\1 (', md)    # Before (
    
    # Fix em-dash spacing
    md = re.sub(r'([a-zA-Z0-9])—', r'\1 —', md)
    md = re.sub(r'—([a-zA-Z])', r'— \1', md)
    
    # Fix camelCase separation
    md = re.sub(r'([a-z])([A-Z])', r'\1 \2', md)
    
    # Clean up excessive spaces
    md = re.sub(r'  +', ' ', md)
    
    # NOW CONVERT MARKDOWN TO HTML
    # Convert **text** to <strong>text</strong>
    md = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', md)
    
    # Convert *text* to <em>text</em>
    md = re.sub(r'(?<!\*)\*(?!\*)([^*]+?)\*(?!\*)', r'<em>\1</em>', md)
    
    # Convert bullet points to proper list items with consistent styling
    # Split into lines and process each one
    lines = md.split('\n')
    processed_lines = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('• '):
            # Convert bullet points to styled divs that match other sections
            content = line[2:]  # Remove '• '
            processed_lines.append(f'<div style="margin: 6px 0; color: #cbd5e1; line-height: 1.7;">• {content}</div>')
        elif line.startswith('<strong>') and line.endswith('</strong>'):
            # Section headers - add proper styling
            header_text = line[8:-9]  # Remove <strong> tags
            processed_lines.append(f'<div class="ai-subheading">{header_text}</div>')
        elif line:
            # Regular paragraphs
            processed_lines.append(f'<p style="margin: 8px 0; line-height: 1.7; color: #cbd5e1;">{line}</p>')
        else:
            # Empty lines for spacing
            processed_lines.append('<br>')
    
    return '\n'.join(processed_lines)

def ai_note(title: str, ctx: dict, icon: str = "💡"):
    """Render a tiny AI note below a chart with caching"""
    ss.setdefault("ai_cache", {})
    
    cache_key = f"{title}:{_hash_ctx(ctx)}"
    
    if cache_key not in ss.ai_cache:
        with st.spinner(f" Analyzing {title}..."):
            try:
                ss.ai_cache[cache_key] = micro_insight(ctx, title)
            except Exception as e:
                ss.ai_cache[cache_key] = f"- Analysis unavailable: {str(e)}"
    
    # Render with nice styling
    st.markdown(f"""
    <div style="
        background: rgba(99, 91, 255, 0.08);
        border-left: 3px solid #635BFF;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 12px 0 24px 0;
        font-size: 0.9rem;
    ">
        <div style="color: #a5b4fc; font-weight: 600; margin-bottom: 6px;">
            {icon} AI Insight
        </div>
        <div style="color: #cbd5e1; line-height: 1.6;">
            {ss.ai_cache[cache_key]}
        </div>
    </div>
    """, unsafe_allow_html=True)

def _weekly_wow(df: pd.DataFrame) -> pd.Series:
    need = {"date","weekly_sales"}
    if df is None or df.empty or need - set(df.columns): 
        return pd.Series([], dtype=float)
    d = df[["date","weekly_sales"]].dropna()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date")
    w = (d.groupby(pd.Grouper(key="date", freq="W"))["weekly_sales"].sum()
           .rename("weekly_sales_sum").reset_index())
    return w["weekly_sales_sum"].pct_change()  # fraction

def _store_4wk_table(df: pd.DataFrame) -> pd.DataFrame:
    # your existing build_store_4wk_table, or this compact variant
    need = {"date","store","weekly_sales"}
    if df is None or df.empty or need - set(df.columns): 
        return pd.DataFrame()
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    w = (d.groupby(["store", pd.Grouper(key="date", freq="W")])["weekly_sales"].sum()
           .reset_index())
    # rolling 4 weeks
    agg = (w.sort_values(["store","date"])
             .assign(last_4w=w.groupby("store")["weekly_sales"].transform(lambda s: s.rolling(4).sum()),
                     prev_4w=w.groupby("store")["weekly_sales"].transform(lambda s: s.shift(4).rolling(4).sum()))
             .dropna(subset=["last_4w","prev_4w"]))
    tail = agg.sort_values("date").groupby("store").tail(1)
    tail["delta"] = (tail["last_4w"] - tail["prev_4w"]) / tail["prev_4w"].replace(0, np.nan)
    return tail[["store","last_4w","prev_4w","delta"]].rename(columns={"delta":"Δ vs prev 4w"})

def _build_page_summary_ctx(df: pd.DataFrame, r: dict) -> dict:
    """Aggregate everything the summary needs from the three sections."""
    ctx = {}

    # --- Momentum (from kpis_weekly)
    w = pd.DataFrame(r.get("kpis_weekly", {})).copy()
    if not w.empty and "date" in w and "weekly_sales_sum" in w:
        w["date"] = pd.to_datetime(w["date"], errors="coerce")
        w = w.dropna(subset=["date"]).sort_values("date")
        y = w["weekly_sales_sum"]
        ma4 = y.rolling(4, min_periods=1).mean()
        wow = y.pct_change() * 100
        ctx["momentum"] = {
            "current_sales": float(y.iloc[-1]) if len(y)>0 else None,
            "ma4": float(ma4.iloc[-1]) if len(ma4)>0 else None,
        }
        ctx["trend_4wk_pct"] = float(((ma4.iloc[-1]-ma4.iloc[-4])/ma4.iloc[-4])*100) if len(ma4) >= 4 and ma4.iloc[-4] else 0.0
        ctx["growth_pulse"] = {
            "last_wow_pct": float(wow.iloc[-1]) if len(wow)>0 else 0.0,
            "avg_wow_pct": float(wow.mean()) if len(wow)>0 else 0.0,
            "pos_weeks": int((wow > 0).sum()),
            "neg_weeks": int((wow < 0).sum()),
        }

    # --- Store pulse (4-week comparison table you already compute)
    try:
        comp = build_store_4wk_table(df)  # your existing helper
    except Exception:
        comp = pd.DataFrame()
    if not comp.empty and "Δ vs prev 4w" in comp:
        ctx["store_pulse"] = {
            "stores_growing": int((comp["Δ vs prev 4w"] > 0).sum()),
            "stores_declining": int((comp["Δ vs prev 4w"] < 0).sum()),
            "best_store": comp.loc[comp["Δ vs prev 4w"].idxmax(), "store"],
            "best_growth_pct": float(comp["Δ vs prev 4w"].max()*100),
            "worst_store": comp.loc[comp["Δ vs prev 4w"].idxmin(), "store"],
            "worst_decline_pct": float(comp["Δ vs prev 4w"].min()*100),
        }

    # --- Benchmarks (stores and departments)
    bench = {"store": {}, "department": {}}
    if not df.empty and {"store","weekly_sales"}.issubset(df.columns):
        g = df.groupby("store", as_index=False)["weekly_sales"].sum()
        avg = g["weekly_sales"].mean()
        top = g.loc[g["weekly_sales"].idxmax(), "store"]
        bot = g.loc[g["weekly_sales"].idxmin(), "store"]
        bench["store"] = {
            "avg_sales": float(avg),
            "top": top,
            "bottom": bot,
            "below_avg_count": int((g["weekly_sales"] < avg).sum()),
            "bottom_gap_pct": float((avg - g.loc[g["store"].eq(bot), "weekly_sales"].iloc[0]) / avg * 100) if avg else 0.0,
        }
    if not df.empty and {"department","weekly_sales"}.issubset(df.columns):
        g = df.groupby("department", as_index=False)["weekly_sales"].sum()
        avg = g["weekly_sales"].mean()
        top = g.loc[g["weekly_sales"].idxmax(), "department"]
        bot = g.loc[g["weekly_sales"].idxmin(), "department"]
        bench["department"] = {
            "avg_sales": float(avg),
            "top": top,
            "bottom": bot,
            "below_avg_count": int((g["weekly_sales"] < avg).sum()),
            "bottom_gap_pct": float((avg - g.loc[g["department"].eq(bot), "weekly_sales"].iloc[0]) / avg * 100) if avg else 0.0,
        }
    ctx["benchmarks"] = bench
    return ctx

def _val_sales(v) -> float:
    """Return a numeric sales value for dict/number/anything."""
    if isinstance(v, dict):
        return float(v.get("sales", 0) or 0)
    if isinstance(v, (int, float, np.number)):
        return float(v)
    return 0.0

def _top_bottom_from_any(obj, df: pd.DataFrame, level: str):
    """
    Extract (top_name, top_val, bottom_name, bottom_val) from different shapes.
    Falls back to computing from df if needed.
    """
    # dict-like: {"Store_5": {"sales": ...}, ...} or {"Store_5": 123, ...}
    if isinstance(obj, dict) and obj:
        rows = []
        for k, v in obj.items():
            s = _val_sales(v)
            if np.isfinite(s):
                rows.append((k, s))
        if rows:
            rows.sort(key=lambda x: x[1])
            bottom_name, bottom_val = rows[0]
            top_name, top_val = rows[-1]
            return top_name, top_val, bottom_name, bottom_val

    # list of dicts: [{"store":"Store_5","sales":...}, ...]
    if isinstance(obj, list) and obj:
        rows = []
        for it in obj:
            if isinstance(it, dict):
                name = it.get(level) or it.get("name") or it.get("label")
                s = _val_sales(it.get("sales"))
                if name is not None and np.isfinite(s):
                    rows.append((name, s))
        if rows:
            rows.sort(key=lambda x: x[1])
            bottom_name, bottom_val = rows[0]
            top_name, top_val = rows[-1]
            return top_name, top_val, bottom_name, bottom_val

    # fallback: compute from df
    if isinstance(df, pd.DataFrame) and not df.empty and level in df.columns and "weekly_sales" in df.columns:
        s = df.groupby(level)["weekly_sales"].sum()
        if not s.empty:
            return s.idxmax(), float(s.max()), s.idxmin(), float(s.min())

    return None, 0.0, None, 0.0

# ---------- AI SUMMARY HELPERS ----------
def _style_fuchsia_headers(md: str) -> str:
    """
    Inside bulleted/numbered lines, convert **Header** into a fuchsia span.
    Keeps your bullets intact and only styles bold headers.
    """
    lines = md.split("\n")
    out = []
    for line in lines:
        if line.strip().startswith(("•", "-", "*", "○", "◦")) or re.match(r"^\s*\d+\.", line):
            line = re.sub(
                r"\*\*([^*]+)\*\*",
                r"<span style='color:#f472b6;font-weight:700;'>\1</span>",
                line
            )
        out.append(line)
    return "\n".join(out)

def _normalize_ai_headers(md: str) -> str:
    """
    Make section headers inside AI cards consistent:
    - Converts common variants (e.g., **IMMEDIATE ACTIONS**, **Immediate Actions**)
      into a uniform, styled div.
    - Keeps canonical labels for the exec summary.
    """
    if not md:
        return md

    # Map common variants to canonical text
    replacements = {
        r"^\s*\*\*momentum\*\*\s*$":                   "Momentum",
        r"^\s*\*\*store pulse\*\*\s*$":                "Store Pulse",
        r"^\s*\*\*ranking\s*&\s*benchmarks\*\*\s*$":   "Ranking & Benchmarks",
        r"^\s*\*\*next\s*7\s*days\s*—?\s*do\s*this\*\*\s*$": "Next 7 Days — Do This",

        # Priority/Action blocks from other generators:
        r"^\s*\*\*immediate actions.*\*\*\s*$":        "Immediate Actions",
        r"^\s*\*\*priority actions.*\*\*\s*$":         "Priority Actions",
        r"^\s*\*\*strategic recommendations.*\*\*\s*$":"Strategic Recommendations",
        r"^\s*\*\*strategic overview.*\*\*\s*$":       "Strategic Overview",
    }

    lines = md.split("\n")
    out = []
    for line in lines:
        # If the line is a pure bold header (no bullet prefix), standardize it
        if not line.strip().startswith(("•","-","*","○","◦")) and "**" in line:
            for pat, label in replacements.items():
                if re.match(pat, line, flags=re.IGNORECASE):
                    line = f"<div class='ai-subheading'>{label}</div>"
                    break
        out.append(line)
    return "\n".join(out)

def render_ai_summary_block(title: str, ctx: dict, *,
                            draft_fn,                 
                            show_toggle: bool = True,
                            toggle_key: str = "show_ai_summary"):
    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

    show = True
    if show_toggle:
        show = st.toggle("Show AI Page Summary", value=True, key=toggle_key)

    if not show:
        return

    # Create cache key
    cache_key = f"{toggle_key}:{_hash_ctx(ctx)}"
    
    # Check cache or generate
    if cache_key not in ss.page_summary_cache:
        with st.spinner(" Generating AI summary..."):
            try:
                raw_md = draft_fn(ctx) or ""
                cleaned_md = _clean_markdown_for_display(raw_md)
                styled_md = _style_fuchsia_headers(cleaned_md)
                final_md   = _normalize_ai_headers(styled_md)
                ss.page_summary_cache[cache_key] = styled_md
            except Exception as e:
                ss.page_summary_cache[cache_key] = f" Unable to generate summary: {str(e)}"
    
    md = ss.page_summary_cache[cache_key]

    section_divider(title)

    st.markdown(f"""
    <div class='ai-summary-card'>
        {md}
    </div>
    """, unsafe_allow_html=True)

    # st.markdown("<hr style='border:0;border-top:1px solid #334155;margin:18px 0 14px;'>",
    #             unsafe_allow_html=True)
    # c1, c2, c3 = st.columns(3)
    # with c1: 
    #     if st.button("📤 Notify Stakeholders", use_container_width=True, key=f"{toggle_key}_notify"):
    #         st.success("✅ Notification sent!")
    # with c2: 
    #     if st.button("📑 Export Action List",  use_container_width=True, key=f"{toggle_key}_export"):
    #         st.success("✅ Export complete!")
    # with c3: 
    #     if st.button("🗓️ Schedule Follow-up", use_container_width=True, key=f"{toggle_key}_followup"):
    #         st.success("✅ Meeting scheduled!")


# --- Helpers to build a single cohesive context (Overview + Drivers) ---
def _build_ai_summary_ctx(df: pd.DataFrame, r: dict) -> dict:
    """
    Cohesive context for AI Summary tab combining:
    - Overview momentum & store pulse
    - Drivers & Performance (regional + department highlights)
    """
    # Base overview ctx you already use elsewhere
    base_ctx = _build_page_summary_ctx(df, r)  # Momentum, WoW, Store pulse, Benchmarks:contentReference[oaicite:2]{index=2}

    # Drivers context (regional & departments) pulled from the same 'result' dict
    regional = (r or {}).get("regional", {}) or {}
    departments = (r or {}).get("departments", {}) or {}
    benchmarks = (r or {}).get("benchmarks", {}) or {}

    # Normalize a compact “drivers” section
    drivers_ctx = {
        "regional": {
            "regions": list(regional.keys()),
            "sales": {k: regional[k].get("sales", 0) for k in regional} if regional else {},
            "top": max(regional, key=lambda k: regional[k].get("sales", 0)) if regional else None,
            "bottom": min(regional, key=lambda k: regional[k].get("sales", 0)) if regional else None,
        },
        "departments": {
            "sales": {k: departments[k].get("sales", 0) for k in departments} if departments else {},
            "top": max(departments, key=lambda k: departments[k].get("sales", 0)) if departments else None,
            "bottom": min(departments, key=lambda k: departments[k].get("sales", 0)) if departments else None,
        },
        "benchmarks": benchmarks,
    }

    base_ctx["drivers"] = drivers_ctx
    return base_ctx


def _ai_card(title: str, body_md: str):
    """Render a single AI card using the same class used elsewhere."""
    st.markdown(
        f"<div class='ai-summary-card'>"
        f"  <div class='ai-summary-header'><h3 class='ai-summary-title'>{title}</h3></div>"
        f"  <hr class='ai-hr'/>"
        f"  {_clean_markdown_for_display(body_md)}"  # reuse your text cleaner:contentReference[oaicite:3]{index=3}
        f"</div>",
        unsafe_allow_html=True
    )

# ---------- SIDEBAR NAVIGATION ----------
with st.sidebar:
    # Load icon safely
    def get_base64_image(image_path):
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except FileNotFoundError:
            st.warning(f"⚠️ Icon file not found: {image_path}")
            return None

    icon_base64 = get_base64_image("images/agent_icon.png")

    # Sidebar Header
    if icon_base64:
        st.markdown(
            f"""
            <div style="
                display:flex;
                justify-content:center;
                align-items:center;
                margin-bottom:10px;
                margin-top:4px;
            ">
                <img src="data:image/png;base64,{icon_base64}" width="80">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="
                display:flex;
                justify-content:center;
                align-items:center;
                margin-bottom:10px;
                margin-top:4px;
                font-size:42px;
            ">
                📊
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("""
    <div style="text-align: center; padding: 20px 0 10px 0;">
        <h1 style="color: #f8fafc; margin: 0; font-size: 1.6rem; font-weight: 800;">
            AI Data-to-Insights Agent
        </h1>
        <p style="color: #64748b; margin: 4px 0 0 0; font-size: 0.85rem;">
            Sales Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    page = option_menu(
        "Navigation",
        ["Upload Data", "Overview", "Drivers & Performance", "AI Insights & Recommendations"],
        icons=["cloud-upload-fill", "speedometer2", "bar-chart-line-fill", "activity", "stars"],
        default_index=1 if ss.result is not None else 0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#94a3b8", "font-size": "18px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "4px 0",
                "padding": "12px 16px",
                "border-radius": "8px",
                "color": "#cbd5e1",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, #635BFF 0%, #4f47d6 100%)",
                "color": "white",
                "font-weight": "600",
            },
        }
    )
    
    st.markdown("---")
    
    # Filters section
    st.markdown("### Filters")
    if ss.df is not None and not ss.df.empty:
        base = ss.df
        def _opts(col): 
            return ["All"] + (sorted(base[col].dropna().unique().tolist()) if col in base.columns else [])
        
        stores = _opts("store")
        depts = _opts("department")
        regs = _opts("region")
        
        ss.filters["store"] = st.selectbox(
            "Store",
            stores,
            index=stores.index(ss.filters["store"]) if ss.filters["store"] in stores else 0
        )
        ss.filters["department"] = st.selectbox(
            "Department",
            depts,
            index=depts.index(ss.filters["department"]) if ss.filters["department"] in depts else 0
        )
        ss.filters["region"] = st.selectbox(
            "Region",
            regs,
            index=regs.index(ss.filters["region"]) if ss.filters["region"] in regs else 0
        )
        
        if st.button("Recompute KPIs", use_container_width=True):
            _recompute(_filter_df(ss.df))
            st.rerun()
    else:
        st.info("No data loaded yet")
    
    st.markdown("---")
    st.markdown("### AI Features")
    ss.setdefault("ai_insights_enabled", True)
    ss.ai_insights_enabled = st.toggle(
        "Show AI insights",
        value=ss.ai_insights_enabled,
        help="Display AI-generated insights below charts"
    )

    # Cache management
    if st.button("Clear AI Cache", use_container_width=True, help="Clear cached AI summaries"):
        ss.ai_cache = {}
        ss.page_summary_cache = {}
        st.success("Cache cleared!")
        time.sleep(0.5)
        st.rerun()
        
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; color: #64748b; font-size: 0.75rem;">
        <p style="margin: 0;">Powered by ChatGPT</p>
        <p style="margin: 4px 0 0 0;">v1.0.0</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== PAGES ====================

# -------- UPLOAD DATA --------
if page == "Upload Data":
    page_header(
        title="Upload Your Data",
        subtitle="Upload CSV or Excel files with automated quality validation and AI-powered analysis",
        gradient_start="#667eea",
        gradient_end="#764ba2"
    )
    
    # Expected format info
    with st.expander(" Expected Data Format & Requirements", expanded=False):
        required_cols = rules.get("dataset", {}).get("required_columns", [])
        st.markdown(f"""
        **Required columns:** {', '.join(f'`{c}`' for c in required_cols)}
        
        **Data requirements:**
        - Minimum 2 stores for comparative analysis
        - Minimum 4 weeks of historical data
        - Dates between 2020-01-01 and today
        - No future dates allowed
        
        **Automatic validation:**
        - Data type checking and conversion
        - Missing value detection
        - Outlier identification (IQR method)
        - Business rule verification
        """)
    
    # File uploader
    file = st.file_uploader(
        "Choose a CSV or XLSX file",
        type=["csv", "xlsx"],
        help="Maximum file size: 200MB"
    )
    
    if file:
        # File info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(" File Name", file.name)
        with col2:
            st.metric(" Size", f"{file.size / 1024:.1f} KB")
        with col3:
            st.metric(" Type", "CSV" if file.name.endswith(".csv") else "Excel")
        
        st.markdown("---")
        
        # Load and validate
        try:
            with st.spinner(" Loading and validating your data..."):
                df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
                time.sleep(0.2)
                
                # Run validation using pipeline
                quality = validate_data_with_pipeline(df, rules)
            
            # ═══════════════════════════════════════════════════════════
            # QUALITY DASHBOARD
            # ═══════════════════════════════════════════════════════════
            
            st.markdown("###  Data Quality Dashboard")
            
            # Quality cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                completeness = quality["completeness"]
                st.metric(
                    "Completeness",
                    f"{completeness:.1f}%",
                    delta="Excellent" if completeness >= 95 else "Good" if completeness >= 85 else "Check",
                    delta_color="normal" if completeness >= 85 else "inverse"
                )
            
            with col2:
                missing = quality["missing_count"]
                st.metric(
                    "Missing Values",
                    f"{missing:,}",
                    delta="Good" if missing < 50 else "Review",
                    delta_color="normal" if missing < 50 else "inverse"
                )
            
            with col3:
                outlier_count = len(quality["outliers"]) if not quality["outliers"].empty else 0
                st.metric(
                    "Outliers",
                    f"{outlier_count}",
                    delta="Normal" if outlier_count < 10 else "Check",
                    delta_color="normal" if outlier_count < 10 else "inverse"
                )
            
            with col4:
                status = quality["status"]
                status_map = {
                    "excellent": {"color": "#22c55e", "label": "✅ Excellent"},
                    "good": {"color": "#f59e0b", "label": "⚠️ Good"},
                    "needs_attention": {"color": "#ef4444", "label": "❌ Review"}
                }
                s = status_map.get(status, {"color": "#94a3b8", "label": status})
                
                st.markdown(f"""
                <div style="
                    background: {s['color']}22;
                    border: 2px solid {s['color']};
                    border-radius: 10px;
                    padding: 12px;
                    text-align: center;
                ">
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 4px;">Status</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: {s['color']};">{s['label']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Issues and warnings
            if quality["issues"] or quality["warnings"]:
                st.markdown("####  Validation Report")
                
                if quality["issues"]:
                    st.error("**Critical Issues** (must be fixed):")
                    for issue in quality["issues"]:
                        st.markdown(f"• {issue}")
                
                if quality["warnings"]:
                    st.warning("**Warnings** (review recommended):")
                    for warning in quality["warnings"]:
                        st.markdown(f"• {warning}")
            else:
                st.success(" **No issues detected!** Your data passed all quality checks.")
            
            # Expandable details
            col1, col2 = st.columns(2)
            
            with col1:
                if not quality["outliers"].empty:
                    with st.expander(f" View {len(quality['outliers'])} Outliers", expanded=False):
                        st.dataframe(
                            quality["outliers"].style.format(
                                {"weekly_sales": "${:,.0f}"} if "weekly_sales" in quality["outliers"].columns else {}
                            ),
                            use_container_width=True,
                            hide_index=True
                        )
                        st.caption(f"💡 Detection: {rules.get('outliers', {}).get('method', 'IQR').upper()} method with k={rules.get('outliers', {}).get('k', 1.5)}")
            
            with col2:
                if quality["missing_count"] > 0:
                    with st.expander(f" Missing Values Detail", expanded=False):
                        missing_by_col = df.isnull().sum()
                        missing_by_col = missing_by_col[missing_by_col > 0]
                        
                        if not missing_by_col.empty:
                            missing_df = pd.DataFrame({
                                "Column": missing_by_col.index,
                                "Count": missing_by_col.values,
                                "%": (missing_by_col.values / len(df) * 100).round(2)
                            })
                            st.dataframe(missing_df, use_container_width=True, hide_index=True)
                            
                            # Show policy from rules
                            null_policy = rules.get("cleaning", {}).get("null_policy", {})
                            if null_policy:
                                st.caption("**Handling Policy:**")
                                for col, policy in null_policy.items():
                                    if col in missing_by_col.index:
                                        st.caption(f"• `{col}`: {policy.replace('_', ' ')}")
            
            st.markdown("---")
            
            # Data preview
            st.markdown("###  Data Preview")
            st.dataframe(df.head(25), use_container_width=True, height=400)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(" Records", f"{quality['stats']['total_rows']:,}")
            
            with col2:
                st.metric(" Columns", quality['stats']['total_columns'])
            
            with col3:
                stores = quality['stats'].get('store_count', 'N/A')
                st.metric(" Stores", stores)
            
            with col4:
                if "total_revenue" in quality['stats']:
                    revenue = quality['stats']['total_revenue']
                    if revenue >= 1_000_000:
                        st.metric(" Revenue", f"${revenue/1_000_000:.1f}M")
                    else:
                        st.metric(" Revenue", f"${revenue/1000:.0f}K")
            
            # Show date range if available
            if "date_range_start" in quality['stats']:
                st.caption(f" Date Range: {quality['stats']['date_range_start']} to {quality['stats']['date_range_end']}")
            
            st.markdown("---")
            
            # Analyze button
            if quality["validation_passed"]:
                st.success(" **Ready to analyze!** All validation checks passed.")
                
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button(" Start AI Analysis", type="primary", use_container_width=True):
                        with st.spinner(" AI analyzing your data..."):
                            ss.df = df
                            time.sleep(0.5)
                            _recompute(_filter_df(df))
                        
                        st.success(" **Analysis complete!** Navigate to **Overview** for insights.")
                        time.sleep(1.5)
                        st.rerun()
            else:
                st.error(" **Cannot analyze.** Fix critical issues above first.")
                
                with st.expander("💡 How to Fix"):
                    st.markdown("""
                    **Common Solutions:**
                    1. **Missing columns**: Ensure your file has all required columns
                    2. **Invalid dates**: Use YYYY-MM-DD format
                    3. **Insufficient data**: Need minimum 4 weeks and 2 stores
                    4. **Missing critical values**: Fill or remove rows
                    
                    **Need help?** Check "Expected Data Format" above.
                    """)
        
        except Exception as e:
            st.error(f" **Error:** {str(e)}")
            with st.expander(" Details"):
                st.code(str(e))
    
    else:
        # Empty state
        st.markdown("""
        <div style="
            text-align: center;
            padding: 80px 20px;
            background: rgba(30, 41, 59, 0.3);
            border: 2px dashed #475569;
            border-radius: 16px;
            margin: 40px 0;
        ">
            <div style="font-size: 80px; margin-bottom: 20px;">📂</div>
            <h3 style="color: #cbd5e1; margin: 0 0 12px 0;">No file uploaded</h3>
            <p style="color: #94a3b8; margin: 0 0 20px 0; font-size: 1.05rem;">
                Drag and drop or click to browse
            </p>
            <p style="color: #64748b; margin: 0; font-size: 0.9rem;">
                 Auto-validation •  Outlier detection •  Quality dashboard
            </p>
        </div>
        """, unsafe_allow_html=True)



# -------- OVERVIEW --------
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
    health_status = status_label.split("(")[0].strip()

    # KPI Cards
    last4 = _last4_weeks_sales_sum(fdf)
    tr = r.get("trend_4wk")

    # Calculate total transactions for last 4 weeks
    if "transactions" in fdf.columns and not fdf.empty:
        fdf_dated = _safe_to_datetime(fdf)
        if not fdf_dated.empty:
            # Get last 4 weeks of transactions
            weekly_txn = (
                fdf_dated.set_index("date")
                .resample("W")["transactions"]
                .sum()
                .tail(4)
            )
            total_transactions = int(weekly_txn.sum()) if not weekly_txn.empty else 0
        else:
            total_transactions = 0
    else:
        total_transactions = 0

    c1, c2, c3 = st.columns(3)
    
    with c1:
        avg_revenue_per_week = last4 / 4  # Calculate weekly average

        st.markdown(f"""
        <div class='card'>
            <h3>Total Revenue (Last 4-Week)</h3>
            <div class='v' style='margin: 12px 0;'>${last4:,.0f}</div>
            <div style='color: #94a3b8; font-size: 0.85rem; margin-top: 14px;'>
                Avg: ${avg_revenue_per_week:,.0f} per week
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        # Calculate average transactions per week
        avg_txn_per_week = total_transactions / 4 if total_transactions > 0 else 0
        
        st.markdown(f"""
        <div class='card'>
            <h3>Total Transactions (Last 4-Week)</h3>
            <div class='v'style='margin: 12px 0;'>{total_transactions:,}</div>
            <div style='color: #94a3b8; font-size: 0.85rem; margin-top: 14px;'>
                Avg: {avg_txn_per_week:,.0f} per week
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        st.markdown(f"""
        <div class='card'>
            <h3>Overall Health</h3>
            <div class='v'style='margin: 12px 0;'>
                <span class='dot {status_color}'></span>{status_label}
            </div>
            <div style='color: #94a3b8; font-size: 0.85rem; margin-top: 14px;'>
                Last 4 weeks vs Prior 4 weeks
            </div>
        </div>
        """, unsafe_allow_html=True)

    # === AI-POWERED EXECUTIVE INSIGHT ===
    if ss.get("ai_insights_enabled", True):
        # Build context
        executive_ctx = {
            "total_revenue": float(last4),
            "trend_4wk_pct": float(tr * 100) if tr is not None and not np.isnan(tr) else 0.0,
            "total_transactions": total_transactions,
            "avg_transaction_value": float(last4 / total_transactions) if total_transactions > 0 else 0.0,
            "health_status": health_status,
        }
        
        # Add store performance
        stores = r.get("stores", {})
        if stores:
            executive_ctx.update({
                "top_store": stores.get("top_store", "Unknown"),
                "top_store_sales": float(stores.get("top_store_sales", 0)),
                "bottom_store": stores.get("bottom_store", "Unknown"),
                "bottom_store_sales": float(stores.get("bottom_store_sales", 0)),
                "store_count": stores.get("count", 0),
            })
        
        # Generate AI insight
        ai_note("Executive Summary Insight", executive_ctx, icon="💡")
    

    # Momentum section
    section_divider("Momentum Analysis")

    # Sales trend
    # Add a toggle checkbox
    show_stores = st.checkbox("Show store breakdown", value=False)

    st.plotly_chart(
        fig_sales_trend_with_stores(
            r.get("kpis_weekly", {}),
            df_source=fdf,
            show_stores=show_stores,
            height=420
        ),
        use_container_width=True
    )
    st.caption("💡 Spikes/dips relative to 4-week Moving Average; shaded area shows 2-Week Forecast")

    if ss.get("ai_insights_enabled", True):
        weekly_data = pd.DataFrame(r.get("kpis_weekly", {}))
        
        if not weekly_data.empty and "weekly_sales_sum" in weekly_data.columns:
            y_col = "weekly_sales_sum"
            
            # Calculate meaningful metrics
            current = float(weekly_data[y_col].iloc[-1])
            previous = float(weekly_data[y_col].iloc[-2]) if len(weekly_data) > 1 else current
            four_weeks_ago = float(weekly_data[y_col].iloc[-5]) if len(weekly_data) > 4 else current
            
            # Calculate velocity (is decline accelerating or slowing?)
            recent_change = current - previous
            historical_change = previous - four_weeks_ago if len(weekly_data) > 4 else 0
            velocity = "accelerating" if abs(recent_change) > abs(historical_change) else "slowing"
            
            # Enhanced context
            ctx = {
                "current_value": current,
                "previous_value": previous,
                "change_amount": current - previous,
                "change_pct": ((current - previous) / previous * 100) if previous > 0 else 0,
                "four_week_avg": float(weekly_data[y_col].tail(4).mean()),
                "vs_four_week_avg": "above" if current > weekly_data[y_col].tail(4).mean() else "below",
                "volatility": float(weekly_data[y_col].std()),
                "velocity": velocity,
                "data_points": len(weekly_data)
            }
            
            ai_note("Sales Trend Analysis", ctx)
    
    # Weekly Trend Analysis
    section_divider("Weekly Momentum & Store Pulse")
    
    # TWO-COLUMN LAYOUT: WoW Growth + Store Comparison Table
    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1.2, 0.8])

    with col_left:
        fig = fig_wow_bars(r.get("kpis_weekly", {}), height=400)
        st.plotly_chart(fig, use_container_width=True, key="wow_chart")   
        st.caption("💡 Blue/pink bars show positive/negative WoW growth—spot acceleration or fatigue")

    with col_right:
        st.markdown("###### 4-Week Store Comparison")
        
        t = table_store_4wk_change(fdf)

        if t.empty:
            st.info("Not enough weekly data to compute 4-week changes.")
        else:
            def _color_delta(val):
                if isinstance(val, str) and val.endswith("%"):
                    try:
                        n = float(val.replace("%",""))
                        color = "#69d3f3" if n > 0 else ("#f347ce" if n < 0 else "#9ca3af")
                        return f"color: {color}; font-weight: 600"
                    except Exception:
                        return ""
                return ""
            
            st.dataframe(
                t.style.map(_color_delta, subset=["Change vs Prev 4 Weeks (%)"]).format({
                    "Last 4 Weeks Sales ($)": "{:,.0f}",
                    "Previous 4 Weeks Sales ($)": "{:,.0f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        
        st.caption("💡 Stores with negative Δ need immediate investigation")

    # === AI Insight directly for Growth Pulse ===
    if ss.get("ai_insights_enabled", True):
        weekly_df = pd.DataFrame(r.get("kpis_weekly", {}))
        wow_ctx = {}
        if not weekly_df.empty and "weekly_sales_sum" in weekly_df.columns:
            ycol = "weekly_sales_sum"
            wow_series = weekly_df[ycol].pct_change() * 100
            wow_ctx = {
                "last_wow_pct": float(wow_series.iloc[-1]) if len(wow_series) > 0 else 0.0,
                "avg_wow_pct": float(wow_series.mean()) if len(wow_series) > 0 else 0.0,
                "pos_weeks": int((wow_series > 0).sum()),
                "neg_weeks": int((wow_series < 0).sum()),
            }

        grow_ctx = {}
        comp = build_store_4wk_table(fdf)
        if not comp.empty:
            grow_ctx = {
                "stores_growing": int((comp["Δ vs prev 4w"] > 0).sum()),
                "stores_declining": int((comp["Δ vs prev 4w"] < 0).sum()),
                "best_store": comp.loc[comp["Δ vs prev 4w"].idxmax(), "store"],
                "best_growth_pct": float(comp["Δ vs prev 4w"].max() * 100),
                "worst_store": comp.loc[comp["Δ vs prev 4w"].idxmin(), "store"],
                "worst_decline_pct": float(comp["Δ vs prev 4w"].min() * 100),
            }

        ctx_growth_pulse = {
            **wow_ctx,
            **grow_ctx,
            "section": "growth_pulse"
        }
        ai_note("Weekly Growth Pulse", ctx_growth_pulse)  # shows a concise, action-oriented bullet
        


    # Ranking & Benchmarks
    section_divider("Ranking & Benchmarks")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_store_benchmark(fdf, height=480), use_container_width=True)

    with col2:
        st.plotly_chart(fig_department_benchmark(fdf, height=480), use_container_width=True)
    st.caption(
        "💡 Blue bars are above average; pink bars are below. Average line shows the benchmark threshold."
    )

    # === AI Insight directly for Performance Benchmarks ===
    if ss.get("ai_insights_enabled", True):
        bench_ctx = {}
        # --- Store-level benchmark context ---
        store_perf = fdf.groupby("store", as_index=False)["weekly_sales"].sum()
        if not store_perf.empty:
            store_avg = store_perf["weekly_sales"].mean()
            above_avg = store_perf.query("weekly_sales > @store_avg")
            below_avg = store_perf.query("weekly_sales <= @store_avg")

            bench_ctx.update({
                "num_stores": len(store_perf),
                "stores_above_avg": len(above_avg),
                "stores_below_avg": len(below_avg),
                "top_store": above_avg.sort_values("weekly_sales", ascending=False)
                                        .iloc[0]["store"] if len(above_avg) else None,
                "top_store_sales": float(above_avg["weekly_sales"].max()) if len(above_avg) else 0.0,
                "bottom_store": below_avg.sort_values("weekly_sales", ascending=True)
                                        .iloc[0]["store"] if len(below_avg) else None,
                "bottom_store_sales": float(below_avg["weekly_sales"].min()) if len(below_avg) else 0.0,
                "store_avg_sales": float(store_avg),
            })

        # --- Department-level benchmark context ---
        dept_perf = fdf.groupby("department", as_index=False)["weekly_sales"].sum()
        if not dept_perf.empty:
            dept_avg = dept_perf["weekly_sales"].mean()
            above_avg_d = dept_perf.query("weekly_sales > @dept_avg")
            below_avg_d = dept_perf.query("weekly_sales <= @dept_avg")

            bench_ctx.update({
                "num_departments": len(dept_perf),
                "depts_above_avg": len(above_avg_d),
                "depts_below_avg": len(below_avg_d),
                "top_dept": above_avg_d.sort_values("weekly_sales", ascending=False)
                                        .iloc[0]["department"] if len(above_avg_d) else None,
                "top_dept_sales": float(above_avg_d["weekly_sales"].max()) if len(above_avg_d) else 0.0,
                "bottom_dept": below_avg_d.sort_values("weekly_sales", ascending=True)
                                        .iloc[0]["department"] if len(below_avg_d) else None,
                "bottom_dept_sales": float(below_avg_d["weekly_sales"].min()) if len(below_avg_d) else 0.0,
                "dept_avg_sales": float(dept_avg),
            })

        ctx_benchmark = {
            **bench_ctx,
            "section": "performance_benchmarks",
        }
        ai_note("Performance Benchmark Insights", ctx_benchmark)

    # --- Page-level AI Summary (Overview) ---
    page_ctx = _build_page_summary_ctx(fdf, r)      # your existing helper
    render_ai_summary_block(
        title="Executive AI Summary",
        ctx=page_ctx,
        draft_fn=draft_page_summary,
        show_toggle=True,
        toggle_key="overview_ai_summary"
    )



# -------- DRIVERS & PERFORMANCE --------
elif page == "Drivers & Performance":
    _ensure_data()
    if ss.result is None: 
        _recompute(_filter_df(ss.df))

    page_header(
        title="Drivers & Performance",
        subtitle="Understand which stores, departments, and regions are driving your success—identify opportunities for growth",
        gradient_start="#667eea",
        gradient_end="#764ba2"
    )
    
    fdf = _filter_df(ss.df)

    rules = load_rules()  # Load configuration rules
    r = compute_kpis(fdf, rules, ss.get("filters"))  # Compute KPIs


    # ═══════════════════════════════════════════════════════════════
    # REGIONAL PERFORMANCE
    # ═══════════════════════════════════════════════════════════════
    
    section_divider("Regional Performance")

    # Explanation box at top
    with st.expander("Understanding Regional Metrics", expanded=False):
        st.markdown("""
        - **Revenue** - Total sales for the region over the last 4 weeks
        - **Trend** - % change compared to the previous 4 weeks (positive = growth, negative = decline)
        - **Transactions** - Number of customer purchases
        - **Avg/txn** - Average sale value per transaction (Revenue ÷ Transactions)
        """)
    

    col1, col2 = st.columns([1.2, 1])

    with col1:
        # Regional Donut Chart
        st.plotly_chart(
            fig_regional_donut(r, height=500),
            use_container_width=True
        )

    with col2:
        st.markdown("""
        <div class='section-spacer'></div>
        <div class='section-spacer'></div>
        <div class='section-spacer'></div>
        """, unsafe_allow_html=True)
        
        regional = r.get("regional", {})

        if regional:
            # Build a tidy table
            rows = []
            for region, m in regional.items():
                rows.append({
                    "Region": region,
                    "Revenue ($)": float(m.get("sales", 0)),
                    "Trend (4w)": float(m.get("trend_4wk", 0)) * 100,  # % value
                    "Transactions": int(m.get("transactions", 0) or 0),
                    "Avg/Txn ($)": float(m.get("avg_transaction_value", 0) or 0.0),
                })

            df_reg = (
                pd.DataFrame(rows)
                .sort_values("Revenue ($)", ascending=False)
                .reset_index(drop=True)
            )

            # Color positive/negative trend (match your blue/pink palette)
            def _trend_color(v):
                try:
                    return (
                        "color: #69d3f3; font-weight:600" if v > 0 else
                        "color: #f347ce; font-weight:600" if v < 0 else
                        "color: #9ca3af;"
                    )
                except Exception:
                    return ""

            st.markdown("###### Regional Metrics")
            st.dataframe(
                df_reg.style
                    .format({
                        "Revenue ($)": "{:,.0f}",
                        "Trend (4w)": "{:+.1f}%",
                        "Transactions": "{:,}",
                        "Avg/Txn ($)": "{:,.2f}",
                    })
                    .applymap(_trend_color, subset=["Trend (4w)"]),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("💡 Trend is 4-week vs prior 4-week. Blue = growth, pink = decline.")
        else:
            st.info("No regional data available for the current filters.")

    # AI INSIGHT: REGIONAL
    if ss.get("ai_insights_enabled", True):
        regional = r.get("regional", {})
        
        regional_ctx = {
            "analysis_type": "regional_performance",
            "regions_count": len(regional),
        }
        
        for region, metrics in regional.items():
            sales = metrics.get("sales", 0)
            trend = metrics.get("trend_4wk", 0)
            txns = metrics.get("transactions", 0)
            atv = metrics.get("avg_transaction_value", 0)
            
            regional_ctx[f"{region}_sales"] = float(sales)
            regional_ctx[f"{region}_trend_pct"] = float(trend * 100)
            regional_ctx[f"{region}_transactions"] = int(txns)
            regional_ctx[f"{region}_avg_transaction"] = float(atv)
        
        # Calculate concentration
        if regional:
            sales_list = [m.get("sales", 0) for m in regional.values()]
            if sales_list:
                regional_ctx["top_region_contribution_pct"] = float(max(sales_list) / sum(sales_list) * 100)
                regional_ctx["regional_balance"] = float(min(sales_list) / max(sales_list) * 100)
        
        ai_note("Regional Performance Analysis", regional_ctx, icon="💡")
    
    
    # ═══════════════════════════════════════════════════════════════
    # DEPARTMENT DRIVERS (PARETO)
    # ═══════════════════════════════════════════════════════════════
    
    section_divider("Department Drivers")
    
    st.markdown("**80/20 Analysis:** Which departments drive most revenue?")
    
    col1, col2 = st.columns([2.5, 1])
    
    with col1:
        departments = r.get("departments", {})
        if departments:
            st.plotly_chart(
                fig_department_pareto(departments, height=500),
                use_container_width=True
            )
        else:
            st.info("No department data available")
    
    with col2:
        st.info('''
        **Pareto Chart Explained**
        
         **Bars** = Department revenue
         **Line** = Cumulative %
        
        **The 80/20 Rule:**
        Typically 80% of revenue comes from 20% of departments.
        
        **Why it matters:**
        - Focus on top performers
        - Allocate resources efficiently
        - Maximize ROI on investments
        - Identify expansion opportunities
        
        **Example:** If 3 departments generate 80% of sales, focus on growing those instead of spreading resources thin.
        ''')
    
    # AI INSIGHT: DEPARTMENTS
    if ss.get("ai_insights_enabled", True):
        departments = r.get("departments", {})
        dept_meta = r.get("departments_meta", {})
        
        dept_ctx = {
            "analysis_type": "department_concentration",
            "total_departments": dept_meta.get("total_count", len(departments)),
            "pareto_80_count": dept_meta.get("pareto_80_count", 0),
        }
        
        if departments:
            # Sort by sales
            sorted_depts = sorted(
                departments.items(),
                key=lambda x: x[1].get("sales", 0),
                reverse=True
            )
            
            total_sales = sum(d[1].get("sales", 0) for d in sorted_depts)
            
            if total_sales > 0:
                dept_ctx["concentration_level"] = (
                    "high" if dept_ctx["pareto_80_count"] <= 3 
                    else "moderate" if dept_ctx["pareto_80_count"] <= 5 
                    else "balanced"
                )
                
                dept_ctx["top_80_pct_of_departments"] = float(
                    dept_ctx["pareto_80_count"] / dept_ctx["total_departments"] * 100
                )
                
                # Top 3 departments details
                for i, (dept, metrics) in enumerate(sorted_depts[:3]):
                    dept_ctx[f"top_{i+1}_department"] = dept
                    dept_ctx[f"top_{i+1}_sales"] = float(metrics.get("sales", 0))
                    dept_ctx[f"top_{i+1}_contribution_pct"] = float(metrics.get("sales", 0) / total_sales * 100)
                    dept_ctx[f"top_{i+1}_trend_pct"] = float(metrics.get("trend_4wk", 0) * 100)
        
        ai_note("Department 80/20 Analysis", dept_ctx, icon="💡")
    
    # ═══════════════════════════════════════════════════════════════
    # EXECUTIVE AI SUMMARY – Drivers & Performance
    # ═══════════════════════════════════════════════════════════════
    if ss.get("ai_insights_enabled", True):
        perf_ctx = {}

        # Regions
        regional = r.get("regional", {})
        trn, trv, brn, brv = _top_bottom_from_any(regional, fdf, level="region")
        if trn or brn:
            perf_ctx.update({
                "top_region_name": trn, "top_region_sales": float(trv),
                "bottom_region_name": brn, "bottom_region_sales": float(brv),
            })

        # Departments
        departments = r.get("departments", {})
        tdn, tdv, bdn, bdv = _top_bottom_from_any(departments, fdf, level="department")
        if tdn or bdn:
            perf_ctx.update({
                "top_dept_name": tdn, "top_dept_sales": float(tdv),
                "bottom_dept_name": bdn, "bottom_dept_sales": float(bdv),
            })

        # Stores
        stores = r.get("stores", {})
        if stores:
            perf_ctx.update({
                "top_store_name": stores.get("top_store", "Unknown"),
                "top_store_sales": float(stores.get("top_store_sales", 0)),
                "bottom_store_name": stores.get("bottom_store", "Unknown"),
                "bottom_store_sales": float(stores.get("bottom_store_sales", 0)),
            })

        # Department concentration (optional 80/20 color)
        dept_vals = []
        if isinstance(departments, dict) and departments:
            dept_vals = [_val_sales(v) for v in departments.values()]
        elif "department" in fdf.columns:
            dept_vals = fdf.groupby("department")["weekly_sales"].sum().tolist()
        dept_vals = [v for v in dept_vals if v > 0]
        if dept_vals:
            perf_ctx["dept_concentration_pct"] = max(dept_vals) / sum(dept_vals) * 100.0

        # Re-use the same renderer used on Overview (no toggle here)
        render_ai_summary_block(
            title="Executive AI Summary",
            ctx=perf_ctx,
            draft_fn=draft_drivers_summary, 
            show_toggle=False,
            toggle_key="drivers_ai_summary"
        )


# -------- AI INSIGHTS & RECOMMENDATIONS --------
elif page == "AI Insights & Recommendations":
    page_header(
        "AI Recommendations & Insights",
        "Comprehensive executive readout with deep insights and contextual AI chat."
    )

    _ensure_data()
    df_filtered = _filter_df(ss.df)
    result = ss.result or {}

    # ---------- Build enhanced context ----------
    from narrative import _build_enhanced_ai_context
    ctx = _build_enhanced_ai_context(df_filtered, result)

    # Initialize AI insights cache
    ss.setdefault("ai_insights_cache", {})

    # ---------- EXECUTIVE AI SUMMARY ----------
    section_divider("Executive AI Summary")
    
    # Cache key for executive summary
    exec_cache_key = f"executive_summary:{_hash_ctx(ctx)}"
    
    if exec_cache_key not in ss.ai_insights_cache:
        with st.spinner("Generating AI executive summary..."):
            try:
                ss.ai_insights_cache[exec_cache_key] = draft_page_summary(ctx)
            except Exception as e:
                ss.ai_insights_cache[exec_cache_key] = f"Executive summary unavailable: {str(e)}"
    
    _ai_card("Strategic Overview", ss.ai_insights_cache[exec_cache_key])


    # ---------- STRATEGIC RECOMMENDATIONS ----------
    section_divider("Strategic Recommendations")
    
    # Cache key for strategic actions
    strategy_cache_key = f"strategic_actions:{_hash_ctx(ctx)}:{_hash_ctx(result)}"
    
    if strategy_cache_key not in ss.ai_insights_cache:
        with st.spinner("Generating strategic recommendations..."):
            try:
                from narrative import _generate_strategic_actions
                ss.ai_insights_cache[strategy_cache_key] = _generate_strategic_actions(ctx, df_filtered, result)
            except Exception as e:
                ss.ai_insights_cache[strategy_cache_key] = f"Strategic recommendations unavailable: {str(e)}"
    
    _ai_card("Priority Actions (Next 30 Days)", ss.ai_insights_cache[strategy_cache_key])

     # Cache management controls
    st.markdown("---")
    st.markdown("**Cache Management**")
    col_cache1, col_cache2, col_cache3 = st.columns(3)
    
    with col_cache1:
        if st.button("Refresh AI Insights", use_container_width=True):
            # Clear AI insights cache
            if "ai_insights_cache" in ss:
                ss.ai_insights_cache.clear()
            st.success("AI insights cache cleared. Page will refresh with new content.")
            st.rerun()
    
    with col_cache2:
        cache_size = len(ss.get("ai_insights_cache", {}))
        st.metric("Cached Insights", cache_size)
    
    with col_cache3:
        total_cache_size = len(ss.get("ai_cache", {})) + len(ss.get("ai_insights_cache", {}))
        st.metric("Total AI Cache", total_cache_size)

        
    # ---------- ADVANCED AI CHAT ----------
    section_divider("AI Analyst Chat")
    
    # Initialize chat history
    if "chat_history" not in ss:
        ss.chat_history = []
    
    # Display chat history
    for i, message in enumerate(ss.chat_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Enhanced chat with conversation memory
    st.caption("Ask follow-up questions about your data. The AI remembers our conversation context.")
    
    # Chat input with suggestions
    example_questions = [
        "What explains the latest sales dip?",
        "Which 2 stores should I prioritize?",
        "What's driving the performance gap?",
        "How can I improve bottom performers?",
        "What are the biggest growth opportunities?"
    ]
    
    with st.expander("Example Questions", expanded=False):
        for q in example_questions:
            if st.button(q, key=f"example_{hash(q)}", use_container_width=True):
                ss.chat_example_question = q

    # Use example question if clicked
    if hasattr(ss, 'chat_example_question'):
        user_q = ss.chat_example_question
        del ss.chat_example_question
    else:
        user_q = st.chat_input("Ask your question here...")

    if user_q:
        # Add user message to history
        ss.chat_history.append({"role": "user", "content": user_q})
        
        with st.chat_message("user"):
            st.markdown(user_q)

        # Enhanced context for chat with accurate data
        enhanced_grounded_ctx = {
            "filters": ss.filters,
            "summary_ctx": ctx,
            "kpis_weekly": result.get("kpis_weekly", {}),
            "regional": result.get("regional", {}),
            "departments": result.get("departments", {}),
            "stores": result.get("stores", {}),
            "benchmarks": result.get("benchmarks", {}),
            "conversation_history": ss.chat_history[-3:]  # Reduced context for more focused responses
        }

        with st.chat_message("assistant"):
            try:
                from openai import OpenAI
                import os, json
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                # Enhanced system prompt for concise, accurate responses
                sys_prompt = """You are a Senior Retail Analytics Consultant providing concise strategic insights.

RESPONSE REQUIREMENTS:
- Maximum 3-4 sentences per response
- Use ONLY numbers from the provided context data
- Be specific and actionable
- Reference exact store names, departments, or regions from the data

DATA HANDLING:
- When specific data IS available: Use exact numbers and entity names
- When specific data is NOT available: Provide general retail best practices and strategic guidance
- Always be helpful - don't just say "data not available"

ANALYSIS APPROACH:
1. Check if relevant data exists in context
2. If YES: Use specific numbers and recommendations
3. If NO: Provide general strategic advice for the situation
4. Always include actionable next steps

NEVER:
- Make up numbers not in the context
- Assume data not provided"""

                # Build conversation with limited history for focus
                messages = [{"role": "system", "content": sys_prompt}]
                
                # Add context with specific data validation
                messages.append({
                    "role": "user", 
                    "content": f"VERIFIED DATA CONTEXT:\n{json.dumps(enhanced_grounded_ctx, indent=2, default=str)}"
                })
                
                # Add recent conversation history (limited)
                for msg in ss.chat_history[-4:-1]:  # Only last 3 exchanges
                    messages.append(msg)
                
                # Add current question
                messages.append({"role": "user", "content": user_q})

                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.2,  # Lower temperature for more consistent, factual responses
                    max_tokens=200,   # Reduced token limit for conciseness
                    presence_penalty=0.1
                )
                
                ai_response = resp.choices[0].message.content
                st.markdown(ai_response)
                
                # Add AI response to history
                ss.chat_history.append({"role": "assistant", "content": ai_response})
                
                # Keep chat history manageable (last 12 messages)
                if len(ss.chat_history) > 12:
                    ss.chat_history = ss.chat_history[-12:]
                    
            except Exception as e:
                error_msg = f"AI response unavailable: {e}"
                st.warning(error_msg)
                ss.chat_history.append({"role": "assistant", "content": error_msg})
    
   
    
    # Chat controls
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat History", use_container_width=True):
            ss.chat_history = []
            st.rerun()
    with col2:
        if st.button("Export Chat Log", use_container_width=True):
            chat_export = "\n\n".join([
                f"**{msg['role'].title()}:** {msg['content']}"
                for msg in ss.chat_history
            ])
            st.download_button(
                "Download Chat Log",
                data=chat_export,
                file_name=f"ai_chat_log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown"
            )



# ==================== END OF APP ====================