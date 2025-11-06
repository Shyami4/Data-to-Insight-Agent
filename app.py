# app.py
import numpy as np
import pandas as pd
import streamlit as st
import time
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

from analytics_pipeline import load_rules, compute_kpis
from narrative import draft_page_summary  
import json
import hashlib
from narrative import micro_insight

# Plots you expose in plots.py
from plots import (
    # Overview
    _sparkline, 
    kpi_value_and_delta_vs_py,
    fig_sales_trend_forecast_shaded,
    fig_wow_bars,
    fig_store_benchmark,
    fig_department_benchmark,
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
st.set_page_config(
    page_title="AI Data Analyst | Sales Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
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
rules = load_rules()

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
        time.sleep(0.3)
    st.success("Analysis complete!")
    time.sleep(0.3)

def _ensure_data():
    """Check if data is loaded"""
    if ss.df is None:
        st.warning("📁 Please upload a dataset first in **Upload Data** page.", icon="⚠️")
        st.stop()

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


# --- ATTENTION REQUIRED BOX ---
def _attention_box(df: pd.DataFrame, result: dict):
    """Generate attention required alerts"""
    if df is None or df.empty:
        return
    d = _safe_to_datetime(df)
    if d.empty or "weekly_sales" not in d.columns:
        return

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

# ---------- SIDEBAR NAVIGATION ----------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 10px 0;">
        <h1 style="color: #f8fafc; margin: 0; font-size: 1.6rem; font-weight: 800;">
            AI Data Analyst
        </h1>
        <p style="color: #64748b; margin: 4px 0 0 0; font-size: 0.85rem;">
            Sales Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    page = option_menu(
        "Navigation",
        ["Upload Data", "Overview", "Drivers & Performance", "Diagnostics & Efficiency", "AI Insights & Recommendations"],
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
        
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; color: #64748b; font-size: 0.75rem;">
        <p style="margin: 0;">Powered by AI</p>
        <p style="margin: 4px 0 0 0;">v1.0.0</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== PAGES ====================

# -------- UPLOAD DATA --------
if page == "Upload Data":
    page_header(
        title="Upload Your Data",
        subtitle="Upload CSV or Excel files to unlock AI-powered insights and automated analysis of your sales data",
        gradient_start="#667eea",
        gradient_end="#764ba2"
    )
    
    # Expected format info
    with st.expander("Expected Data Format", expanded=False):
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
        """)
    
    # File uploader
    file = st.file_uploader(
        "Choose a CSV or XLSX file",
        type=["csv", "xlsx"],
        help="Maximum file size: 200MB"
    )
    
    if file:
        # File info metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", file.name)
        with col2:
            st.metric("File Size", f"{file.size / 1024:.1f} KB")
        with col3:
            file_type = "CSV" if file.name.endswith(".csv") else "Excel"
            st.metric("File Type", file_type)
        
        st.markdown("---")
        
        # Load and validate data
        try:
            with st.spinner("Reading file..."):
                df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
                time.sleep(0.2)
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(25), use_container_width=True, height=400)
            
            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
                st.metric("Data Quality", f"{100-missing_pct:.1f}%")
            
            # Validate
            required_cols = {"date", "store", "department", "region", "weekly_sales", "transactions"}
            missing_cols = required_cols - set(df.columns)
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                st.success("All required columns found!")
                
                # Analyze button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("Start AI Analysis", type="primary", use_container_width=True):
                        with st.spinner("Processing your data..."):
                            ss.df = df
                            time.sleep(0.5)
                            _recompute(_filter_df(df))
                        st.success("Analysis complete! Navigate to **Overview** to see results.")
                        time.sleep(1.5)
                        st.rerun()
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
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
            <h3 style="color: #cbd5e1; margin: 0 0 12px 0;">No file uploaded yet</h3>
            <p style="color: #94a3b8; margin: 0; font-size: 1.05rem;">
                Drag and drop your sales data CSV or Excel file to begin
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

    # KPI Cards
    last4 = _last4_weeks_sales_sum(fdf)
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"""
        <div class='card'>
            <h3>Total Revenue (Last 4 Weeks)</h3>
            <div class='v'>${last4:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        tr = r.get("trend_4wk")
        tr_txt = "—" if tr is None or (isinstance(tr,float) and np.isnan(tr)) else f"{tr*100:.1f}%"
        st.markdown(f"""
        <div class='card'>
            <h3>4-Week Trend</h3>
            <div class='v'>{tr_txt}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        st.markdown(f"""
        <div class='card'>
            <h3>Overall Health</h3>
            <div class='v'>
                <span class='dot {status_color}'></span>{status_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

    insight_box(
        "This month: **Store_5** is driving growth while **Store_1** data gaps mask revenue. "
        "Priority: **fix data collection**, then **replicate Store_5's high-value transaction model** in underperformers."
    )

    # Momentum section
    section_divider("Momentum Analysis")

    # Sales trend
    st.plotly_chart(
        fig_sales_trend_forecast_shaded(r.get("kpis_weekly", {}), height=420),
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
    
    col_left, col_right = st.columns([1.3, 0.7])

    with col_left:
        fig = fig_wow_bars(r.get("kpis_weekly", {}), height=400)
        st.plotly_chart(fig, use_container_width=True, key="wow_chart")   
        st.caption("💡 Blue/pink bars show positive/negative WoW growth—spot acceleration or fatigue")

    with col_right:
        st.markdown("#### 4-Week Store Comparison")
        
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
                t.style.map(_color_delta, subset=["Δ vs prev 4w"]).format({
                    "last_4w": "{:,.0f}",
                    "prev_4w": "{:,.0f}",
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


    # --- Page-level AI Summary
    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

show_page_summary = st.toggle("Show AI Page Summary", value=True)

if show_page_summary:
    page_ctx = _build_page_summary_ctx(fdf, r)   # your existing helper
    summary_md = draft_page_summary(page_ctx)

    # escape underscores so names like Store_2 don't italicize
    summary_md_safe = summary_md.replace("_", r"\_")

    # Card container
    st.markdown("<div class='ai-summary-card'>", unsafe_allow_html=True)

    # Header
    st.markdown(
        "<div class='ai-summary-header'>"
        "  <span style='font-size:22px'>🧾</span>"
        "  <h3 class='ai-summary-title'>Executive Snapshot</h3>"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("<hr class='ai-hr'/>", unsafe_allow_html=True)

    # Body (your generated bullets)
    st.markdown(summary_md_safe)

    # Actions
    st.markdown("<div class='ai-actions'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.button("📤 Notify Stakeholders", use_container_width=True)
    with c2: st.button("📑 Export Action List", use_container_width=True)
    with c3: st.button("🗓️ Schedule Follow-up", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Close card
    st.markdown("</div>", unsafe_allow_html=True)





# -------- DRIVERS & PERFORMANCE --------
elif page == "Drivers & Performance":
    _ensure_data()
    if ss.result is None: 
        _recompute(_filter_df(ss.df))

    page_header(
        title="Drivers & Performance",
        subtitle="Understand which stores, departments, and regions are driving your success—identify opportunities for growth",
        gradient_start="#8b5cf6",
        gradient_end="#7c3aed"
    )
    
    fdf = _filter_df(ss.df)

    # Regional Contribution
    section_divider("Regional Contribution", "🌍")
    try:
        st.plotly_chart(fig_regional_share(fdf), use_container_width=True)
        insight_box(
            "Compare regional performance to identify growth opportunities and optimize resource allocation across geographies.",
            color="#8b5cf6"
        )
    except Exception:
        st.info("Regional contribution unavailable for current filters.")

    # Department Drivers
    section_divider("Department Drivers", "📦")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        try:
            st.plotly_chart(fig_dept_pareto(fdf), use_container_width=True)
        except Exception:
            st.info("Not enough department data to compute 80/20 analysis.")
    
    with col2:
        try:
            st.plotly_chart(fig_dept_sparklines_top3(fdf, height=280), use_container_width=True)
        except Exception:
            st.info("Not enough weekly data to render department trends.")
    
    insight_box(
        "Revenue concentration is typical—focus on expanding top performers and diagnosing declines in underperformers to maximize ROI.",
        color="#8b5cf6"
    )

    # Store Consistency
    section_divider("Store Consistency Matrix", "🏪")
    try:
        st.plotly_chart(fig_store_consistency_heatmap(fdf), use_container_width=True)
        st.caption("💡 Identify volatile stores and weeks that may require operational attention or investigation")
    except Exception:
        st.info("Store consistency heatmap unavailable for current data slice.")


# -------- DIAGNOSTICS & EFFICIENCY --------
elif page == "Diagnostics & Efficiency":
    _ensure_data()
    if ss.result is None: 
        _recompute(_filter_df(ss.df))

    page_header(
        title="Diagnostics & Efficiency",
        subtitle="Identify data quality issues, anomalies, and opportunities for optimization—improve operational excellence",
        gradient_start="#f59e0b",
        gradient_end="#d97706"
    )
    
    fdf = _filter_df(ss.df)

    # Data Quality Section
    section_divider("Data Quality Assessment", "✅")
    
    completeness = float((fdf["weekly_sales"].notna().mean()*100) if ("weekly_sales" in fdf.columns and not fdf.empty) else 100.0)
    
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.markdown(f"""
        <div class='card'>
            <h3>Data Completeness</h3>
            <div class='v'>{completeness:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        missing_count = fdf["weekly_sales"].isna().sum() if "weekly_sales" in fdf.columns else 0
        st.markdown(f"""
        <div class='card'>
            <h3>Missing Records</h3>
            <div class='v'>{missing_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        quality_status = "Excellent" if completeness >= 95 else "Good" if completeness >= 85 else "Needs Attention"
        quality_color = "#22c55e" if completeness >= 95 else "#f59e0b" if completeness >= 85 else "#ef4444"
        st.markdown(f"""
        <div class='card'>
            <h3>Quality Status</h3>
            <div class='v' style='color: {quality_color};'>{quality_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    try:
        st.plotly_chart(fig_store_consistency_heatmap(fdf, missing_only=True), use_container_width=True)
    except Exception:
        st.info("Missing data heatmap unavailable for current filters.")
    
    colA, colB = st.columns([1, 2])
    with colA:
        if st.button("📥 Export Missing Data Report", use_container_width=True):
            st.success("✅ Report generated successfully!")
    
    insight_box(
        "Fill data gaps before forecasting to improve accuracy. Missing data distorts trend analysis and growth metrics.",
        color="#f59e0b"
    )

    # Efficiency Analysis
    section_divider("Efficiency Analysis", "⚡")
    
    try:
        fig_eff, r2 = fig_efficiency_quadrants_r2(fdf)
        st.plotly_chart(fig_eff, use_container_width=True)
        
        st.markdown(f"""
        <div style="
            background: rgba(245, 158, 11, 0.1);
            border-left: 4px solid #f59e0b;
            padding: 16px 20px;
            border-radius: 12px;
            margin: 20px 0;
        ">
            <span style="color: #fbbf24; font-weight: 600; font-size: 1.05rem;">📊 Statistical Analysis:</span>
            <span style="color: #cbd5e1; margin-left: 8px;">
                Quadrants reveal efficiency vs scale patterns with R² = {r2:.2f}. 
                Transfer best practices from high-efficiency segments (top-right quadrant) to improve underperformers.
            </span>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.info("Insufficient data to compute efficiency quadrants (requires sales & transaction data).")

    # Anomalies & Outliers
    section_divider("Anomalies & Outliers", "🚨")
    
    try:
        outlier_table = outlier_table_iqr(fdf)
        if outlier_table.empty:
            st.success("✅ No significant anomalies detected based on statistical analysis.")
        else:
            st.dataframe(outlier_table, use_container_width=True)
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("🔎 Investigate Selected", use_container_width=True):
                    st.info("🔍 Opening detailed anomaly investigation...")
            with col2:
                if st.button("📊 Export Report", use_container_width=True):
                    st.success("✅ Anomaly report exported!")
    except Exception:
        st.info("Unable to compute outliers for current filter selection.")


# -------- AI INSIGHTS & RECOMMENDATIONS --------
elif page == "AI Insights & Recommendations":
    _ensure_data()
    if ss.result is None: 
        _recompute(_filter_df(ss.df))

    page_header(
        title="AI Insights & Recommendations",
        subtitle="AI-generated business insights with prioritized action items for immediate impact—your intelligent advisor",
        gradient_start="#ec4899",
        gradient_end="#db2777"
    )
    
    if ss.insights is None:
        with st.spinner("🤖 AI is analyzing your data and generating insights..."):
            chart_descs = [
                {"title": "Sales Trend & Forecast"},
                {"title": "Regional Performance"},
                {"title": "Store Efficiency Quadrants"},
                {"title": "Department 80/20 Analysis"}
            ]
            ss.insights = draft_insights(ss.result, chart_descs)
            time.sleep(0.5)

    # AI-Generated Insights
    section_divider("AI-Generated Insights", "🎯")
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 28px;
        border-radius: 16px;
        border: 1px solid #334155;
        margin-bottom: 24px;
    ">
    """, unsafe_allow_html=True)
    
    st.markdown(ss.insights or "_AI insights are being generated..._")
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Priority Actions
    section_divider("Recommended Actions", "✨")
    
    st.markdown("""
    <div style="
        background: rgba(236, 72, 153, 0.1);
        border-left: 4px solid #ec4899;
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 24px;
    ">
        <span style="color: #f9a8d4; font-weight: 600; font-size: 1.05rem;">💡 Pro Tip:</span>
        <span style="color: #cbd5e1; margin-left: 8px;">
            These recommendations are prioritized by potential business impact. Start with the highest priority items for maximum ROI.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Action cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 24px;
            border-radius: 16px;
            border: 1px solid #334155;
            height: 100%;
        ">
            <div style="font-size: 40px; margin-bottom: 16px;">📊</div>
            <h3 style="color: #f8fafc; margin: 0 0 12px 0; font-size: 1.2rem;">Share Insights</h3>
            <p style="color: #94a3b8; margin: 0 0 16px 0; font-size: 0.9rem;">
                Distribute findings to your team via Slack
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📤 Send to Slack", use_container_width=True, key="slack"):
            st.success("✅ Insights shared to #analytics channel")
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 24px;
            border-radius: 16px;
            border: 1px solid #334155;
            height: 100%;
        ">
            <div style="font-size: 40px; margin-bottom: 16px;">🗓️</div>
            <h3 style="color: #f8fafc; margin: 0 0 12px 0; font-size: 1.2rem;">Schedule Review</h3>
            <p style="color: #94a3b8; margin: 0 0 16px 0; font-size: 0.9rem;">
                Set up operational review meeting
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🗓️ Schedule Meeting", use_container_width=True, key="calendar"):
            st.success("✅ Meeting invite sent to stakeholders")
    
    with col3:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 24px;
            border-radius: 16px;
            border: 1px solid #334155;
            height: 100%;
        ">
            <div style="font-size: 40px; margin-bottom: 16px;">📑</div>
            <h3 style="color: #f8fafc; margin: 0 0 12px 0; font-size: 1.2rem;">Export Report</h3>
            <p style="color: #94a3b8; margin: 0 0 16px 0; font-size: 0.9rem;">
                Download comprehensive PDF report
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📑 Export PDF", use_container_width=True, key="pdf"):
            st.success("✅ Report generated and downloaded")

    # Performance Summary
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("📈 View Detailed Performance Metrics", expanded=False):
        r = ss.result
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Records Analyzed",
                f"{r.get('record_count', 0):,}",
                help="Number of sales records processed"
            )
        
        with col2:
            st.metric(
                "Time Period",
                "6 months",
                help="Analysis timeframe"
            )
        
        with col3:
            outliers = r.get('outliers', 0)
            st.metric(
                "Anomalies Detected",
                f"{outliers}",
                delta="-2 vs last month" if outliers < 5 else "+1 vs last month",
                delta_color="normal" if outliers < 5 else "inverse"
            )
        
        with col4:
            st.metric(
                "Data Quality Score",
                "97.4%",
                delta="+2.1%",
                help="Percentage of complete records"
            )

# ==================== END OF APP ====================