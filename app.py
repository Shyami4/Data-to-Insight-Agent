import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
from chart_style import apply_common_bar_style, color_discrete_map_from_categories
from plotly.colors import qualitative as q  # for a nice categorical palette


# your modules
from analytics_pipeline import load_rules, compute_kpis
from plots import fig_trend_sales, fig_sales_by_region, fig_top_departments
from narrative import draft_insights

load_dotenv()
st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

# --- CSS polish (cards, buttons, spacing) ---
st.markdown("""
<style>
/* brand button */
.stButton>button {background:#635BFF;color:white;border-radius:10px;font-weight:600;padding:0.45rem 0.9rem;}
.stButton>button:hover{background:#4f47d6;}
/* metric cards */
.card {background:#111827;padding:16px;border-radius:12px;box-shadow:0 2px 10px rgba(0,0,0,.25);border:1px solid #1f2937;}
.card h3{margin:0 0 8px 0;font-size:0.95rem;color:#9ca3af;}
.card .v{font-size:1.6rem;font-weight:700;color:#e5e7eb;}
/* tidy plot padding */
.block-container {padding-top: 1.2rem;}
</style>
""", unsafe_allow_html=True)

# --- session state ---
ss = st.session_state
ss.setdefault("df", None)
ss.setdefault("result", None)
ss.setdefault("chart_descs", None)
ss.setdefault("insights", None)

rules = load_rules()

# ---------- Sidebar Navigation ----------
with st.sidebar:
    # if result exists and we just analyzed, jump to Insights
    default_idx = 1 if (st.session_state.get("_go_insights") or st.session_state.get("result") is not None) else 0
    page = option_menu(
        "Navigation",
        ["Upload Data", "Insights", "Settings / About"],
        icons=["cloud-upload", "bar-chart-line", "gear"],
        default_index=default_idx,
    )
    # clear the flag so future reruns don’t keep forcing Insights
    if st.session_state.get("_go_insights"):
        st.session_state["_go_insights"] = False

# ---------- Helper: run analysis ----------
def run_analysis(df: pd.DataFrame):
    result = compute_kpis(df, rules)

    # build charts + descriptors (for optional AI)
    figs, descs = [], []
    f1, d1 = fig_trend_sales(result["kpis_weekly"]); figs.append(f1); descs.append(d1)
    f2, d2 = fig_sales_by_region(result["by_region"]); figs.append(f2); descs.append(d2)
    f3, d3 = fig_top_departments(result["by_department"]); figs.append(f3); descs.append(d3)

    ss.df = df
    ss.result = result
    ss.chart_descs = descs
    ss.insights = None

# ===================== PAGES =====================

# -------- Upload Page --------
if page == "Upload Data":
    st.title("📥 Upload Your Data")
    st.write("Upload a CSV/XLSX with columns: `date, store, department, region, weekly_sales, transactions`.")

    file = st.file_uploader("Choose a file", type=["csv","xlsx"])
    if file:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        st.write("**Preview**")
        st.dataframe(df.head(25), use_container_width=True)
        if st.button("🔍 Analyze"):
            run_analysis(df)             # computes ss.result / ss.chart_descs
            st.session_state["_go_insights"] = True
            st.rerun()

# -------- Insights Page --------
elif page == "Insights":
    st.title("📈 Insights Dashboard")
    if ss.result is None:
        st.info("Upload a dataset on the **Upload Data** tab first.")
        st.stop()

    r = ss.result

    # KPI cards
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"""<div class="card"><h3>Rows</h3><div class="v">{r['shape']['rows']}</div></div>""", unsafe_allow_html=True)
    with c2:
        trend = "—" if pd.isna(r["trend_4wk"]) else f"{r['trend_4wk']*100:.2f}%"
        st.markdown(f"""<div class="card"><h3>4-Week Trend</h3><div class="v">{trend}</div></div>""", unsafe_allow_html=True)
    with c3: st.markdown(f"""<div class="card"><h3>Outliers Flagged</h3><div class="v">{r['outliers']}</div></div>""", unsafe_allow_html=True)

    st.divider()

    # Charts
    colA, colB = st.columns(2)
    fig1, _ = fig_trend_sales(r["kpis_weekly"]);   colA.plotly_chart(fig1, use_container_width=True)
    fig2, _ = fig_sales_by_region(r["by_region"]); colB.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig_top_departments(r["by_department"])[0], use_container_width=True)

    st.divider()

    # AI narrative (optional)
    use_ai = st.toggle("Generate AI Narrative & Recommendations", value=True)
    if use_ai:
        if ss.insights is None:
            with st.spinner("Drafting insights..."):
                ss.insights = draft_insights(r, ss.chart_descs)
        st.subheader("🧠 AI-Generated Insights")
        st.markdown(ss.insights)
    else:
        st.info("AI narrative disabled. Deterministic KPIs & charts shown above.")

# -------- Settings / About --------
elif page == "Settings / About":
    st.title("⚙️ Settings / About")
    st.markdown(
        """
        **Analysis Mode:** Deterministic (config-driven).  
        - Cleaning rules, KPI math, and outlier policy live in `config/rules.yaml`.  
        - LLM is used *only* to summarize approved results.
        
        **How to use**
        1. Go to **Upload Data**, add your CSV/XLSX.  
        2. Click **Analyze** → review the **Insights** page.  
        3. Toggle AI narrative on/off.
        """
    )
