# plots.py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Colors used across the app (top-level in plots.py) ---
COLOR_ABOVE = "#69d3f3"
COLOR_BELOW = "#f547ce"
COLOR_AVG   = "#a3a3a3"

# ----------------------- Utilities -----------------------

def prep_common(df: pd.DataFrame):
    df = df.copy()
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # harmonize names if source still uses weekly_sales
    if "sales" not in df and "weekly_sales" in df:
        df["sales"] = df["weekly_sales"]
    for c in ["profit", "quantity"]:
        if c not in df:
            df[c] = np.nan
    if "date" in df:
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["week"] = df["date"].dt.isocalendar().week.astype(int)
    return df

def _weekly_guard(df, need):
    """Return (ok: bool, df) after basic column/date checks."""
    if df is None or df.empty or need - set(df.columns):
        return False, df
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    return (not d.empty), d

# ----------------------- Overview ------------------------

def fig_sales_trend_forecast_shaded(weekly_dict, height=240, horizon=2) -> go.Figure:
    """Trend with 4-wk MA and shaded simple forecast (extending last MA)."""
    w = pd.DataFrame(weekly_dict)
    if w.empty or "date" not in w.columns:
        return go.Figure()
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    ycol = "weekly_sales_sum" if "weekly_sales_sum" in w.columns else w.columns[-1]
    w = w.dropna(subset=["date", ycol]).sort_values("date")
    if w.empty:
        return go.Figure()

    w["ma4"] = w[ycol].rolling(4, min_periods=1).mean()
    fig = go.Figure()
    fig.add_scatter(x=w["date"], y=w[ycol], mode="lines+markers", name="Sales")
    fig.add_scatter(x=w["date"], y=w["ma4"], mode="lines", name="4-wk MA",
                    line=dict(dash="dash"))

    # shaded forecast: extend last MA forward by horizon weeks
    last_date = w["date"].iloc[-1]
    future_dates = pd.date_range(last_date, periods=horizon+1, freq="W", inclusive="right")
    if len(future_dates) > 0:
        last_ma = float(w["ma4"].iloc[-1])
        fig.add_scatter(
            x=list(future_dates), y=[last_ma]*len(future_dates),
            name="Forecast", line=dict(width=0), fill="tozeroy",
            fillcolor="rgba(99,91,255,.15)", mode="lines"
        )

    fig.update_layout(
        title="Sales Trend + 4-Week MA (2-week shaded forecast)",
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Week", yaxis_title="Sales ($)", hovermode="x unified"
    )
    return fig

def fig_wow_bars(weekly_dict: dict, height=280) -> go.Figure:
    import numpy as np
    import pandas as pd
    import plotly.express as px

    w = pd.DataFrame(weekly_dict).copy()
    if w.empty or "date" not in w or "weekly_sales_sum" not in w:
        return go.Figure()

    # Prepare data
    w["date"] = pd.to_datetime(w["date"])
    w = w.sort_values("date")
    w["wow"] = w["weekly_sales_sum"].pct_change()

    # Format hover text (human-readable)
    w["hover_text"] = w.apply(
        lambda r: f"{'+' if r['wow'] >= 0 else ''}{r['wow']*100:.1f}% growth<br>on {r['date'].strftime('%b %d, %Y')}",
        axis=1
    )

    # Use softer, modern colors (teal/coral)
    colors = np.where(w["wow"].fillna(0) >= 0,COLOR_ABOVE, COLOR_BELOW)

    # Build bar chart
    fig = px.bar(
        w,
        x="date",
        y="wow",
        title="Week-over-Week Revenue Growth",
        labels={"date": "Week", "wow": "WoW Growth (%)"},
        hover_name="hover_text"  # custom hover
    )

    # Apply color and layout enhancements
    fig.update_traces(
        marker_color=colors,
        hovertemplate="%{customdata[0]}"  # use custom hover text
    )

    # Fix hover display using 'customdata'
    fig.update_traces(customdata=w[["hover_text"]].values)

    fig.update_layout(
        height=height,
        title_font=dict(size=22, color="#e5e7eb", family="Inter, sans-serif"),
        xaxis_title="Week",
        yaxis_title="Growth (%)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, l=40, r=20, b=40),
        hoverlabel=dict(
            bgcolor="#1f2937",
            font_size=13,
            font_color="#e5e7eb"
        )
    )

    fig.update_yaxes(tickformat=".0%")

    return fig


def fig_top_movers_stores(df: pd.DataFrame, top_n=3) -> go.Figure:
    ok, d = _weekly_guard(df, {"store", "weekly_sales", "date"})
    if not ok: return go.Figure()
    s = (d.groupby("store", as_index=False)["weekly_sales"].sum()
           .sort_values("weekly_sales", ascending=False).head(top_n))
    fig = px.bar(s, x="weekly_sales", y="store", orientation="h",
                 title=f"Top {top_n} Stores",
                 labels={"weekly_sales":"Sales ($)", "store":""},
                 text="weekly_sales")
    fig.update_traces(texttemplate="%{text:,.0f}")
    fig.update_layout(height=300, showlegend=False)
    return fig

def fig_bottom_departments(df: pd.DataFrame, bottom_n=3) -> go.Figure:
    """Standardized signature used by app.py (bottom_n)."""
    ok, d = _weekly_guard(df, {"department", "weekly_sales", "date"})
    if not ok: return go.Figure()
    x = (d.groupby("department", as_index=False)["weekly_sales"].sum()
           .sort_values("weekly_sales", ascending=True).head(bottom_n))
    fig = px.bar(x, x="weekly_sales", y="department", orientation="h",
                 title=f"Bottom {bottom_n} Departments",
                 labels={"weekly_sales":"Sales ($)", "department":""},
                 text="weekly_sales")
    fig.update_traces(marker_color="#ef4444", texttemplate="%{text:,.0f}")
    fig.update_layout(height=300, showlegend=False)
    return fig

# ------------------ Drivers & Performance ------------------

def fig_regional_share(df: pd.DataFrame, height=360) -> go.Figure:
    ok, d = _weekly_guard(df, {"region", "weekly_sales", "date"})
    if not ok: return go.Figure()
    g = d.groupby("region", as_index=False)["weekly_sales"].sum()
    total = g["weekly_sales"].sum()
    g["share"] = g["weekly_sales"] / (total if total else 1)
    fig = px.bar(g.sort_values("share", ascending=False),
                 x="region", y="share",
                 title="Regional Contribution (% of total)",
                 labels={"region":"Region","share":"Share"})
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=height, showlegend=False)
    return fig

def fig_dept_pareto(df: pd.DataFrame, height=360) -> go.Figure:
    ok, d = _weekly_guard(df, {"department", "weekly_sales", "date"})
    if not ok: return go.Figure()
    d2 = (d.groupby("department", as_index=False)["weekly_sales"].sum()
            .sort_values("weekly_sales", ascending=False))
    d2["cum_share"] = d2["weekly_sales"].cumsum() / d2["weekly_sales"].sum()
    fig = go.Figure()
    fig.add_bar(x=d2["department"], y=d2["weekly_sales"], name="Sales")
    fig.add_scatter(x=d2["department"], y=d2["cum_share"], name="Cumulative %",
                    yaxis="y2", line=dict(dash="dash"))
    fig.update_layout(
        title="Department 80/20 (Pareto)", height=height,
        yaxis=dict(title="Sales ($)"),
        yaxis2=dict(title="Share", overlaying="y", side="right", tickformat=".0%")
    )
    return fig

def fig_dept_sparklines_top3(df: pd.DataFrame, k: int = 3, height: int = 280):
    need = {"date", "department", "weekly_sales"}
    ok, d = _weekly_guard(df, need)
    if not ok: return go.Figure()

    weekly = (d.groupby(["department", pd.Grouper(key="date", freq="W")], as_index=False)
                ["weekly_sales"].sum())

    top_names = (weekly.groupby("department", as_index=False)["weekly_sales"]
                      .sum().sort_values("weekly_sales", ascending=False)
                      .head(k)["department"].tolist())
    if not top_names:
        return go.Figure()

    top = weekly[weekly["department"].isin(top_names)].copy()
    cat_order = pd.Categorical(top["department"], categories=top_names, ordered=True)
    top = top.assign(department=cat_order).sort_values(["department", "date"])

    fig = px.line(
        top, x="date", y="weekly_sales", color=None, facet_row="department",
        height=max(height, 220 + 60*(len(top_names)-1))
    )
    fig.update_traces(line=dict(width=2))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_yaxes(title="Sales ($)", matches=None, showgrid=True)
    fig.update_xaxes(title=None, showgrid=False)
    fig.update_layout(
        title="Top Departments – Weekly Trends",
        showlegend=False, margin=dict(l=40, r=20, t=60, b=30),
        hovermode="x unified"
    )
    return fig

def fig_store_consistency_heatmap(df: pd.DataFrame, missing_only=False, height=360) -> go.Figure:
    need = {"date", "store", "weekly_sales"}
    ok, d = _weekly_guard(df, need)
    if not ok: return go.Figure()

    pivot = d.pivot_table(
        index="store",
        columns=d["date"].dt.to_period("W"),
        values="weekly_sales",
        aggfunc="sum"
    )

    if missing_only:
        pivot = pivot.isna().astype(int)
        title = "Missing Sales Heatmap (dark = more missing)"
        colors = "Reds"
    else:
        title = "Store × Week Heatmap (Sales)"
        colors = "Blues"

    pivot = pivot.fillna(0)
    pivot.columns = [p.start_time for p in pivot.columns]  # Period → timestamp

    fig = px.imshow(pivot, color_continuous_scale=colors, aspect="auto")
    fig.update_layout(height=height, title=title, xaxis_title="Week", yaxis_title="Store")
    return fig

# ---------------- Diagnostics & Efficiency ----------------

def _linreg_r2(x, y):
    x = np.asarray(x); y = np.asarray(y)
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x = x[mask]; y = y[mask]
    if len(x) < 2:
        return 0.0, (0.0, 0.0)
    b1, b0 = np.polyfit(x, y, 1)
    y_hat = b1*x + b0
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0
    return float(r2), (float(b1), float(b0))

def fig_efficiency_quadrants_r2(df: pd.DataFrame, height=420):
    need = {"weekly_sales", "transactions", "store", "region"}
    if df is None or df.empty or need - set(df.columns):
        return go.Figure(), 0.0
    g = (df.groupby(["store","region"], as_index=False)
           .agg(sales=("weekly_sales","sum"), txns=("transactions","sum")))
    if g.empty:
        return go.Figure(), 0.0
    g["value_per_txn"] = g["sales"] / g["txns"].clip(lower=1)
    x_med, y_med = g["value_per_txn"].median(), g["txns"].median()
    r2, (b1, b0) = _linreg_r2(g["txns"], g["sales"])

    fig = px.scatter(
        g, x="txns", y="sales", color="region", size="value_per_txn",
        labels={"txns":"Transactions", "sales":"Sales ($)"},
        title="Efficiency: Sales vs Transactions (Quadrants)"
    )
    fig.add_hline(y=y_med, line_dash="dot", opacity=.5)
    fig.add_vline(x=x_med, line_dash="dot", opacity=.5)

    xs = np.linspace(g["txns"].min(), g["txns"].max(), 50)
    ys = b1*xs + b0
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Trend"))
    fig.update_layout(height=height, legend_title_text="Region")
    return fig, r2

def outlier_table_iqr(df: pd.DataFrame, k=1.5) -> pd.DataFrame:
    need = {"date", "store", "department", "weekly_sales"}
    if df is None or df.empty or need - set(df.columns):
        return pd.DataFrame()
    d = df.dropna(subset=["weekly_sales", "date"]).copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    out_rows = []
    for store, g in d.groupby("store"):
        q1, q3 = g["weekly_sales"].quantile(0.25), g["weekly_sales"].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k*iqr, q3 + k*iqr
        mask = ~g["weekly_sales"].between(lo, hi)
        if mask.any():
            out_rows.append(g.loc[mask, ["date", "store", "department", "weekly_sales"]])
    if not out_rows:
        return pd.DataFrame()
    out = pd.concat(out_rows).sort_values("weekly_sales", ascending=False).head(50)
    out["date"] = out["date"].dt.date
    out["z_flag"] = np.sign(out["weekly_sales"] - out["weekly_sales"].median()).map({1:"Spike",-1:"Drop"}).fillna("Anomaly")
    return out.rename(columns={"weekly_sales":"Sales ($)"})

# ---------------- KPI Helpers ----------------

def _sparkline(df, ycol, height=70):
    if df is None or df.empty or {"date", ycol} - set(df.columns):
        return go.Figure()
    s = (df.dropna(subset=[ycol, "date"])
           .sort_values("date")
           .groupby(pd.Grouper(key="date", freq="W"))[ycol]
           .sum()
           .reset_index())
    fig = px.line(s, x="date", y=ycol)
    fig.update_traces(mode="lines+markers", line_width=2, marker_size=5)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      showlegend=False)
    return fig

def kpi_value_and_delta_vs_py(df, ycol):
    if df is None or df.empty or {"date", ycol} - set(df.columns):
        return 0.0, 0.0
    d = df.dropna(subset=["date", ycol]).copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    if d.empty:
        return 0.0, 0.0
    d["year"] = d["date"].dt.year
    curr = d["year"].max()
    d_curr = d[d["year"] == curr]
    d_py = d[d["year"] == curr - 1]
    if d_py.empty:
        return float(d_curr[ycol].sum()), 0.0
    curr_sum = d_curr.groupby(d_curr["date"].dt.isocalendar().week)[ycol].sum().sum()
    py_sum = d_py.groupby(d_py["date"].dt.isocalendar().week)[ycol].sum().sum()
    delta = 0.0 if py_sum == 0 else (curr_sum - py_sum) / py_sum
    return float(curr_sum), float(delta)

import pandas as pd
import numpy as np
import plotly.express as px

def _safe_weekly(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date", "weekly_sales"])
    d = d.sort_values("date")
    return d

def _aggregate_and_color(df: pd.DataFrame, cat_col: str, val_col: str):
    """
    Group by cat_col, sum val_col, sort descending, compute avg and colors.
    Returns (agg_df, avg, colors).
    """
    if df is None or df.empty or {cat_col, val_col} - set(df.columns):
        return pd.DataFrame(columns=[cat_col, val_col]), 0.0, []

    g = (df.dropna(subset=[val_col])
           .groupby(cat_col, as_index=False)[val_col]
           .sum()
           .sort_values(val_col, ascending=False))

    avg = g[val_col].mean() if not g.empty else 0.0
    colors = np.where(g[val_col] >= avg, COLOR_ABOVE, COLOR_BELOW).tolist()
    return g, float(avg), colors

def fig_benchmark_bars(
    df: pd.DataFrame,
    cat_col: str,
    val_col: str = "weekly_sales",
    title: str = "Benchmark",
    height: int = 420,
) -> go.Figure:
    """
    Vertical bar chart with average reference line and conditional coloring.
    - Bars sorted descending
    - Blue above avg, Pink below avg
    - Average line labeled
    """
    g, avg, colors = _aggregate_and_color(df, cat_col, val_col)
    if g.empty:
        return go.Figure()

    # variance vs avg for hover
    variance = g[val_col] - avg
    variance_pct = np.where(avg == 0, 0, variance / avg)

    fig = go.Figure()

    fig.add_bar(
        x=g[cat_col],
        y=g[val_col],
        marker_color=colors,
        hovertemplate=(
            f"<b>%{{x}}</b><br>{val_col.replace('_', ' ').title()}: $%{{y:,.0f}}"
            "<br>Δ vs Avg: $%{customdata[0]:,.0f} (%{customdata[1]:+.1%})<extra></extra>"
        ),
        customdata=np.stack([variance, variance_pct], axis=1),
    )

    # Average reference line
    fig.add_hline(
        y=avg,
        line_dash="dot",
        line_color=COLOR_AVG,
        annotation_text=f"Avg: ${avg/1000:,.1f}K",
        annotation_position="top right",   # moved to right side
        annotation_font=dict(color=COLOR_AVG, size=12, family="Inter, sans-serif"),
    )

    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis_title=None,
        yaxis_title="Sales ($)",
        hovermode="x unified",
    )
    return fig

def fig_store_benchmark(df: pd.DataFrame, height: int = 420) -> go.Figure:
    # Backward compatibility if dataset uses 'weekly_sales'
    return fig_benchmark_bars(
        df=df,
        cat_col="store",
        val_col="weekly_sales",
        title="Store Rankings (Total Sales vs Avg)",
        height=height,
    )

def fig_department_benchmark(df: pd.DataFrame, height: int = 420) -> go.Figure:
    return fig_benchmark_bars(
        df=df,
        cat_col="department",
        val_col="weekly_sales",
        title="Department Rankings (Total Sales vs Avg)",
        height=height,
    )

def _share_topn(df: pd.DataFrame, cat_col: str, val_col: str, top_n: int = 5):
    if df is None or df.empty or {cat_col, val_col} - set(df.columns):
        return pd.DataFrame(columns=[cat_col, val_col, "share"])
    g = (df.dropna(subset=[val_col])
           .groupby(cat_col, as_index=False)[val_col]
           .sum()
           .sort_values(val_col, ascending=False))
    total = g[val_col].sum()
    if total == 0:
        g["share"] = 0.0
        return g.iloc[:top_n, :].copy()

    top = g.iloc[:top_n, :].copy()
    top["share"] = top[val_col] / total

    # collapse the remainder into "Others"
    if len(g) > top_n:
        others_val = g.iloc[top_n:, :][val_col].sum()
        if others_val > 0:
            top = pd.concat([top, pd.DataFrame({cat_col:["Others"], val_col:[others_val], "share":[others_val/total]})])

    return top.reset_index(drop=True)

def table_store_4wk_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each store:
      last_4w  = sum of the most recent 4 weekly buckets
      prev_4w  = sum of the 4 weeks immediately before that
      pct_change = (last_4w - prev_4w) / prev_4w
    Returns a styled DataFrame ready for st.dataframe.
    """
    d = _safe_weekly(df)
    if d.empty or "store" not in d:
        return pd.DataFrame(columns=["store","last_4w","prev_4w","pct_change"])

    # aggregate to weekly buckets first (handles multiple rows per week)
    w = (d.groupby(["store", pd.Grouper(key="date", freq="W")], as_index=False)["weekly_sales"]
          .sum()
          .rename(columns={"weekly_sales":"sales"}))

    # take last 8 weeks per store so we can compare 4 vs previous 4
    def _last8(g):
        g = g.sort_values("date").tail(8)
        last4 = g.tail(4)["sales"].sum()
        prev4 = g.head(max(len(g)-4, 0))["sales"].tail(4).sum()
        pct   = np.nan if prev4 == 0 else (last4 - prev4)/prev4
        return pd.Series({"last_4w": float(last4), "prev_4w": float(prev4), "pct_change": float(pct) if pd.notna(pct) else np.nan})

    t = w.groupby("store").apply(_last8).reset_index()
    t = t.sort_values("pct_change", ascending=False)

    # neat display columns
    t["Δ vs prev 4w"] = t["pct_change"].apply(lambda v: "" if pd.isna(v) else f"{v*100:+.1f}%")
    t["last_4w"] = t["last_4w"].round(0).astype(int)
    t["prev_4w"] = t["prev_4w"].round(0).astype(int)

    return t[["store","last_4w","prev_4w","Δ vs prev 4w"]]


# Add to plots.py

def fig_regional_growth_comparison(df: pd.DataFrame, height: int = 360) -> go.Figure:
    """Regional sales with growth overlay - shows both volume and momentum"""
    ok, d = _weekly_guard(df, {"region", "weekly_sales", "date"})
    if not ok:
        return go.Figure()
    
    # Calculate regional totals and growth
    regional = d.groupby(["region", pd.Grouper(key="date", freq="W")], as_index=False)["weekly_sales"].sum()
    
    # Get latest total by region
    latest_totals = regional.groupby("region").tail(1)
    latest_totals = latest_totals.sort_values("weekly_sales", ascending=False)
    
    # Calculate WoW growth
    regional["prev"] = regional.groupby("region")["weekly_sales"].shift(1)
    regional["growth"] = (regional["weekly_sales"] - regional["prev"]) / regional["prev"] * 100
    latest_growth = regional.groupby("region").tail(1)
    
    # Merge
    combined = latest_totals.merge(
        latest_growth[["region", "growth"]],
        on="region",
        how="left"
    )
    
    fig = go.Figure()
    
    # Bars for total sales
    fig.add_trace(go.Bar(
        x=combined["region"],
        y=combined["weekly_sales"],
        name="Total Sales",
        marker_color="#3b82f6",
        yaxis="y"
    ))
    
    # Line for growth
    fig.add_trace(go.Scatter(
        x=combined["region"],
        y=combined["growth"],
        name="Growth %",
        mode="lines+markers",
        marker=dict(size=10, color="#22c55e"),
        line=dict(width=3, color="#22c55e"),
        yaxis="y2"
    ))
    
    fig.update_layout(
        title="Regional Performance: Sales Volume & Growth",
        height=height,
        yaxis=dict(title="Sales ($)"),
        yaxis2=dict(
            title="Growth (%)",
            overlaying="y",
            side="right",
            zeroline=True,
            zerolinecolor="#94a3b8",
            zerolinewidth=2
        ),
        hovermode="x unified"
    )
    
    return fig


def fig_department_trends_small_multiples(df: pd.DataFrame, height: int = 400) -> go.Figure:
    """Show all departments as small multiple trends for pattern recognition"""
    ok, d = _weekly_guard(df, {"date", "department", "weekly_sales"})
    if not ok:
        return go.Figure()
    
    weekly = d.groupby(["department", pd.Grouper(key="date", freq="W")], as_index=False)["weekly_sales"].sum()
    
    depts = sorted(weekly["department"].unique())
    
    fig = px.line(
        weekly,
        x="date",
        y="weekly_sales",
        facet_col="department",
        facet_col_wrap=3,
        height=height
    )
    
    fig.update_traces(line_color="#635BFF", line_width=2)
    fig.update_xaxes(title=None, showgrid=False)
    fig.update_yaxes(title="Sales", matches=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    
    fig.update_layout(
        title="Department Trends (All Departments)",
        showlegend=False,
        margin=dict(t=60, l=40, r=20, b=30)
    )
    
    return fig


def fig_store_efficiency_scatter(df: pd.DataFrame, height: int = 400) -> go.Figure:
    """Scatter plot: Avg Transaction Value vs Transaction Volume by store"""
    need = {"store", "weekly_sales", "transactions"}
    if df is None or df.empty or need - set(df.columns):
        return go.Figure()
    
    store_metrics = df.groupby("store").agg({
        "weekly_sales": "sum",
        "transactions": "sum"
    }).reset_index()
    
    store_metrics["avg_transaction_value"] = (
        store_metrics["weekly_sales"] / store_metrics["transactions"]
    )
    
    fig = px.scatter(
        store_metrics,
        x="transactions",
        y="avg_transaction_value",
        text="store",
        size="weekly_sales",
        title="Store Efficiency: Transaction Volume vs Average Value",
        labels={
            "transactions": "Total Transactions",
            "avg_transaction_value": "Avg Transaction Value ($)"
        },
        height=height
    )
    
    fig.update_traces(
        textposition="top center",
        marker=dict(color="#8b5cf6", line=dict(width=2, color="white"))
    )
    
    # Add quadrant lines
    median_txns = store_metrics["transactions"].median()
    median_value = store_metrics["avg_transaction_value"].median()
    
    fig.add_vline(x=median_txns, line_dash="dash", line_color="#64748b", opacity=0.5)
    fig.add_hline(y=median_value, line_dash="dash", line_color="#64748b", opacity=0.5)
    
    fig.update_layout(height=height)
    
    return fig