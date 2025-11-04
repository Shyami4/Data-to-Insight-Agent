from chart_style import apply_common_style, color_discrete_map_from_categories
from plotly.colors import qualitative as q
import plotly.express as px
import pandas as pd


def fig_trend_sales(weekly_dict):
    w = pd.DataFrame(weekly_dict)
    y = "weekly_sales_sum" if "weekly_sales_sum" in w.columns else w.columns[-1]
    fig = px.line(w, x="date", y=y, title="Weekly Sales Trend")
    fig = apply_common_style(
        fig,
        title="Weekly Sales Trend",
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        height=500,
        show_legend=False,   # single series; no legend needed
        dark=True,
    )
    desc = {
        "title": "Weekly Sales Trend",
        "points": int(len(w)),
        "last_value": float(w[y].iloc[-1]) if len(w) else None,
    }
    return fig, desc

def fig_sales_by_region(reg_dict):
    r = pd.DataFrame(reg_dict)
    r = r.sort_values("weekly_sales_sum", ascending=False)
    color_map = color_discrete_map_from_categories(r["region"].tolist(), q.Set2)

    fig = px.bar(
        r,
        x="region",
        y="weekly_sales_sum",
        text="weekly_sales_sum",
        color="region",
        color_discrete_map=color_map,
    )
    fig = apply_common_style(
        fig,
        title="Sales by Region",
        xaxis_title="Region",
        yaxis_title="Total Sales ($)",
        legend_title="Region",
        height=500,
    )
    desc = {
        "title": "Sales by Region",
        "top_regions": r["region"].head(3).tolist(),
        "top_values": [float(x) for x in r["weekly_sales_sum"].head(3)],
    }
    return fig, desc


def fig_top_departments(dep_dict, top_n=10):
    d = pd.DataFrame(dep_dict).sort_values("weekly_sales_sum", ascending=False).head(top_n)
    color_map = color_discrete_map_from_categories(d["department"].tolist(), q.Set2)

    fig = px.bar(
        d,
        x="department",
        y="weekly_sales_sum",
        text="weekly_sales_sum",
        color="department",
        color_discrete_map=color_map,
    )
    fig = apply_common_style(
        fig,
        title=f"Top {top_n} Departments by Sales",
        xaxis_title="Department",
        yaxis_title="Total Sales ($)",
        legend_title="Department",
        height=500,
    )
    desc = {
        "title": "Top Departments",
        "top_departments": d["department"].head(3).tolist(),
        "top_values": [float(x) for x in d["weekly_sales_sum"].head(3)],
    }
    return fig, desc

