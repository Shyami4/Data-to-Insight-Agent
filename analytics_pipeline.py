# analytics_pipeline.py
from __future__ import annotations
from pandas.api.types import is_integer_dtype
import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any, Optional


# ───────────────────────────────────────────────────────────────────────────────
# Rules loader
# ───────────────────────────────────────────────────────────────────────────────

def load_rules(path: str = "config/rules.yaml") -> dict:
    """Load YAML rules (dataset dtypes, cleaning, KPI config, etc.)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ───────────────────────────────────────────────────────────────────────────────
# Dtype coercion & cleaning
# ───────────────────────────────────────────────────────────────────────────────

def coerce_dtypes(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Coerce dataframe columns to types specified in rules['dataset']['dtypes']."""
    df = df.copy()
    dt = (rules.get("dataset", {}).get("dtypes") or {})
    for col, typ in dt.items():
        if col not in df.columns:
            continue
        if typ == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif typ == "string":
            df[col] = df[col].astype("string")
        elif typ == "category":
            df[col] = df[col].astype("category")
        elif typ == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif typ == "int":
            # use pandas nullable int
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def apply_cleaning(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    c = rules["cleaning"]

    if c.get("trim_strings", False):
        for col in df.select_dtypes(include="string").columns:
            df[col] = df[col].str.strip()

    if c.get("drop_duplicates", False):
        df = df.drop_duplicates()

    # Clip ranges
    for col, rng in (c.get("clip_ranges") or {}).items():
        if col not in df.columns:
            continue
        lo, hi = rng
        if lo is not None: df[col] = df[col].clip(lower=float(lo))
        if hi is not None: df[col] = df[col].clip(upper=float(hi))

    # Null policy (safe for integer dtypes)
    nullp = c.get("null_policy") or {}
    for col, policy in nullp.items():
        if col not in df.columns:
            continue

        ser = df[col]

        if policy == "drop_row":
            df = df[ser.notna()]
            continue

        if policy == "zero":
            fill_val = 0
            # cast type-appropriate zero
            if is_integer_dtype(ser.dtype):
                df[col] = ser.fillna(int(fill_val))
            else:
                df[col] = ser.fillna(float(fill_val))
            continue

        if policy == "impute_median":
            med = ser.median(skipna=True)
            # handle all-null columns safely
            if pd.isna(med):
                med = 0
            if is_integer_dtype(ser.dtype):
                df[col] = ser.fillna(int(round(float(med))))
            else:
                df[col] = ser.fillna(float(med))
            continue

        # default: no-op

    return df

# ───────────────────────────────────────────────────────────────────────────────
# Robust outlier flagging (IQR) – non-destructive
# ───────────────────────────────────────────────────────────────────────────────

def flag_outliers_iqr(
    df: pd.DataFrame,
    col: str = "weekly_sales",
    group_by: Optional[str] = None,
    k: float = 1.5,
) -> pd.DataFrame:
    """
    Flag outliers using IQR method. Adds boolean column 'is_outlier'.
    If group_by is provided, thresholds are computed within each group.
    """
    df = df.copy()
    df["is_outlier"] = False

    if group_by is not None and group_by in df.columns:
        def _mark(g: pd.DataFrame) -> pd.DataFrame:
            s = g[col].dropna().astype(float)
            if s.empty:
                return g
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lb, ub = q1 - k * iqr, q3 + k * iqr
            g.loc[~g[col].between(lb, ub), "is_outlier"] = True
            return g

        return df.groupby(group_by, group_keys=False).apply(_mark)

    s = df[col].dropna().astype(float)
    if not s.empty:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lb, ub = q1 - k * iqr, q3 + k * iqr
        df.loc[~df[col].between(lb, ub), "is_outlier"] = True

    return df


# ───────────────────────────────────────────────────────────────────────────────
# KPI Aggregation
# ───────────────────────────────────────────────────────────────────────────────

def _eval_derived_expressions(weekly: pd.DataFrame, derived_rules: dict) -> pd.DataFrame:
    """
    Evaluate simple derived expressions from rules['kpis']['derived'] into the weekly dataframe.
    Supports NULLIF(x,0) -> np.where(x==0, np.nan, x).
    """
    if not derived_rules:
        return weekly

    safe_weekly = weekly.copy()
    env = {"np": np, **{c: safe_weekly[c] for c in safe_weekly.columns}}

    for name, expr in derived_rules.items():
        # very light safety transform for NULLIF(x,0)
        safe_expr = expr.replace("NULLIF(", "np.where(").replace(",0)", ",0, np.nan)")
        safe_weekly[name] = eval(safe_expr, {}, env)

    return safe_weekly


def compute_kpis(df: pd.DataFrame, rules: dict) -> Dict[str, Any]:
    """
    Deterministic KPI computation with:
      - type coercion & cleaning (rules)
      - robust IQR outlier flagging (non-destructive)
      - resampled KPIs (freq & aggregates from rules)
      - derived metrics
      - weekly WoW & 4-week moving average
      - Pareto 80/20 by department
      - missingness table
    Returns a plain-JSON-serializable dict for the UI layer.
    """
    if df is None or df.empty:
        return {
            "shape": {"rows": 0, "cols": 0},
            "kpis_weekly": {},
            "by_region": {},
            "by_dept": {},
            "pareto_topN_dept": 0,
            "missing": {},
            "outliers": 0,
            "trend_4wk": np.nan,
        }

    # 1) Validate required columns
    req = (rules.get("dataset", {}).get("required_columns") or [])
    missing_cols = [c for c in req if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # 2) Coerce & clean
    df = coerce_dtypes(df, rules)
    df = apply_cleaning(df, rules)

    # 3) Non-destructive outlier flagging
    out_cfg = rules.get("outliers", {}) or {}
    out_group = out_cfg.get("group_by", "store") if out_cfg else "store"
    out_k = float(out_cfg.get("k", 1.5)) if out_cfg else 1.5
    df = flag_outliers_iqr(df, col="weekly_sales", group_by=out_group, k=out_k)

    # 4) Time resampling for weekly KPIs
    #    Build aggregation map from rules['kpis']['aggregates']
    freq = rules.get("kpis", {}).get("time_freq", "W")  # default weekly
    dft = df.set_index("date").sort_index()

    agg_map = {}
    for col, aggs in (rules.get("kpis", {}).get("aggregates") or {}).items():
        for a in aggs:
            agg_map[f"{col}_{a}"] = (col, a)

    if not agg_map:
        # sensible default to avoid KeyError if rules are thin
        agg_map = {"weekly_sales_sum": ("weekly_sales", "sum"),
                   "transactions_sum": ("transactions", "sum")}

    weekly = dft.resample(freq).agg(**agg_map).reset_index()

    # 5) Derived metrics from rules (optional)
    weekly = _eval_derived_expressions(weekly, rules.get("kpis", {}).get("derived"))

    # 6) Add WoW growth & 4-week MA if sales present
    if "weekly_sales_sum" in weekly.columns:
        weekly = weekly.sort_values("date").reset_index(drop=True)
        weekly["wow_growth_pct"] = weekly["weekly_sales_sum"].pct_change()
        weekly["ma_4wk"] = weekly["weekly_sales_sum"].rolling(4, min_periods=1).mean()

        if len(weekly) >= 4:
            arr = weekly["weekly_sales_sum"].tail(4).to_numpy(dtype=float)
            trend_4wk = float((arr[-1] - arr[0]) / max(arr[0], 1e-9))
        else:
            trend_4wk = np.nan
    else:
        trend_4wk = np.nan

    # 7) By region / by department
    by_region = (
        df.groupby("region", as_index=False)["weekly_sales"]
          .sum()
          .rename(columns={"weekly_sales": "weekly_sales_sum"})
          .sort_values("weekly_sales_sum", ascending=False)
    )

    by_dept = (
        df.groupby("department", as_index=False)["weekly_sales"]
          .sum()
          .rename(columns={"weekly_sales": "weekly_sales_sum"})
          .sort_values("weekly_sales_sum", ascending=False)
    )

    # 8) Pareto (80/20) over departments
    if not by_dept.empty:
        cum = by_dept["weekly_sales_sum"].cumsum() / by_dept["weekly_sales_sum"].sum()
        pareto_topN = int((cum <= 0.8).sum())
        by_dept = by_dept.assign(cum_share=cum)
    else:
        pareto_topN = 0

    # 9) Missingness table
    missing = (
        df.isna().mean().sort_values(ascending=False)
          .reset_index()
          .rename(columns={"index": "column", 0: "missing_rate"})
    )

    # 10) Output payload
    result = {
        "shape": {"rows": int(len(df)), "cols": int(len(df.columns))},
        "kpis_weekly": weekly.to_dict(orient="list") if not weekly.empty else {},
        "by_region": by_region.to_dict(orient="list") if not by_region.empty else {},
        "by_department": by_dept.to_dict(orient="list") if not by_dept.empty else {},
        "outliers": int(df["is_outlier"].sum()) if "is_outlier" in df.columns else 0,
        "trend_4wk": float(trend_4wk) if pd.notna(trend_4wk) else None,
    }
    return result
