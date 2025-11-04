import pandas as pd, numpy as np, yaml
from typing import Dict, Any

# --------- rules loader ----------
def load_rules(path: str = "config/rules.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# --------- dtype & cleaning ----------
def coerce_dtypes(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    dt = rules["dataset"]["dtypes"]
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
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df

def apply_cleaning(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    c = rules["cleaning"]
    if c.get("trim_strings", False):
        for col in df.select_dtypes(include="string").columns:
            df[col] = df[col].str.strip()
    if c.get("drop_duplicates", False):
        df = df.drop_duplicates()

    # clip ranges
    for col, rng in (c.get("clip_ranges") or {}).items():
        if col not in df.columns: 
            continue
        lo, hi = rng
        if lo is not None: df[col] = df[col].clip(lower=float(lo))
        if hi is not None: df[col] = df[col].clip(upper=float(hi))

    # column-specific null policy
    nullp = c.get("null_policy") or {}
    for col, policy in nullp.items():
        if col not in df.columns: 
            continue
        if policy == "drop_row":
            df = df[df[col].notna()]
        elif policy == "zero":
            df[col] = df[col].fillna(0)
        elif policy == "impute_median":
            df[col] = df[col].fillna(df[col].median())
    return df

# --------- outlier detection ----------
def flag_outliers(df: pd.DataFrame, rules: dict, target="weekly_sales") -> pd.DataFrame:
    o = rules["outliers"]
    df = df.copy()
    if target not in df.columns:
        df["is_outlier"] = False
        return df

    method = o["method"]; gcols = o.get("group_by") or []
    df["is_outlier"] = False

    def mark(group: pd.DataFrame) -> pd.DataFrame:
        s = group[target].astype(float)
        mask = pd.Series(False, index=s.index)
        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            th = o["threshold"]["iqr"]
            lb, ub = q1 - th * iqr, q3 + th * iqr
            mask = (s < lb) | (s > ub)
        elif method == "zscore":
            z = (s - s.mean()) / (s.std(ddof=0) or 1)
            mask = z.abs() > o["threshold"]["zscore"]
        elif method == "mad":
            med = s.median()
            mad = (s - med).abs().median() or 1
            z = 0.6745 * (s - med) / mad
            mask = z.abs() > o["threshold"]["mad"]
        group["is_outlier"] = mask

        # treatment
        if o["treat"] == "winsorize":
            lo, hi = s.quantile(.05), s.quantile(.95)
            group.loc[mask, target] = np.clip(group.loc[mask, target], lo, hi)
        elif o["treat"] == "remove":
            group = group.loc[~mask]
        return group

    if gcols:
        df = df.groupby(gcols, group_keys=False).apply(mark)
    else:
        df = mark(df)
    return df

# --------- KPI aggregation ----------
def compute_kpis(df: pd.DataFrame, rules: dict) -> Dict[str, Any]:
    req = rules["dataset"]["required_columns"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = coerce_dtypes(df, rules)
    df = apply_cleaning(df, rules)
    df = flag_outliers(df, rules, target="weekly_sales")

    # resample weekly
    freq = rules["kpis"]["time_freq"]
    dft = df.set_index("date").sort_index()
    # build agg map
    agg_map = {}
    for col, aggs in rules["kpis"]["aggregates"].items():
        for a in aggs:
            agg_map[f"{col}_{a}"] = (col, a)

    weekly = dft.resample(freq).agg(**agg_map).reset_index()

    # derived metrics (simple evaluator)
    for name, expr in (rules["kpis"].get("derived") or {}).items():
        # convert NULLIF(x,0) -> np.where(x==0, np.nan, x)
        safe = expr.replace("NULLIF(", "np.where(").replace(",0)", ",0, np.nan)")
        env = {"np": np, **{c: weekly[c] for c in weekly.columns}}
        weekly[name] = eval(safe, {}, env)

    by_region = df.groupby("region", as_index=False)["weekly_sales"].sum().rename(columns={"weekly_sales":"weekly_sales_sum"})
    by_dept   = df.groupby("department", as_index=False)["weekly_sales"].sum().rename(columns={"weekly_sales":"weekly_sales_sum"})

    # simple 4-week trend %
    if len(weekly) >= 4 and "weekly_sales_sum" in weekly.columns:
        arr = weekly["weekly_sales_sum"].tail(4).to_numpy(dtype=float)
        trend_4wk = float((arr[-1] - arr[0]) / max(arr[0], 1e-9))
    else:
        trend_4wk = np.nan

    result = {
        "shape": {"rows": int(len(df)), "cols": int(len(df.columns))},
        "kpis_weekly": weekly.to_dict(orient="list"),
        "by_region": by_region.to_dict(orient="list"),
        "by_department": by_dept.to_dict(orient="list"),
        "outliers": int(df["is_outlier"].sum()),
        "trend_4wk": trend_4wk,
    }
    return result
