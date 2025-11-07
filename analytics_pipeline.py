from __future__ import annotations
from pandas.api.types import is_integer_dtype
import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────────
# Rules loader
# ───────────────────────────────────────────────────────────────────────────────

def load_rules(path: str = "config/rules.yaml") -> dict:
    """Load YAML rules with fallback to defaults."""
    possible_paths = [path, Path(__file__).parent / path, Path(__file__).parent.parent / path, Path.cwd() / path, "rules.yaml"]
    
    for p in possible_paths:
        if Path(p).exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error loading {p}: {e}")
    
    logger.warning("⚠️ Rules file not found, using defaults")
    return _default_rules()


def _default_rules() -> dict:
    """Fallback rules."""
    return {
        "dataset": {
            "required_columns": ["date", "store", "department", "region", "weekly_sales", "transactions"],
            "dtypes": {"date": "datetime", "store": "string", "department": "string", "region": "category", "weekly_sales": "float", "transactions": "int"}
        },
        "cleaning": {
            "trim_strings": True,
            "drop_duplicates": True,
            "null_policy": {"weekly_sales": "drop_row", "transactions": "impute_median"},
            "clip_ranges": {"weekly_sales": [0, None], "transactions": [0, None]}
        },
        "outliers": {"method": "iqr", "group_by": "store", "k": 1.5, "treat": "flag"},
        "kpis": {
            "time_freq": "W",
            "aggregates": {"weekly_sales": ["sum"], "transactions": ["sum"]},
            "thresholds": {"growth": {"strong": 0.05, "stable": 0.0, "declining": -0.05}}
        }
    }


# ───────────────────────────────────────────────────────────────────────────────
# Validation & Cleaning (UNCHANGED - Working well)
# ───────────────────────────────────────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame, rules: dict) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate and clean dataframe according to rules."""
    report = {"initial_rows": len(df), "initial_cols": len(df.columns), "errors": [], "warnings": [], "info": [], "rows_removed": 0, "final_rows": 0}
    df_clean = df.copy()
    
    # Check required columns
    dataset_rules = rules.get("dataset", {})
    required = dataset_rules.get("required_columns", [])
    missing_cols = set(required) - set(df_clean.columns)
    
    if missing_cols:
        report["errors"].append(f"❌ Missing required columns: {', '.join(missing_cols)}")
        return df, report
    
    report["info"].append(f"✓ All required columns present")
    
    # Coerce dtypes
    try:
        df_clean = coerce_dtypes(df_clean, rules)
        report["info"].append("✓ Data types coerced successfully")
    except Exception as e:
        report["errors"].append(f"❌ Error coercing dtypes: {e}")
        return df, report
    
    # Date validation
    if "date" in df_clean.columns:
        df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")
        invalid_dates = df_clean["date"].isna().sum()
        
        if invalid_dates > 0:
            df_clean = df_clean.dropna(subset=["date"])
            report["warnings"].append(f"⚠️ Removed {invalid_dates} rows with invalid dates")
        
        if not df_clean.empty:
            date_range = f"{df_clean['date'].min().strftime('%Y-%m-%d')} to {df_clean['date'].max().strftime('%Y-%m-%d')}"
            report["info"].append(f"📅 Date range: {date_range}")
    
    # Apply cleaning
    initial_count = len(df_clean)
    try:
        df_clean = apply_cleaning(df_clean, rules)
        removed = initial_count - len(df_clean)
        if removed > 0:
            report["warnings"].append(f"⚠️ Cleaning removed {removed} rows ({removed/initial_count*100:.1f}%)")
        report["info"].append(f"✓ Applied cleaning rules")
    except Exception as e:
        report["errors"].append(f"❌ Error during cleaning: {e}")
        return df, report
    
    # Check suspicious patterns
    if "weekly_sales" in df_clean.columns:
        zero_sales = (df_clean["weekly_sales"] == 0).sum()
        if zero_sales > len(df_clean) * 0.1:
            report["warnings"].append(f"⚠️ {zero_sales} rows ({zero_sales/len(df_clean)*100:.1f}%) have zero sales")
        
        negative_sales = (df_clean["weekly_sales"] < 0).sum()
        if negative_sales > 0:
            report["errors"].append(f"❌ {negative_sales} rows have negative sales")
    
    report["rows_removed"] = report["initial_rows"] - len(df_clean)
    report["final_rows"] = len(df_clean)
    
    return df_clean, report


def coerce_dtypes(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Coerce column types."""
    df = df.copy()
    dt = rules.get("dataset", {}).get("dtypes", {})
    
    for col, typ in dt.items():
        if col not in df.columns:
            continue
        try:
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
        except Exception as e:
            logger.warning(f"Could not coerce {col} to {typ}: {e}")
    
    return df


def apply_cleaning(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Apply cleaning rules."""
    c = rules.get("cleaning", {})

    # Trim strings
    if c.get("trim_strings"):
        for col in df.select_dtypes(include="string").columns:
            df[col] = df[col].str.strip()

    # Drop duplicates
    if c.get("drop_duplicates"):
        before = len(df)
        df = df.drop_duplicates()
        if before > len(df):
            logger.info(f"Dropped {before - len(df)} duplicates")

    # Clip ranges
    for col, rng in (c.get("clip_ranges") or {}).items():
        if col in df.columns:
            lo, hi = rng
            if lo is not None:
                df[col] = df[col].clip(lower=float(lo))
            if hi is not None:
                df[col] = df[col].clip(upper=float(hi))

    # Null policy
    for col, policy in (c.get("null_policy") or {}).items():
        if col not in df.columns or df[col].isna().sum() == 0:
            continue

        if policy == "drop_row":
            df = df[df[col].notna()]
        elif policy == "zero":
            df[col] = df[col].fillna(0)
        elif policy == "impute_median":
            med = df[col].median(skipna=True)
            df[col] = df[col].fillna(med if pd.notna(med) else 0)

    return df


# ───────────────────────────────────────────────────────────────────────────────
# Outlier Detection (UNCHANGED)
# ───────────────────────────────────────────────────────────────────────────────

def flag_outliers_iqr(df: pd.DataFrame, col: str = "weekly_sales", group_by: Optional[str] = None, k: float = 1.5) -> pd.DataFrame:
    """Flag outliers using IQR method."""
    df = df.copy()
    df["is_outlier"] = False

    if group_by and group_by in df.columns:
        def _mark(g):
            s = g[col].dropna().astype(float)
            if len(s) < 4:
                return g
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lb, ub = q1 - k * iqr, q3 + k * iqr
            g.loc[~g[col].between(lb, ub), "is_outlier"] = True
            return g
        return df.groupby(group_by, group_keys=False).apply(_mark)

    s = df[col].dropna().astype(float)
    if len(s) >= 4:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lb, ub = q1 - k * iqr, q3 + k * iqr
        df.loc[~df[col].between(lb, ub), "is_outlier"] = True

    return df


# ───────────────────────────────────────────────────────────────────────────────
# NEW: Filter Application
# ───────────────────────────────────────────────────────────────────────────────

def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply store/department/region filters."""
    if not filters:
        return df
    
    result = df.copy()
    
    if filters.get("store") and filters["store"] != "All" and "store" in result.columns:
        result = result[result["store"] == filters["store"]]
    
    if filters.get("department") and filters["department"] != "All" and "department" in result.columns:
        result = result[result["department"] == filters["department"]]
    
    if filters.get("region") and filters["region"] != "All" and "region" in result.columns:
        result = result[result["region"] == filters["region"]]
    
    return result


# ───────────────────────────────────────────────────────────────────────────────
# NEW: Derived Fields (CRITICAL!)
# ───────────────────────────────────────────────────────────────────────────────

def _add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived fields before aggregation."""
    df = df.copy()
    
    # Average Transaction Value
    if {'transactions', 'weekly_sales'}.issubset(df.columns):
        df['avg_transaction_value'] = df['weekly_sales'] / df['transactions'].replace(0, np.nan)
    
    # Store Efficiency
    if 'avg_transaction_value' in df.columns and 'store' in df.columns:
        overall_avg = df['avg_transaction_value'].mean()
        if overall_avg > 0:
            store_avg = df.groupby('store')['avg_transaction_value'].transform('mean')
            df['store_efficiency_pct'] = ((store_avg / overall_avg) - 1) * 100
    
    # Week-over-Week
    if {'date', 'store', 'weekly_sales'}.issubset(df.columns):
        df = df.sort_values(['store', 'date'])
        df['wow_pct'] = df.groupby('store')['weekly_sales'].pct_change() * 100
    
    # Revenue Contribution
    if 'date' in df.columns:
        total = df.groupby('date')['weekly_sales'].transform('sum')
        df['revenue_contribution_pct'] = (df['weekly_sales'] / total) * 100
    
    return df


# ───────────────────────────────────────────────────────────────────────────────
# ENHANCED: Store Metrics
# ───────────────────────────────────────────────────────────────────────────────

def _compute_store_metrics(df: pd.DataFrame) -> dict:
    """Compute comprehensive store metrics."""
    if df.empty or 'store' not in df.columns:
        return {}
    
    stores = {}
    stores["total_sales"] = df.groupby("store")["weekly_sales"].sum().to_dict()
    stores["transactions"] = df.groupby("store")["transactions"].sum().to_dict()
    
    # Trends
    def compute_trend(group):
        group = group.sort_values('date')
        if len(group) < 8:
            return 0.0
        recent = group.tail(4)["weekly_sales"].sum()
        prior = group.tail(8).head(4)["weekly_sales"].sum()
        return float((recent - prior) / prior) if prior > 0 else 0.0
    
    if 'date' in df.columns:
        stores["trends"] = df.groupby("store").apply(compute_trend).to_dict()
    
    # Derived fields
    if 'avg_transaction_value' in df.columns:
        stores["avg_transaction_values"] = df.groupby("store")["avg_transaction_value"].mean().to_dict()
    
    if 'store_efficiency_pct' in df.columns:
        stores["efficiency_pct"] = df.groupby("store")["store_efficiency_pct"].mean().to_dict()
    
    if 'wow_pct' in df.columns:
        stores["week_over_week_pct"] = df.sort_values(['store', 'date']).groupby("store")["wow_pct"].last().to_dict()
    
    stores["count"] = len(stores["total_sales"])
    
    # Top/bottom
    if stores["total_sales"]:
        stores["top_store"] = max(stores["total_sales"], key=stores["total_sales"].get)
        stores["top_store_sales"] = float(stores["total_sales"][stores["top_store"]])
        stores["bottom_store"] = min(stores["total_sales"], key=stores["total_sales"].get)
        stores["bottom_store_sales"] = float(stores["total_sales"][stores["bottom_store"]])
    
    return stores


# ───────────────────────────────────────────────────────────────────────────────
# NEW: Regional Metrics
# ───────────────────────────────────────────────────────────────────────────────

def _compute_regional_metrics(df: pd.DataFrame) -> dict:
    """Compute regional metrics."""
    if df.empty or 'region' not in df.columns:
        return {}
    
    regional = {}
    
    for region in df['region'].unique():
        region_df = df[df['region'] == region]
        sales = float(region_df['weekly_sales'].sum())
        txns = int(region_df['transactions'].sum())
        
        # Trend
        trend = 0.0
        if 'date' in df.columns and len(region_df) >= 8:
            region_df = region_df.sort_values('date')
            recent = region_df.tail(4)['weekly_sales'].sum()
            prior = region_df.tail(8).head(4)['weekly_sales'].sum()
            if prior > 0:
                trend = float((recent - prior) / prior)
        
        regional[str(region)] = {
            "sales": sales,
            "transactions": txns,
            "trend_4wk": trend,
            "avg_transaction_value": float(sales / txns) if txns > 0 else 0.0
        }
    
    return regional


# ───────────────────────────────────────────────────────────────────────────────
# NEW: Department Metrics
# ───────────────────────────────────────────────────────────────────────────────

def _compute_department_metrics(df: pd.DataFrame) -> dict:
    """Compute department metrics with Pareto."""
    if df.empty or 'department' not in df.columns:
        return {}
    
    dept_sales = df.groupby("department")["weekly_sales"].sum().sort_values(ascending=False).to_dict()
    total = sum(dept_sales.values())
    
    # Cumulative
    cumsum = 0
    cumulative = {}
    for dept, sales in dept_sales.items():
        cumsum += sales
        cumulative[dept] = float(cumsum / total * 100)
    
    pareto_count = sum(1 for pct in cumulative.values() if pct <= 80)
    
    # Structure
    departments = {}
    for dept, sales in dept_sales.items():
        departments[dept] = {"sales": float(sales), "cumulative_pct": cumulative[dept]}
        
        # Trend
        if 'date' in df.columns:
            dept_df = df[df['department'] == dept]
            if len(dept_df) >= 8:
                dept_df = dept_df.sort_values('date')
                recent = dept_df.tail(4)['weekly_sales'].sum()
                prior = dept_df.tail(8).head(4)['weekly_sales'].sum()
                if prior > 0:
                    departments[dept]["trend_4wk"] = float((recent - prior) / prior)
    
    return {"departments": departments, "pareto_80_count": pareto_count, "total_count": len(departments)}


# ───────────────────────────────────────────────────────────────────────────────
# MAIN: KPI Computation (UPDATED)
# ───────────────────────────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame, rules: dict, filters: Optional[dict] = None) -> Dict[str, Any]:
    """
    Main KPI computation with filters and derived fields.
    
    CHANGES:
    - Added filters parameter
    - Added derived fields calculation
    - Enhanced store/regional/department outputs
    - Removed redundant code
    """
    if df is None or df.empty:
        return _empty_result()

    # Validate
    req = rules.get("dataset", {}).get("required_columns", [])
    if any(c not in df.columns for c in req):
        return _empty_result()

    # Clean
    try:
        df = coerce_dtypes(df, rules)
        df = apply_cleaning(df, rules)
    except Exception as e:
        logger.error(f"Error in data prep: {e}")
        return _empty_result()

    if df.empty:
        return _empty_result()

    # Apply filters
    if filters:
        df = _apply_filters(df, filters)
        if df.empty:
            return _empty_result()

    # Add derived fields
    df = _add_derived_fields(df)

    # Outliers
    out_cfg = rules.get("outliers", {}) or {}
    df = flag_outliers_iqr(df, col="weekly_sales", group_by=out_cfg.get("group_by", "store"), k=float(out_cfg.get("k", 1.5)))

    # Weekly aggregation
    dft = df.set_index("date").sort_index()
    weekly = dft.resample(rules.get("kpis", {}).get("time_freq", "W")).agg(
        weekly_sales_sum=("weekly_sales", "sum"),
        transactions_sum=("transactions", "sum")
    ).reset_index()

    # Trends
    trend_4wk = np.nan
    wow_last = np.nan
    
    if len(weekly) > 1:
        weekly = weekly.sort_values("date")
        weekly["wow_growth_pct"] = weekly["weekly_sales_sum"].pct_change()
        weekly["ma_4wk"] = weekly["weekly_sales_sum"].rolling(4, min_periods=1).mean()

        if len(weekly) >= 2:
            wow_last = float(weekly["wow_growth_pct"].iloc[-1])
        
        if len(weekly) >= 8:
            last_4 = weekly["weekly_sales_sum"].tail(4).mean()
            prev_4 = weekly["weekly_sales_sum"].iloc[-8:-4].mean()
            if prev_4 > 0:
                trend_4wk = float((last_4 - prev_4) / prev_4)

    # Aggregations
    stores = _compute_store_metrics(df)
    regional = _compute_regional_metrics(df)
    dept_results = _compute_department_metrics(df)

    # Build result
    return {
        "record_count": len(df),
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "kpis_weekly": weekly.to_dict(orient="list") if not weekly.empty else {},
        "stores": stores,
        "regional": regional,
        "departments": dept_results.get("departments", {}),
        "departments_meta": {"pareto_80_count": dept_results.get("pareto_80_count", 0), "total_count": dept_results.get("total_count", 0)},
        "outliers": int(df["is_outlier"].sum()) if "is_outlier" in df.columns else 0,
        "trend_4wk": float(trend_4wk) if pd.notna(trend_4wk) else None,
        "wow_last": float(wow_last) if pd.notna(wow_last) else None,
    }


def _empty_result() -> Dict[str, Any]:
    """Empty result structure."""
    return {"record_count": 0, "shape": {"rows": 0, "cols": 0}, "kpis_weekly": {}, "stores": {}, "regional": {}, "departments": {}, "departments_meta": {}, "outliers": 0, "trend_4wk": None, "wow_last": None}