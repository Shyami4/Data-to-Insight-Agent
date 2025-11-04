import pandas as pd
import numpy as np
from scipy import stats
import mlflow
import os

def analyze_data(df: pd.DataFrame) -> dict:
    """Compute summary stats, simple correlations, and z-score anomalies.
    Logs basic metrics to MLflow.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("data-to-insight")
    with mlflow.start_run(run_name="analyze_data"):
        # Basic shape params
        mlflow.log_param("num_rows", int(df.shape[0]))
        mlflow.log_param("num_columns", int(df.shape[1]))
        num_df = df.select_dtypes(include=np.number)

        summary = num_df.describe().to_dict()
        correlations = num_df.corr(numeric_only=True).to_dict()

        anomalies = {}
        for col in num_df.columns:
            series = num_df[col].dropna()
            if series.shape[0] < 5:
                continue
            z = np.abs(stats.zscore(series))
            outliers = series[z > 3]
            if not outliers.empty:
                anomalies[col] = outliers.head(10).tolist()

        mlflow.log_metric("numeric_cols", len(num_df.columns))
        mlflow.log_metric("anomaly_cols", len(anomalies.keys()))

        result = {
            "summary": summary,
            "correlations": correlations,
            "anomalies": anomalies,
            "columns": list(df.columns),
            "rows": int(df.shape[0])
        }
        # Log as artifact for traceability
        import json, tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(result, f, indent=2)
            tmp = f.name
        mlflow.log_artifact(tmp, artifact_path="analysis")
    return result
