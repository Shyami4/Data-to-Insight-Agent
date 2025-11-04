import pandas as pd
from analysis_engine import analyze_data

def test_analyze_data_basic():
    df = pd.DataFrame({"a":[1,2,3,100], "b":[2,3,4,5]})
    out = analyze_data(df)
    assert "summary" in out and "anomalies" in out
