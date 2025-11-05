# narrative.py
import os, json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = (
    "You are a concise business analyst. Use only the provided context. "
    "Do not invent numbers or facts. Write crisp, action-oriented bullets."
)

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def draft_insights(result: dict, chart_descs: list) -> str:
    # Build a minimal, robust payload (handles missing keys)
    rows = int(result.get("shape", {}).get("rows", 0))
    trend_raw = result.get("trend_4wk")
    trend_4wk = None if trend_raw is None or (isinstance(trend_raw, float) and np.isnan(trend_raw)) else float(trend_raw)
    outliers = int(result.get("outliers", 0))

    segments = {
        "region": result.get("by_region", {}),
        "department": result.get("by_department", {}),
    }

    payload = {
        "rows": rows,
        "trend_4wk": trend_4wk,
        "outliers": outliers,
        "charts": chart_descs,
        "segments": segments,
    }

    prompt = f"""
Context:
{json.dumps(payload, indent=2)}

Write:
1) 4–6 data-backed insights (reference metric/segment names available).
2) 2 anomalies/risks to investigate.
3) 3 recommended actions for the next week.

Rules:
- Only reference numbers/segments present in Context.
- If trend_4wk is null, say trend is inconclusive.
- Be concise. Bulleted output preferred.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content
