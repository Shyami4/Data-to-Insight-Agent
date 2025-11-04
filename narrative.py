import json, os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = (
    "You are a concise business analyst. Use only the provided context. "
    "Do not invent numbers or facts. Write crisp, action-oriented bullets."
)

def draft_insights(result: dict, chart_descs: list) -> str:
    payload = {
        "rows": result["shape"]["rows"],
        "trend_4wk": result["trend_4wk"],
        "outliers": result["outliers"],
        "charts": chart_descs,
        "segments": {
            "region": result["by_region"],
            "department": result["by_department"]
        }
    }
    prompt = f"""
Context:
{json.dumps(payload, indent=2)}

Write:
1) 4–6 data-backed insights (reference metric/segment names).
2) 2 anomalies/risks to investigate.
3) 3 recommended actions for the next week.

Rules:
- Only reference numbers/segments present in Context.
- If trend_4wk is null, say trend is inconclusive.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content
