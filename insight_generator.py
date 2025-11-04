import json, os
from openai import OpenAI
from dotenv import load_dotenv
import mlflow

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_insights(analysis: dict) -> str:
    """Use GPT to turn analysis output into a concise business narrative."""
    mlflow.set_experiment("data-to-insight")
    with mlflow.start_run(run_name="generate_insights"):
        prompt = f"""Act as a senior business analyst. You are given a structured analysis summary from a dataset.
Write:
1) 5 bullet key insights
2) Anomalies to double-click
3) 3 actionable recommendations
Keep it plain, specific, and reference concrete columns/metrics when possible.

ANALYSIS:
{json.dumps(analysis, indent=2)}
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = resp.choices[0].message.content
        # Log to MLflow for traceability
        mlflow.log_text(text, "insights/insights.md")
    return text
