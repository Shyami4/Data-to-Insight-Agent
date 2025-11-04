import os
import pandas as pd
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from analysis_engine import analyze_data
from insight_generator import generate_insights
from dotenv import load_dotenv

load_dotenv()


@tool("analyze_data_tool", return_direct=False)
def analyze_data_tool(file_path: str):
    """Analyze a CSV file on disk and return a structured summary dict."""
    df = pd.read_csv(file_path)
    return analyze_data(df)

@tool("summarize_trends_tool", return_direct=True)
def summarize_trends_tool(analysis: dict):
    """Turn the analysis dict into a human-readable insight report."""
    return generate_insights(analysis)

def build_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    tools = [analyze_data_tool, summarize_trends_tool]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent

if __name__ == "__main__":
    agent = build_agent()
    # Expect a local file path 'sample_data/sample_sales.csv'
    path = os.environ.get("SAMPLE_FILE", "sample_data/sample_sales.csv")
    out = agent.run(f"Analyze the file at path: {path}. Then use summarize_trends_tool to produce recommendations.")
    print(out)
