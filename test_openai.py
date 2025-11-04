from openai import OpenAI
from dotenv import load_dotenv
import os

# Load the .env file automatically
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Test message: Say hello!"}]
    )
    print("✅ API connection successful!")
    print("Model response:", response.choices[0].message.content)
except Exception as e:
    print("❌ Connection failed:")
    print(e)
