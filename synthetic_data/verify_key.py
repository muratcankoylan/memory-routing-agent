import os
import cohere
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("CO_API_KEY")
print(f"Key: {key[:5]}... (len={len(key) if key else 0})")

try:
    client = cohere.ClientV2(api_key=key)
    resp = client.chat(model="command-r-plus", messages=[{"role": "user", "content": "hi"}])
    print("API Connection Success!")
except Exception as e:
    print(f"API Error: {e}")

