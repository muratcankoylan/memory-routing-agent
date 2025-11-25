import os
from dotenv import load_dotenv
import cohere

load_dotenv()

key = os.getenv("CO_API_KEY")

if key:
    print(f"Key found. Length: {len(key)}")
    print(f"Prefix: {key[:4]}...")
    
    try:
        client = cohere.ClientV2(api_key=key)
        print("Client initialized.")
        # Simple test call
        print("Testing simple chat...")
        response = client.chat(
            model="command-r-plus", # Use a cheaper/standard model for quick test
            messages=[{"role": "user", "content": "Hello"}]
        )
        print("Response received!")
        print(response)
    except Exception as e:
        print(f"Error: {e}")
else:
    print("Key NOT found in environment.")

