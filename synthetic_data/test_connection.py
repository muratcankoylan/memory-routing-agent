import os
import cohere
import time
from dotenv import load_dotenv

load_dotenv()

def test_connection():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("Error: COHERE_API_KEY not found.")
        return

    print(f"Testing connection with API Key: {api_key[:4]}...{api_key[-4:]}")
    client = cohere.ClientV2(api_key=api_key)
    
    print("Sending request to command-a-reasoning-08-2025...")
    start_time = time.time()
    try:
        response = client.chat(
            messages=[{"role": "user", "content": "Say 'Hello, World!'"}],
            model="command-a-reasoning-08-2025",
            thinking={"type": "enabled"},
            temperature=0.7
        )
        elapsed = time.time() - start_time
        print(f"Response received in {elapsed:.2f}s")
        print("Response object:", response)
        
        # Try to extract text
        if hasattr(response, 'message') and response.message.content:
            for block in response.message.content:
                if block.type == 'text':
                    print(f"Text content: {block.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_connection()

