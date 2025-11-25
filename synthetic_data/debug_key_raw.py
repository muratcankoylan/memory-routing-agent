import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("CO_API_KEY")

if key:
    print(f"Key raw: '{key}'")
    print(f"Length: {len(key)}")
else:
    print("Key NOT found.")

