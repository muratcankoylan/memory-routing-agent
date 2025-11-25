"""Quick test of diverse generation."""
import json
import random
import os
from dotenv import load_dotenv
load_dotenv()

import cohere

client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

# Test one generation
category = "company.tools_config"
industry = "Series A fintech building a neobank"
persona = "a growth lead obsessed with metrics"
situation = "debugging why a campaign tanked"
tone = "frustrated"

prompt = f"""You are a world-class creative writer generating training data for an AI memory routing system.

Create a completely unique, realistic conversation between {persona} at a {industry} and their AI marketing assistant.

Context: They are {situation}. The tone is {tone}.

CATEGORY TO DEMONSTRATE: {category}
The conversation should involve tool setup, integrations, APIs, or workflow automation.

CREATIVE FREEDOM:
- Invent specific, realistic details (names, numbers, dates, products)
- The conversation can start anywhere - mid-thought, mid-project, mid-crisis
- Vary structure dramatically
- Include natural speech patterns
- Make it feel like eavesdropping on a real conversation

The ONLY hard requirement: the conversation must clearly demonstrate {category}.

Output as JSON:
{{"scenario_id": "unique_id", "conversation": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}], "labels": {{"categories": ["{category}"]}}, "metadata": {{"primary_category": "{category}", "industry": "{industry}"}}}}"""

print("Sending request...")
response = client.chat(
    messages=[{"role": "user", "content": prompt}],
    temperature=0.95,
    model="command-r-plus-08-2024",
    response_format={"type": "json_object"}
)

content = response.message.content[0].text
print("\n=== RAW RESPONSE ===")
print(content[:500])

data = json.loads(content)
print("\n=== PARSED ===")
print(f"Categories: {data.get('labels', {}).get('categories', [])}")
conv = data.get("conversation", [])
if conv:
    for i, turn in enumerate(conv[:4]):
        if isinstance(turn, dict):
            print(f"\n[{turn.get('role', 'unknown')}]: {turn.get('content', '')[:150]}...")
        else:
            print(f"\n[turn {i}]: {str(turn)[:150]}...")

