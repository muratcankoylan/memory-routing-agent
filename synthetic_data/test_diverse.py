"""Quick test of diverse generation with high temperature."""
import json
import random
import os
from dotenv import load_dotenv
load_dotenv()

import cohere

client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

INDUSTRIES = ["fintech startup", "healthcare SaaS", "e-commerce fashion"]
STARTERS = ["So about", "Following up on", "I've been thinking about"]

# Test 3 different categories with HIGH temperature
test_cases = [
    ("company.tools_config", "fintech startup", "growth hacker"),
    ("user.communication_style", "healthcare SaaS", "CMO"),
    ("none", "e-commerce fashion", "marketing manager")
]

for category, industry, role in test_cases:
    starter = random.choice(STARTERS)
    turns = random.randint(3, 6)
    
    if category == "none":
        prompt = f"""Create a UNMEMORABLE conversation between a {role} at a {industry} and AI.
Purely transactional - status check, scheduling, confirmation. NO specific details.
{turns} turns. Start with "{starter}..."
Return JSON: {{"conversation": [...], "labels": {{"categories": ["none"]}}}}"""
    else:
        prompt = f"""Create a marketing conversation for a {role} at a {industry}.
Must demonstrate: {category}
{turns} turns. Start with "{starter}..."
Be SPECIFIC with realistic details unique to {industry}.
Return JSON: {{"conversation": [...], "labels": {{"categories": ["{category}"]}}}}"""

    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.95,
        model="command-r-plus-08-2024",
        response_format={"type": "json_object"}
    )
    
    content = response.message.content[0].text
    data = json.loads(content)
    
    print(f"\n{'='*60}")
    print(f"Category: {category} | Industry: {industry}")
    print(f"Output categories: {data.get('labels', {}).get('categories', [])}")
    conv = data.get("conversation", [])
    if conv:
        first = conv[0]
        if isinstance(first, dict):
            print(f"First turn: {first.get('content', '')[:120]}...")
        else:
            print(f"First turn: {str(first)[:120]}...")

