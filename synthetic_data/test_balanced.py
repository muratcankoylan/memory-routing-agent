"""Quick test of balanced generation for underrepresented categories."""

import json
import os
from dotenv import load_dotenv
load_dotenv()

import cohere

client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

# Test with underrepresented categories
test_categories = ["company.tools_config", "company.knowledge_artifacts", "none"]

for category in test_categories:
    print(f"\n{'='*60}")
    print(f"Testing: {category}")
    print("="*60)
    
    if category == "none":
        prompt = """Generate a marketing conversation that has NO long-term memory value.

The conversation should be transactional, vague, or temporary.
Examples: checking status, scheduling, confirming receipt.

Generate 4 turns. Start mid-conversation (no greetings).

OUTPUT FORMAT (JSON only):
{
  "scenario_id": "none_001",
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "labels": {
    "categories": ["none"],
    "rationale": "..."
  }
}"""
    else:
        prompt = f"""Generate a marketing conversation that clearly demonstrates: {category}

The conversation MUST contain clear signals for this category.
4-6 turns, start mid-conversation (no greetings).

CRITICAL: The categories array MUST include "{category}".

OUTPUT FORMAT (JSON only):
{{
  "scenario_id": "{category.replace('.', '_')}_001",
  "conversation": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ],
  "labels": {{
    "categories": ["{category}"],
    "rationale": "..."
  }}
}}"""

    try:
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            model="command-r-plus-08-2024",
            response_format={"type": "json_object"}
        )
        
        content = response.message.content[0].text
        data = json.loads(content)
        
        output_cats = data.get("labels", {}).get("categories", [])
        print(f"Target: {category}")
        print(f"Output: {output_cats}")
        print(f"Match: {'YES' if category in output_cats else 'NO'}")
        
        if data.get("conversation"):
            print(f"First turn: {data['conversation'][0]['content'][:80]}...")
    except Exception as e:
        print(f"Error: {e}")

