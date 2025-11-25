"""
Balanced Dataset Generation Script

This script generates a balanced training dataset with:
1. STRICT category enforcement - the model MUST output the target category
2. Equal distribution across all categories
3. Improved prompts for underrepresented categories
"""

import json
import random
import time
import sys
import asyncio
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import cohere
from dotenv import load_dotenv

load_dotenv()

# BALANCED DISTRIBUTION - Equal weight for all categories
BALANCED_DISTRIBUTION = {
    "company.brand_core": 80,
    "company.strategic_signatures": 80,
    "company.knowledge_artifacts": 80,
    "company.business_priorities": 80,
    "company.tools_config": 80,
    "company.performance_context": 80,
    "user.communication_style": 80,
    "user.strategic_approach": 80,
    "user.role_context": 80,
    "user.workflow_patterns": 80,
    "user.session_history": 80,
    "user.interaction_preferences": 80,
    "none": 80,
}

# Category-specific examples and signals for better generation
CATEGORY_EXAMPLES = {
    "company.brand_core": {
        "description": "Brand voice, values, positioning, visual identity, tone guidelines",
        "example_signals": [
            "Our brand voice is warm and conversational",
            "We always use sentence case for headlines",
            "Our primary color is #2563EB",
            "We never use corporate jargon",
            "Our tagline is 'Simplify Everything'"
        ],
        "example_conversation": "USER: Remember, our brand personality is 'friendly expert' - knowledgeable but approachable."
    },
    "company.strategic_signatures": {
        "description": "Decision frameworks, strategic heuristics, recurring patterns in how the company operates",
        "example_signals": [
            "We always prioritize retention over acquisition",
            "Our 80/20 rule: 80% proven tactics, 20% experiments",
            "We never launch without A/B testing",
            "Customer lifetime value drives all decisions"
        ],
        "example_conversation": "USER: Our strategic principle is 'land and expand' - start small with enterprises then grow."
    },
    "company.knowledge_artifacts": {
        "description": "Style guides, playbooks, SOPs, documented processes, templates",
        "example_signals": [
            "Here's our content style guide",
            "The campaign playbook says...",
            "According to our SOP for launches",
            "Our template for proposals includes..."
        ],
        "example_conversation": "USER: I'm attaching our updated brand guidelines PDF. Make sure all content follows section 3.2."
    },
    "company.business_priorities": {
        "description": "Quarterly goals, seasonal campaigns, current OKRs, active initiatives",
        "example_signals": [
            "Q4 focus is enterprise expansion",
            "This quarter's target is 500 MQLs",
            "Holiday campaign launches December 1st",
            "We're prioritizing APAC market this quarter"
        ],
        "example_conversation": "USER: For Q1, we're shifting focus entirely to the SMB segment. All campaigns should target companies under 100 employees."
    },
    "company.tools_config": {
        "description": "Integrations, API keys, workflow settings, tool configurations",
        "example_signals": [
            "The Slack webhook URL is...",
            "Configure HubSpot to sync with...",
            "The API key for analytics is...",
            "Set up the Zapier integration to..."
        ],
        "example_conversation": "USER: Here's the API key for our analytics dashboard: sk-xxx-123. Make sure it syncs every 6 hours."
    },
    "company.performance_context": {
        "description": "Campaign metrics, retrospectives, learnings, performance data",
        "example_signals": [
            "Last campaign had 24% open rate",
            "CTR improved by 15% after the redesign",
            "The retrospective showed we need more testing",
            "Conversion rate dropped after the price change"
        ],
        "example_conversation": "USER: The email campaign results are in: 28% open rate, 4.2% CTR. That's our best performance this year."
    },
    "user.communication_style": {
        "description": "Preferred tone, verbosity, format expectations, writing style",
        "example_signals": [
            "I prefer bullet points over paragraphs",
            "Keep responses under 200 words",
            "Use casual, friendly tone with me",
            "I like data-driven explanations"
        ],
        "example_conversation": "USER: Just so you know, I prefer concise bullet points. No need for lengthy explanations with me."
    },
    "user.strategic_approach": {
        "description": "Personal priorities, success definitions, decision-making style",
        "example_signals": [
            "I always prioritize speed over perfection",
            "My philosophy is test fast, fail fast",
            "I measure success by customer feedback",
            "I believe in data-driven decisions only"
        ],
        "example_conversation": "USER: My approach is always 'done is better than perfect'. I'd rather ship and iterate."
    },
    "user.role_context": {
        "description": "Title, scope, decision authority, reporting structure",
        "example_signals": [
            "As VP of Marketing, I approve all campaigns",
            "I report directly to the CMO",
            "My budget authority is up to $50k",
            "I manage a team of 12 marketers"
        ],
        "example_conversation": "USER: Just for context, I'm the Director of Growth and I have final say on all acquisition campaigns."
    },
    "user.workflow_patterns": {
        "description": "Review cadence, collaboration norms, meeting schedules",
        "example_signals": [
            "I review drafts every Monday morning",
            "Don't send me anything on Fridays",
            "I prefer async communication via Slack",
            "Weekly sync is Tuesdays at 2pm"
        ],
        "example_conversation": "USER: My review schedule is Monday mornings only. Anything sent Friday won't be seen until next week."
    },
    "user.session_history": {
        "description": "Immediate context, recent asks, current working session",
        "example_signals": [
            "As we discussed yesterday...",
            "Continuing from our last conversation",
            "The proposal we started earlier",
            "Following up on the draft you sent"
        ],
        "example_conversation": "USER: Let's pick up where we left off yesterday on the Johnson account proposal."
    },
    "user.interaction_preferences": {
        "description": "Coaching style, feedback expectations, collaboration preferences",
        "example_signals": [
            "I want you to push back on my ideas",
            "Give me options, not just one answer",
            "Be direct with feedback, don't sugarcoat",
            "I prefer you ask clarifying questions"
        ],
        "example_conversation": "USER: I want you to challenge my assumptions. If you think I'm wrong, tell me directly."
    },
    "none": {
        "description": "Transactional, vague, or temporary content with no memory value",
        "example_signals": [
            "What time is the meeting?",
            "Can you check the status?",
            "Just confirming receipt",
            "Quick question about the attachment"
        ],
        "example_conversation": "USER: Hey, what's the status on that thing we discussed? Just checking in."
    }
}

class BalancedDataGenerator:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY not found")
        self.client = cohere.ClientV2(api_key=self.api_key)
        self.model = "command-r-plus-08-2024"
    
    def _extract_text(self, response) -> Optional[str]:
        if not response or not getattr(response, "message", None):
            return None
        blocks = getattr(response.message, "content", []) or []
        for block in blocks:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                return text
        return None
    
    def generate_for_category(self, category: str, max_retries: int = 3) -> Optional[Dict]:
        """Generate a conversation that MUST contain the specified category."""
        
        cat_info = CATEGORY_EXAMPLES.get(category, {})
        description = cat_info.get("description", category)
        example_signals = cat_info.get("example_signals", [])
        example_conv = cat_info.get("example_conversation", "")
        
        # Build a very specific prompt
        if category == "none":
            prompt = f"""Generate a realistic marketing conversation that has NO long-term memory value.

The conversation should be:
- Transactional (checking status, scheduling, confirming)
- Vague or generic (no specific details worth remembering)
- Temporary (only relevant for this moment)

Examples of "none" conversations:
- "What time is the meeting tomorrow?"
- "Just confirming you received the file"
- "Quick status check on the project"
- "Can you resend that link?"

Generate a 4-6 turn conversation between USER and ASSISTANT.
Start mid-conversation (no greetings).

OUTPUT FORMAT (JSON only):
{{
  "scenario_id": "none_{random.randint(100,999)}",
  "conversation": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ],
  "labels": {{
    "categories": ["none"],
    "persistence_horizon": "short",
    "memory_scope": "none",
    "rationale": "This conversation is transactional/temporary with no memory value"
  }},
  "metadata": {{
    "primary_category": "none",
    "turn_count": 4
  }}
}}"""
        else:
            prompt = f"""Generate a marketing conversation that clearly demonstrates the category: {category}

CATEGORY DEFINITION:
{description}

SIGNALS THAT INDICATE THIS CATEGORY:
{chr(10).join(f"- {s}" for s in example_signals[:4])}

EXAMPLE UTTERANCE:
{example_conv}

REQUIREMENTS:
1. The conversation MUST contain clear signals for {category}
2. The USER should explicitly state information that maps to this category
3. Make it natural and realistic - embed the signals organically
4. 4-6 turns, start mid-conversation (no greetings)
5. Include specific, concrete details (names, numbers, dates)

CRITICAL: The output categories array MUST include "{category}" as the primary category.
You may include 1 additional category if naturally present, but {category} MUST be there.

OUTPUT FORMAT (JSON only):
{{
  "scenario_id": "{category.replace('.', '_')}_{random.randint(100,999)}",
  "conversation": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ],
  "labels": {{
    "categories": ["{category}"],
    "persistence_horizon": "long|medium|short",
    "memory_scope": "company|user",
    "rationale": "Explanation of why {category} applies"
  }},
  "metadata": {{
    "primary_category": "{category}",
    "turn_count": 4
  }}
}}"""

        for attempt in range(max_retries):
            try:
                response = self.client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    model=self.model,
                    response_format={"type": "json_object"}
                )
                
                content = self._extract_text(response)
                if not content:
                    continue
                
                # Clean JSON
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                
                data = json.loads(content.strip())
                
                # VALIDATE: Ensure target category is present
                categories = data.get("labels", {}).get("categories", [])
                if category.lower() not in [c.lower() for c in categories]:
                    print(f"  Warning: Target {category} not in output {categories}. Retrying...")
                    continue
                
                # Clean: Remove "none" if other categories exist
                if len(categories) > 1 and "none" in [c.lower() for c in categories]:
                    data["labels"]["categories"] = [c for c in categories if c.lower() != "none"]
                
                return data
                
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}")
                time.sleep(5 * (attempt + 1))
        
        return None


async def generate_balanced_dataset(output_dir: str = "synthetic_data", target_per_category: int = 80):
    """Generate a balanced dataset with equal examples per category."""
    
    os.makedirs(output_dir, exist_ok=True)
    generator = BalancedDataGenerator()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/balanced_dataset_{timestamp}.jsonl"
    log_file = f"{output_dir}/balanced_generation_log_{timestamp}.txt"
    
    all_data = []
    category_counts = {cat: 0 for cat in BALANCED_DISTRIBUTION.keys()}
    
    print("=" * 70, flush=True)
    print("BALANCED DATASET GENERATION", flush=True)
    print("=" * 70, flush=True)
    print(f"Target per category: {target_per_category}", flush=True)
    print(f"Total categories: {len(BALANCED_DISTRIBUTION)}", flush=True)
    print(f"Expected total: {target_per_category * len(BALANCED_DISTRIBUTION)}", flush=True)
    print(flush=True)
    
    with open(log_file, "w") as log:
        log.write(f"Balanced Generation Started: {timestamp}\n")
        log.write(f"Target per category: {target_per_category}\n\n")
        
        for category in BALANCED_DISTRIBUTION.keys():
            print(f"\n--- Generating {target_per_category} examples for: {category} ---", flush=True)
            log.write(f"\n=== {category} ===\n")
            log.flush()
            
            for i in range(target_per_category):
                result = generator.generate_for_category(category)
                
                if result:
                    all_data.append(result)
                    category_counts[category] += 1
                    
                    # Save incrementally
                    with open(output_file, "a") as f:
                        f.write(json.dumps(result) + "\n")
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Progress: {i+1}/{target_per_category}", flush=True)
                        log.write(f"  {i+1}/{target_per_category} complete\n")
                        log.flush()
                else:
                    print(f"  Failed: {i+1}", flush=True)
                    log.write(f"  Failed to generate example {i+1}\n")
                    log.flush()
                
                # Rate limiting
                await asyncio.sleep(0.5)
            
            print(f"  Completed: {category_counts[category]}/{target_per_category}", flush=True)
    
    # Final summary
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nCategory Distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_data) * 100 if all_data else 0
        print(f"  {cat:<40} {count:>4} ({pct:.1f}%)")
    
    print(f"\nTotal examples: {len(all_data)}")
    print(f"Output file: {output_file}")
    
    return output_file


if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    asyncio.run(generate_balanced_dataset(target_per_category=target))

