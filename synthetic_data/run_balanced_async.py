"""
Balanced Dataset Generation with Concurrent API Calls

Generates 10 items simultaneously per batch for faster generation.
"""

import json
import random
import time
import sys
import asyncio
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import cohere
from dotenv import load_dotenv

load_dotenv()

# Target counts per category (balanced)
CATEGORY_TARGETS = {
    "company.brand_core": 77,
    "company.strategic_signatures": 77,
    "company.knowledge_artifacts": 77,
    "company.business_priorities": 77,
    "company.tools_config": 77,
    "company.performance_context": 77,
    "user.communication_style": 77,
    "user.strategic_approach": 77,
    "user.role_context": 77,
    "user.workflow_patterns": 77,
    "user.session_history": 77,
    "user.interaction_preferences": 77,
    "none": 77,
}

CATEGORY_EXAMPLES = {
    "company.brand_core": {
        "signals": ["brand voice is warm", "primary color is #2563EB", "never use jargon", "tagline is..."],
    },
    "company.strategic_signatures": {
        "signals": ["always prioritize retention", "80/20 rule", "never launch without testing"],
    },
    "company.knowledge_artifacts": {
        "signals": ["style guide says", "playbook recommends", "SOP for launches", "template includes"],
    },
    "company.business_priorities": {
        "signals": ["Q4 focus is", "this quarter's target", "holiday campaign", "prioritizing APAC"],
    },
    "company.tools_config": {
        "signals": ["Slack webhook URL", "HubSpot sync", "API key is", "Zapier integration"],
    },
    "company.performance_context": {
        "signals": ["24% open rate", "CTR improved by", "retrospective showed", "conversion dropped"],
    },
    "user.communication_style": {
        "signals": ["prefer bullet points", "keep it under 200 words", "casual tone", "data-driven"],
    },
    "user.strategic_approach": {
        "signals": ["prioritize speed over perfection", "test fast fail fast", "customer feedback"],
    },
    "user.role_context": {
        "signals": ["As VP of Marketing", "report to CMO", "budget authority up to", "manage team of"],
    },
    "user.workflow_patterns": {
        "signals": ["review drafts Monday", "don't send Friday", "async via Slack", "weekly sync Tuesday"],
    },
    "user.session_history": {
        "signals": ["as we discussed yesterday", "continuing from last", "proposal we started"],
    },
    "user.interaction_preferences": {
        "signals": ["push back on my ideas", "give me options", "be direct", "ask clarifying questions"],
    },
    "none": {
        "signals": ["what time is meeting", "checking status", "confirming receipt", "quick question"],
    },
}


class BalancedAsyncGenerator:
    def __init__(self):
        self.api_key = os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY not found")
        self.client = cohere.ClientV2(api_key=self.api_key)
        self.model = "command-r-plus-08-2024"
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    def _extract_text(self, response) -> Optional[str]:
        if not response or not getattr(response, "message", None):
            return None
        blocks = getattr(response.message, "content", []) or []
        for block in blocks:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                return text
        return None
    
    def _generate_sync(self, category: str) -> Optional[Dict]:
        """Synchronous generation for a single category."""
        signals = CATEGORY_EXAMPLES.get(category, {}).get("signals", [])
        signals_text = "\n".join(f"- {s}" for s in signals[:4])
        
        if category == "none":
            prompt = f"""Generate a marketing conversation with NO long-term memory value.
Transactional, vague, or temporary only. Examples: status check, scheduling, confirming.
4-6 turns, no greetings, start mid-conversation.

OUTPUT (JSON only):
{{"scenario_id": "none_{random.randint(100,999)}", "conversation": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}], "labels": {{"categories": ["none"], "persistence_horizon": "short", "memory_scope": "none", "rationale": "..."}}, "metadata": {{"primary_category": "none", "turn_count": 4}}}}"""
        else:
            prompt = f"""Generate a marketing conversation demonstrating: {category}

SIGNALS FOR THIS CATEGORY:
{signals_text}

REQUIREMENTS:
1. MUST contain clear signals for {category}
2. 4-6 turns, no greetings, start mid-conversation
3. Include specific details (names, numbers, dates)

CRITICAL: categories array MUST include "{category}"

OUTPUT (JSON only):
{{"scenario_id": "{category.replace('.', '_')}_{random.randint(100,999)}", "conversation": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}], "labels": {{"categories": ["{category}"], "persistence_horizon": "long", "memory_scope": "company", "rationale": "..."}}, "metadata": {{"primary_category": "{category}", "turn_count": 4}}}}"""

        try:
            response = self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                model=self.model,
                response_format={"type": "json_object"}
            )
            
            content = self._extract_text(response)
            if not content:
                return None
            
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            data = json.loads(content.strip())
            
            # Validate target category is present
            categories = data.get("labels", {}).get("categories", [])
            if category.lower() not in [c.lower() for c in categories]:
                return None
            
            # Clean: Remove "none" if other categories exist
            if len(categories) > 1 and "none" in [c.lower() for c in categories]:
                data["labels"]["categories"] = [c for c in categories if c.lower() != "none"]
            
            return data
            
        except Exception as e:
            return None
    
    async def generate_batch(self, categories: List[str]) -> List[Dict]:
        """Generate a batch of items concurrently."""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self.executor, self._generate_sync, cat)
            for cat in categories
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, dict)]


async def run_balanced_generation_async():
    """Run balanced generation with concurrent batches."""
    
    generator = BalancedAsyncGenerator()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"synthetic_data/balanced_dataset_{timestamp}.jsonl"
    
    # Track progress per category
    category_counts = {cat: 0 for cat in CATEGORY_TARGETS}
    all_data = []
    
    print("=" * 70, flush=True)
    print("BALANCED CONCURRENT DATASET GENERATION", flush=True)
    print("=" * 70, flush=True)
    print(f"Target per category: 77", flush=True)
    print(f"Total categories: {len(CATEGORY_TARGETS)}", flush=True)
    print(f"Expected total: {77 * len(CATEGORY_TARGETS)}", flush=True)
    print(f"Batch size: 10 concurrent requests", flush=True)
    print(flush=True)
    
    batch_num = 0
    
    while True:
        # Find categories that still need examples
        needed = []
        for cat, target in CATEGORY_TARGETS.items():
            remaining = target - category_counts[cat]
            needed.extend([cat] * min(remaining, 2))  # Up to 2 per category per batch
        
        if not needed:
            break
        
        # Take up to 10 for this batch
        batch_categories = needed[:10]
        batch_num += 1
        
        print(f"\n[Batch {batch_num}] Generating {len(batch_categories)} items...", flush=True)
        
        results = await generator.generate_batch(batch_categories)
        
        # Process results
        for result in results:
            if result:
                primary = result.get("metadata", {}).get("primary_category") or \
                         result.get("labels", {}).get("categories", ["unknown"])[0]
                
                if primary in category_counts:
                    category_counts[primary] += 1
                    all_data.append(result)
                    
                    # Save incrementally
                    with open(output_file, "a") as f:
                        f.write(json.dumps(result) + "\n")
        
        # Progress report
        total_done = sum(category_counts.values())
        total_target = sum(CATEGORY_TARGETS.values())
        print(f"  Success: {len(results)}/{len(batch_categories)} | Total: {total_done}/{total_target}", flush=True)
        
        # Show category progress every 10 batches
        if batch_num % 10 == 0:
            print("\n  Category Progress:", flush=True)
            for cat, count in sorted(category_counts.items()):
                target = CATEGORY_TARGETS[cat]
                bar = "█" * (count * 20 // target) + "░" * (20 - count * 20 // target)
                print(f"    {cat:<35} [{bar}] {count}/{target}", flush=True)
        
        # Rate limit: wait 3 seconds between batches
        await asyncio.sleep(3)
    
    # Final summary
    print("\n" + "=" * 70, flush=True)
    print("GENERATION COMPLETE", flush=True)
    print("=" * 70, flush=True)
    print(f"\nFinal Distribution:", flush=True)
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_data) * 100 if all_data else 0
        print(f"  {cat:<40} {count:>4} ({pct:.1f}%)", flush=True)
    
    print(f"\nTotal examples: {len(all_data)}", flush=True)
    print(f"Output file: {output_file}", flush=True)
    
    return output_file


if __name__ == "__main__":
    asyncio.run(run_balanced_generation_async())

