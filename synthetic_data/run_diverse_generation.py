"""
World-Class Diverse Dataset Generation - 20 Concurrent API Calls per Batch

Key features:
- 20 API calls simultaneously per batch
- Wait for batch to complete, then next batch
- Temperature 0.95 for maximum diversity
- No templates, maximum creative freedom
"""

import json
import random
import os
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import cohere
from dotenv import load_dotenv

load_dotenv()

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

INDUSTRIES = [
    "Series A fintech building a neobank", "hospital network digitizing patient intake",
    "DTC sneaker brand scaling to retail", "industrial valve manufacturer going digital",
    "K-12 tutoring platform expanding to Asia", "commercial real estate analytics startup",
    "ghost kitchen aggregator in NYC", "enterprise zero-trust security vendor",
    "luxury cruise line post-pandemic", "connected fitness hardware company",
    "immigration law firm automating visas", "recruiting platform for nurses",
    "pet insurance disruptor", "last-mile drone delivery startup",
    "indie game studio with a viral hit", "podcast network monetizing premium content",
    "EV charging network operator", "solar panel installer franchise",
    "modular home construction startup", "veterinary telehealth platform",
    "wine subscription service", "corporate wellness SaaS",
    "NFT marketplace pivoting to digital art", "AI code review tool for enterprises",
    "climate risk analytics for insurers", "restaurant POS system provider",
    "online therapy platform", "B2B payments infrastructure",
    "influencer marketing agency", "smart home security company"
]

PERSONAS = [
    "a stressed CMO preparing for board review",
    "a junior marketing coordinator on their first campaign",
    "a VP who just joined from a competitor",
    "a founder wearing multiple hats",
    "a seasoned brand director with 20 years experience",
    "a growth lead obsessed with metrics",
    "a creative director frustrated with process",
    "a demand gen manager under pressure to hit pipeline",
    "a content strategist building a new team",
    "a marketing ops person drowning in tools",
    "a product marketer launching next week",
    "an email specialist optimizing deliverability",
    "a social media manager handling a PR crisis",
    "a field marketer planning regional events",
    "a partner marketing lead negotiating co-marketing",
    "an analyst presenting attribution findings"
]

SITUATIONS = [
    "in the middle of a heated planning session",
    "wrapping up a long day before vacation",
    "preparing for a last-minute executive ask",
    "debugging why a campaign tanked",
    "celebrating a successful launch",
    "onboarding after joining last week",
    "dealing with budget cuts",
    "scaling something that unexpectedly worked",
    "cleaning up a predecessor's mess",
    "trying to align with a difficult stakeholder"
]

TONES = ["urgent", "casual", "frustrated", "excited", "methodical", "skeptical", "collaborative", "directive"]

CATEGORY_HINTS = {
    "company.brand_core": "The conversation should naturally surface brand identity elements - could be voice, visuals, values, positioning, or personality.",
    "company.strategic_signatures": "The conversation should reveal how this company makes decisions - their frameworks, principles, or recurring patterns.",
    "company.knowledge_artifacts": "The conversation should reference internal documentation - guides, playbooks, templates, or SOPs.",
    "company.business_priorities": "The conversation should touch on current goals, quarterly targets, or active initiatives.",
    "company.tools_config": "The conversation should involve tool setup, integrations, APIs, or workflow automation.",
    "company.performance_context": "The conversation should discuss metrics, campaign results, or performance learnings.",
    "user.communication_style": "The user should express how they prefer to receive information - format, length, tone, or style.",
    "user.strategic_approach": "The user should reveal their personal philosophy, priorities, or decision-making style.",
    "user.role_context": "The user should mention their role, responsibilities, authority, or team structure.",
    "user.workflow_patterns": "The user should describe their schedule, review process, or collaboration preferences.",
    "user.session_history": "The conversation should reference recent context, ongoing work, or previous discussions.",
    "user.interaction_preferences": "The user should express how they want the AI to behave - proactivity, feedback style, or coaching level.",
    "none": "The conversation should be purely transactional with nothing worth remembering long-term."
}


class ConcurrentGenerator:
    def __init__(self):
        self.api_key = os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY not found")
        self.client = cohere.ClientV2(api_key=self.api_key)
        self.model = "command-r-plus-08-2024"
        self.executor = ThreadPoolExecutor(max_workers=20)
    
    def _extract_text(self, response) -> Optional[str]:
        if not response or not getattr(response, "message", None):
            return None
        blocks = getattr(response.message, "content", []) or []
        for block in blocks:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                return text
        return None
    
    def _generate_one(self, category: str) -> Optional[Dict]:
        """Generate a single example with maximum creativity."""
        
        industry = random.choice(INDUSTRIES)
        persona = random.choice(PERSONAS)
        situation = random.choice(SITUATIONS)
        tone = random.choice(TONES)
        turns = random.randint(3, 10)
        hint = CATEGORY_HINTS.get(category, "")
        
        if category == "none":
            prompt = f"""You are a creative writer generating training data for an AI memory system.

Create a completely realistic conversation between {persona} at a {industry} and their AI marketing assistant.

Context: They are {situation}. The tone is {tone}.

THIS CONVERSATION MUST BE FORGETTABLE - nothing worth storing in long-term memory:
- Quick status checks, scheduling, or confirmations
- Vague questions without actionable details
- Chitchat or temporary context that expires immediately

Be creative. Make it feel real. No templates. Surprise me.

Output as JSON with this structure:
{{"scenario_id": "unique_id", "conversation": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}], "labels": {{"categories": ["none"], "persistence_horizon": "short", "memory_scope": "none", "rationale": "why this is unmemorable"}}, "metadata": {{"primary_category": "none", "turn_count": {turns}, "industry": "{industry}"}}}}"""

        else:
            prompt = f"""You are a world-class creative writer generating training data for an AI memory routing system.

Create a completely unique, realistic conversation between {persona} at a {industry} and their AI marketing assistant.

Context: They are {situation}. The tone is {tone}.

CATEGORY TO DEMONSTRATE: {category}
{hint}

CREATIVE FREEDOM:
- Invent specific, realistic details (names, numbers, dates, products)
- The conversation can start anywhere - mid-thought, mid-project, mid-crisis
- Vary structure dramatically - could be rapid-fire, could be detailed
- Include natural speech patterns, interruptions, tangents
- Make it feel like eavesdropping on a real conversation
- {turns} turns, but quality over quantity

The ONLY hard requirement: the conversation must clearly demonstrate {category}.

Output as JSON:
{{"scenario_id": "unique_id", "conversation": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}], "labels": {{"categories": ["{category}"], "persistence_horizon": "long/medium/short", "memory_scope": "{category.split('.')[0]}", "rationale": "why this fits {category}"}}, "metadata": {{"primary_category": "{category}", "turn_count": {turns}, "industry": "{industry}"}}}}"""

        try:
            response = self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.95,
                model=self.model,
                response_format={"type": "json_object"}
            )
            
            content = self._extract_text(response)
            if not content:
                return None
            
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            data = json.loads(content.strip())
            
            categories = data.get("labels", {}).get("categories", [])
            if category.lower() not in [c.lower() for c in categories]:
                return None
            
            if len(categories) > 1 and "none" in [c.lower() for c in categories]:
                data["labels"]["categories"] = [c for c in categories if c.lower() != "none"]
            
            return data
            
        except Exception as e:
            return None
    
    async def generate_batch_concurrent(self, categories: List[str]) -> List[Dict]:
        """Generate 20 items concurrently."""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self.executor, self._generate_one, cat)
            for cat in categories
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, dict)]


async def run_generation():
    generator = ConcurrentGenerator()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"synthetic_data/diverse_dataset_{timestamp}.jsonl"
    
    category_counts = {cat: 0 for cat in CATEGORY_TARGETS}
    all_data = []
    
    print("=" * 70, flush=True)
    print("WORLD-CLASS DATASET GENERATION (20 Concurrent)", flush=True)
    print("=" * 70, flush=True)
    print(f"Batch size: 20 concurrent API calls", flush=True)
    print(f"Temperature: 0.95", flush=True)
    print(f"Target: 77 per category x 13 = 1001 total", flush=True)
    print(f"Output: {output_file}", flush=True)
    print("=" * 70, flush=True)
    
    batch_num = 0
    start_time = datetime.now()
    
    while True:
        # Build list of needed categories
        needed = []
        for cat, target in CATEGORY_TARGETS.items():
            remaining = target - category_counts[cat]
            if remaining > 0:
                needed.extend([cat] * min(remaining, 3))  # Up to 3 per category per batch
        
        if not needed:
            break
        
        random.shuffle(needed)
        batch_categories = needed[:20]  # 20 concurrent
        batch_num += 1
        
        print(f"\n[Batch {batch_num}] Launching 20 concurrent requests...", flush=True)
        batch_start = datetime.now()
        
        results = await generator.generate_batch_concurrent(batch_categories)
        
        batch_time = (datetime.now() - batch_start).seconds
        
        for result in results:
            if result:
                primary = result.get("metadata", {}).get("primary_category") or \
                         result.get("labels", {}).get("categories", ["unknown"])[0]
                
                if primary in category_counts:
                    category_counts[primary] += 1
                    all_data.append(result)
                    
                    with open(output_file, "a") as f:
                        f.write(json.dumps(result) + "\n")
                    
                    conv = result.get("conversation", [])
                    if conv and len(conv) > 0:
                        first_msg = conv[0].get("content", "") if isinstance(conv[0], dict) else str(conv[0])
                        print(f"  [{primary}] {first_msg[:60]}...", flush=True)
        
        total_done = sum(category_counts.values())
        total_target = sum(CATEGORY_TARGETS.values())
        elapsed = (datetime.now() - start_time).seconds
        rate = total_done / max(elapsed, 1) * 60
        eta = (total_target - total_done) / max(rate, 0.1)
        
        print(f"  Batch: {len(results)}/20 success in {batch_time}s | Total: {total_done}/{total_target} | Rate: {rate:.1f}/min | ETA: {eta:.0f}min", flush=True)
        
        # Progress every 10 batches
        if batch_num % 10 == 0:
            print("\n  === Category Breakdown ===", flush=True)
            for cat in sorted(category_counts.keys()):
                count = category_counts[cat]
                target = CATEGORY_TARGETS[cat]
                bar = "█" * (count * 20 // target) + "░" * (20 - count * 20 // target)
                print(f"    {cat:<35} [{bar}] {count:>3}/{target}", flush=True)
            print()
        
        # Wait 3 seconds between batches
        await asyncio.sleep(3)
    
    print("\n" + "=" * 70, flush=True)
    print("GENERATION COMPLETE", flush=True)
    print("=" * 70, flush=True)
    elapsed_total = (datetime.now() - start_time).seconds / 60
    print(f"Total: {len(all_data)} examples in {elapsed_total:.1f} minutes", flush=True)
    print(f"Output: {output_file}", flush=True)
    
    return output_file


if __name__ == "__main__":
    asyncio.run(run_generation())
