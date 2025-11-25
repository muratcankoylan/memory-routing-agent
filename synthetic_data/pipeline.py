import os
import json
import random
import asyncio
import time
import cohere
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

DOMAIN_CONTEXTS = [
    "B2B SaaS workflow automation for enterprise teams",
    "Consumer fintech budgeting assistant rolling out in LATAM",
    "Healthcare patient engagement platform coordinating compliance content",
    "Retail omnichannel loyalty program for a fashion brand",
    "EdTech company designing AI tutoring playbooks",
    "Hospitality chain redefining guest personalization across regions",
    "Developer tools startup improving product-led growth motions",
    "Sports media network negotiating sponsorship activations",
    "Gaming studio planning live-ops launches",
    "Non-profit fundraising platform balancing donor messaging",
    "Enterprise cybersecurity firm running incident response playbooks",
    "Supply-chain analytics platform optimizing vendor collaboration",
    "CPG beverage brand planning seasonal launches with agencies",
    "Real-estate marketplace coordinating broker enablement",
    "Mobility/ride-hailing service planning driver communications",
    "Streaming media company managing international content drops",
    "Insurance carrier modernizing agent training workflows",
    "Energy provider coordinating demand-response campaigns",
    "Professional services firm standardizing proposal playbooks",
    "AI infrastructure startup refining go-to-market with partners",
    "Luxury beauty brand orchestrating influencer activations",
    "Food delivery platform improving courier retention messaging",
    "Corporate learning company updating compliance curricula",
    "Outdoor gear company rolling out omnichannel retail pilots"
]

class SyntheticDataPipeline:
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 5):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")
        self.client = cohere.ClientV2(api_key=self.api_key)
        # Switched to command-r-plus-08-2024 due to rate limits on reasoning model
        self.model = "command-r-plus-08-2024"
        self.max_retries = max_retries

    def _sample_domain_context(self) -> str:
        return random.choice(DOMAIN_CONTEXTS)

    @staticmethod
    def _extract_text(response) -> Optional[str]:
        """Extract the first text block from a Cohere response."""
        if not response or not getattr(response, "message", None):
            return None
        blocks = getattr(response.message, "content", []) or []
        for block in blocks:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                return text
        return None

    def generate_scenario_spec(self, category: str, distractor: Optional[str] = None, 
                             persistence: str = "long", tone: str = "neutral", 
                             turns: int = 6, special_reqs: str = "") -> Dict[str, Any]:
        """Stage 1: Generate a scenario specification."""
        domain_context = self._sample_domain_context()
        midstream_note = "Conversation should start mid-thread (no greetings) and refer back to earlier collaboration."
        diversity_note = "Keep subject matter aligned with the given domain context; avoid repeating eco/climate themes unless category demands it."
        combined_reqs = " | ".join(filter(None, [special_reqs, midstream_note, diversity_note]))
        
        if category == "none":
            prompt = f"""Generate a JSON scenario specification for a conversation that has NO long-term memory value (Category: none).
The conversation should be strictly transactional, vague, or temporary.
Examples: checking status, scheduling a meeting, asking a clarification, greeting, small talk, or discussing weather/lunch.

CONTEXT: General professional setting. Do NOT include any strategic projects, specific brand details, or user preferences that would trigger memory storage.

Requirements:
- Primary Category: none
- Distractor Category: {distractor if distractor else "None"}
- Persistence Level: short
- Turn Count: {turns}
- Special Requirements: {combined_reqs}

Return a JSON object with:
{{
  "scenario_description": "Brief narrative setup (2-3 sentences) - MUST BE NON-MEMORABLE",
  "user_profile": "User role",
  "key_signals_to_include": ["List of 2-4 signals that are specifically IRRELEVANT or TEMPORARY"],
  "distractor_signals": ["Optional list of signals"],
  "suggested_turn_breakdown": "Flow of conversation"
}}
"""
        else:
            prompt = f"""You are designing training scenarios for an AI memory system in marketing context. Generate a scenario specification tailored to this business setting: {domain_context}.

Requirements:
- Primary Category: {category}
- Distractor Category: {distractor if distractor else "None"}
- Persistence Level: {persistence}
- Emotional Tone: {tone}
- Turn Count: {turns}
- Special Requirements: {combined_reqs}

Return a JSON object with:
{{
  "scenario_description": "Brief narrative setup (2-3 sentences)",
  "user_profile": "User role and context",
  "key_signals_to_include": ["List of 2-4 specific memory-worthy signals"],
  "distractor_signals": ["Optional list of noise/irrelevant info"],
  "suggested_turn_breakdown": "How the conversation should flow"
}}
"""

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    model=self.model,
                    response_format={"type": "json_object"}
                )
                content = self._extract_text(response)
                if not content:
                    raise ValueError("No text content found in scenario response")

                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                return json.loads(content.strip())
            except Exception as e:
                print(f"Scenario generation failed (attempt {attempt+1}/{self.max_retries+1}): {e}")
                if attempt < self.max_retries:
                    sleep_time = 10 * (2 ** attempt)
                    print(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
        return {}

    def generate_conversation(self, scenario_spec: Dict[str, Any], turn_count: int = 6, category: Optional[str] = None) -> Dict[str, Any]:
        """Stage 2: Generate conversation based on scenario spec."""
        
        domain_context = self._sample_domain_context()
        
        # Detect if this is a NONE category scenario
        is_none = category == "none" or (category is None and "none" in str(scenario_spec).lower())
        
        if is_none:
             prompt = f"""You are generating a realistic conversation between a user and an AI assistant.
The conversation should be transactional, casual, or vague. IT SHOULD NOT contain any significant long-term memory value for a marketing context.

CONTEXT: General professional setting.
SCENARIO SPECIFICATION:
{json.dumps(scenario_spec, indent=2)}

GENERATION RULES:
1. Make it natural and fluid.
2. DO NOT include detailed strategic plans, brand values, or user preferences.
3. Focus on immediate tasks (scheduling, clarifications, small talk).
4. Length: {turn_count} turns.
5. Avoid opening pleasantries like "Hi" - start mid-thread if appropriate, or just dive in.

OUTPUT FORMAT:
Return a JSON object with:
{{
  "scenario_id": "none_transactional_{{random_3_digit_number}}",
  "conversation": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ],
  "labels": {{
    "categories": ["none"],
    "persistence_horizon": "short",
    "memory_scope": "none",
    "rationale": "Explanation why this is not memory-worthy"
  }},
  "metadata": {{
    "scenario_type": "negative_example",
    "primary_category": "none",
    "distractor_present": false,
    "turn_count": {turn_count},
    "signals_present": []
  }}
}}

CRITICAL: Respond with ONLY the JSON object.
"""
        else:
            prompt = f"""You are generating realistic marketing conversations between a user and an AI marketing assistant. Generate natural dialogue that contains specific information worth storing in long-term memory. The conversation should start mid-thread (no greetings) and reference the ongoing initiative described below.

CONTEXT:
You will create a conversation that exemplifies certain memory categories while maintaining realism and natural flow. Assume this is part of {domain_context}.

SCENARIO SPECIFICATION:
{json.dumps(scenario_spec, indent=2)}

MEMORY TAXONOMY (for reference):
COMPANY MEMORY:
- company.brand_core: Voice, values, positioning, identity anchors (Persistence: Long >1y)
- company.strategic_signatures: Decision frameworks, strategic heuristics (Persistence: Long >1y)
- company.knowledge_artifacts: Docs, style guides, playbooks (Persistence: Long >1y)
- company.business_priorities: Quarterly/seasonal goals, active campaigns (Persistence: Short <3m)
- company.tools_config: Integrations, API keys, workflow settings (Persistence: Medium ~6m)
- company.performance_context: Campaign metrics, retrospectives, learnings (Persistence: Rolling ~6m)

USER MEMORY:
- user.communication_style: Tone, verbosity, format expectations (Persistence: Long >1y)
- user.strategic_approach: Personal priorities, success definitions (Persistence: Long >1y)
- user.role_context: Title, scope, decision authority (Persistence: Medium ~1y)
- user.workflow_patterns: Review cadence, collaboration norms (Persistence: Medium ~1y)
- user.session_history: Immediate context, recent asks (Persistence: Short <2w)
- user.interaction_preferences: Coaching style, feedback expectations (Persistence: Evolving)

SPECIAL:
- none: Irrelevant, vague, or transactional content

GENERATION RULES:
1. Make conversations feel natural - include some filler, transitions, acknowledgments
2. Embed memory-worthy information organically (don't make it too obvious)
3. Include 1-2 utterances that should map to "none" for realism
4. If multi-label scenario, ensure signals for both categories are present
5. Length: {turn_count} turns (alternating user/assistant)
6. Include specific, concrete details (not generic statements)
7. For company.* categories: use "we", "our company", "our brand"
8. For user.* categories: use "I prefer", "my approach", "I typically"
9. Avoid opening pleasantries like "Hi" or "Hello"—jump straight into the ongoing topic.
10. **CRITICAL CONSTRAINT**: Limit output to 1-3 categories maximum.
11. **EXCLUSIVE NONE**: If "none" is in the categories list, it MUST be the ONLY category. NEVER mix "none" with other categories. If valid signals exist, do NOT include "none".

    OUTPUT FORMAT:
Return a JSON object with:
{{
  "scenario_id": "{{primary_category}}_{{scenario_type}}_{{random_3_digit_number}}",
  "conversation": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    ...
  ],
  "labels": {{
    "categories": ["array of applicable categories"],
    "persistence_horizon": "long|medium|short",
    "memory_scope": "company|user|mixed|none",
    "rationale": "1-2 sentence explanation of category choices"
  }},
  "metadata": {{
    "scenario_type": "descriptive_label",
    "primary_category": "main_category",
    "distractor_present": true|false,
    "turn_count": integer,
    "signals_present": ["list of specific signals included"]
  }}
}}

CRITICAL: Respond with ONLY the JSON object. No markdown formatting, no explanation, no preamble.

Generate the conversation now."""

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    model=self.model,
                    response_format={"type": "json_object"}
                )
                content = self._extract_text(response)
                if not content:
                    raise ValueError("No text content found in conversation response")

                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                return json.loads(content.strip())
            except Exception as e:
                print(f"Conversation generation failed (attempt {attempt+1}/{self.max_retries+1}): {e}")
                if attempt < self.max_retries:
                    sleep_time = 10 * (2 ** attempt)
                    print(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
        return {}

    def run_batch(self, count: int = 1, category: str = "company.brand_core") -> List[Dict[str, Any]]:
        """Run a batch generation."""
        results = []
        print(f"Starting batch generation for {count} examples of {category}...")
        
        for i in range(count):
            print(f"Generating example {i+1}/{count}...")
            scenario = self.generate_scenario_spec(category=category)
            if not scenario:
                print("Skipping due to scenario generation failure")
                continue
                
            conversation = self.generate_conversation(scenario)
            if conversation:
                results.append(conversation)
                print(f"Successfully generated conversation: {conversation.get('scenario_id', 'unknown')}")
            else:
                print("Failed to generate conversation")
                
        return results

if __name__ == "__main__":
    # Simple test run
    pipeline = SyntheticDataPipeline()
    results = pipeline.run_batch(count=1)
    print(json.dumps(results, indent=2))

