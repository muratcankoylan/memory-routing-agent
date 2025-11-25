"""
Quick evaluation of SFT model on test data and novel inputs.
"""

import asyncio
import json
import os
from dotenv import load_dotenv

# Load env before imports
load_dotenv()

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

BASE_MODEL = "meta-llama/Llama-3.1-8B"
SFT_CHECKPOINT = "tinker://44184486-b599-5739-8819-c38f1d5d8ad0:train:0/sampler_weights/sft_step_0049"

SYSTEM_PROMPT = """You route marketing conversations into structured memory categories.

Available categories:
- company.brand_core: Voice, values, positioning, identity anchors (Long >1y)
- company.strategic_signatures: Decision frameworks, strategic heuristics (Long >1y)
- company.knowledge_artifacts: Docs, style guides, playbooks (Long >1y)
- company.business_priorities: Quarterly/seasonal goals, active campaigns (Short <3m)
- company.tools_config: Integrations, API keys, workflow settings (Medium ~6m)
- company.performance_context: Campaign metrics, retrospectives, learnings (Rolling ~6m)
- user.communication_style: Tone, verbosity, format expectations (Long >1y)
- user.strategic_approach: Personal priorities, success definitions (Long >1y)
- user.role_context: Title, scope, decision authority (Medium ~1y)
- user.workflow_patterns: Review cadence, collaboration norms (Medium ~1y)
- user.session_history: Immediate context, recent asks (Short <2w)
- user.interaction_preferences: Coaching style, feedback expectations (Evolving)
- none: Irrelevant, vague, or transactional content

Respond with comma-separated categories. Use 'none' only if no other category applies."""

VALID_CATEGORIES = {
    "company.brand_core", "company.strategic_signatures", "company.knowledge_artifacts",
    "company.business_priorities", "company.tools_config", "company.performance_context",
    "user.communication_style", "user.strategic_approach", "user.role_context",
    "user.workflow_patterns", "user.session_history", "user.interaction_preferences",
    "none"
}

# Novel test cases the model has NEVER seen
NOVEL_TESTS = [
    {
        "name": "CEO Strategy Discussion",
        "conversation": """USER: As CEO, I've decided we're pivoting from B2B to B2C next quarter. This is confidential until the board meeting.
ASSISTANT: Understood. This strategic pivot will require significant changes across marketing, sales, and product. Should I help outline the key considerations?
USER: Yes, and remember - I prefer bullet points over long paragraphs. Keep updates brief.""",
        "expected_categories": ["user.strategic_approach", "company.strategic_signatures", "user.communication_style"]
    },
    {
        "name": "Tool Integration Request",
        "conversation": """USER: Can you help me set up the Slack integration with our CRM? The API key is sk-xxx-123.
ASSISTANT: I'll help you configure the Slack-CRM integration. First, let me verify the API key format...
USER: Also, make sure notifications go to #marketing-alerts channel.""",
        "expected_categories": ["company.tools_config"]
    },
    {
        "name": "Casual Chat (None)",
        "conversation": """USER: Hey, what's the weather like today?
ASSISTANT: I don't have access to real-time weather data. You might want to check a weather app or website.
USER: No worries, just making small talk while I wait for my coffee.""",
        "expected_categories": ["none"]
    },
    {
        "name": "Brand Voice Guidelines",
        "conversation": """USER: Our brand voice should always be warm, approachable, and slightly playful. We never use corporate jargon.
ASSISTANT: Got it - warm, approachable, playful, no jargon. Should I apply this to all future content?
USER: Yes, and we always sign off emails with 'Cheers' not 'Best regards'.""",
        "expected_categories": ["company.brand_core", "company.knowledge_artifacts"]
    },
    {
        "name": "Campaign Performance Review",
        "conversation": """USER: The Q3 email campaign had a 24% open rate and 3.2% CTR. That's up from 18% and 2.1% last quarter.
ASSISTANT: Great improvement! The new subject line strategy seems to be working. What do you attribute the gains to?
USER: Definitely the personalization we added. Let's document this for future campaigns.""",
        "expected_categories": ["company.performance_context", "company.knowledge_artifacts"]
    },
    {
        "name": "Session Context",
        "conversation": """USER: Let's continue working on the proposal we started yesterday.
ASSISTANT: Of course! We were drafting the executive summary for the Johnson account. You wanted to emphasize ROI projections.
USER: Right, and I mentioned I need this done by Friday 3pm.""",
        "expected_categories": ["user.session_history"]
    },
    {
        "name": "Role and Authority",
        "conversation": """USER: I'm the VP of Marketing and I report directly to the CMO. I have budget authority up to $50k.
ASSISTANT: Noted. With that authority level, you can approve most campaign expenditures directly.
USER: Exactly. Anything above $50k needs Sarah's sign-off.""",
        "expected_categories": ["user.role_context"]
    },
    {
        "name": "Workflow Preferences",
        "conversation": """USER: I review all campaign drafts on Monday mornings. Don't send me anything on Fridays - I'm in back-to-back meetings.
ASSISTANT: Understood. Monday morning reviews, no Friday submissions.
USER: And I always want to see the A/B test variants side by side, not in separate documents.""",
        "expected_categories": ["user.workflow_patterns", "user.interaction_preferences"]
    },
    {
        "name": "Q4 Goals",
        "conversation": """USER: For Q4, we're focusing on lead gen for the enterprise segment. Target is 500 MQLs.
ASSISTANT: Got it - enterprise focus, 500 MQL target for Q4. Should I prioritize content for that segment?
USER: Yes, and the holiday campaign should tie into this goal.""",
        "expected_categories": ["company.business_priorities"]
    },
    {
        "name": "Technical Documentation",
        "conversation": """USER: Here's our updated style guide. Headlines should be sentence case, max 60 characters. Body copy in AP style.
ASSISTANT: Noted - sentence case headlines under 60 chars, AP style for body.
USER: I'm also attaching our brand color codes: primary #2563EB, secondary #10B981.""",
        "expected_categories": ["company.knowledge_artifacts", "company.brand_core"]
    }
]


async def test_model():
    print("=" * 70)
    print("SFT MODEL EVALUATION")
    print("=" * 70)
    print(f"Checkpoint: {SFT_CHECKPOINT}")
    print()
    
    # Initialize
    service_client = tinker.ServiceClient()
    tokenizer = get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(name="llama3", tokenizer=tokenizer)
    sampling_client = service_client.create_sampling_client(model_path=SFT_CHECKPOINT)
    
    stop_sequences = renderer.get_stop_sequences()
    params = types.SamplingParams(max_tokens=100, temperature=0.1, stop=stop_sequences)
    
    # Part 1: Test on held-out test data
    print("-" * 70)
    print("PART 1: TEST SET EVALUATION (50 examples)")
    print("-" * 70)
    
    with open("synthetic_data/training_dataset_1000.jsonl") as f:
        all_data = [json.loads(l) for l in f]
    
    # Use last 200 as test, sample 50
    test_data = all_data[-200:][:50]
    
    correct_any = 0
    correct_exact = 0
    
    for i, item in enumerate(test_data):
        conv = item.get("conversation", [])
        gold = item.get("labels", {}).get("categories", [])
        
        # Build conversation text
        conv_text = ""
        for turn in conv:
            if isinstance(turn, dict):
                conv_text += f"{turn['role'].upper()}: {turn['content']}\n"
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Conversation:\n{conv_text}"}
        ]
        
        prompt = renderer.build_generation_prompt(messages)
        result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
        response, _ = renderer.parse_response(result.sequences[0].tokens)
        pred = response["content"]
        
        pred_set = set([c.strip().lower() for c in pred.split(",") if c.strip().lower() in VALID_CATEGORIES])
        gold_set = set([c.lower() for c in gold])
        
        if pred_set & gold_set:
            correct_any += 1
        if pred_set == gold_set:
            correct_exact += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/50...")
    
    print()
    print(f"Any Match Accuracy: {correct_any}/{len(test_data)} = {correct_any/len(test_data):.1%}")
    print(f"Exact Match Accuracy: {correct_exact}/{len(test_data)} = {correct_exact/len(test_data):.1%}")
    
    # Part 2: Novel inputs
    print()
    print("-" * 70)
    print("PART 2: NOVEL INPUTS (Never seen during training)")
    print("-" * 70)
    
    novel_correct = 0
    novel_exact = 0
    
    for test in NOVEL_TESTS:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Conversation:\n{test['conversation']}"}
        ]
        
        prompt = renderer.build_generation_prompt(messages)
        result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
        response, _ = renderer.parse_response(result.sequences[0].tokens)
        pred = response["content"]
        
        pred_set = set([c.strip().lower() for c in pred.split(",") if c.strip().lower() in VALID_CATEGORIES])
        expected_set = set([c.lower() for c in test["expected_categories"]])
        
        any_match = bool(pred_set & expected_set)
        exact_match = pred_set == expected_set
        
        if any_match:
            novel_correct += 1
        if exact_match:
            novel_exact += 1
        
        match_icon = "✓" if any_match else "✗"
        exact_icon = " [EXACT]" if exact_match else ""
        
        print(f"\n{match_icon} {test['name']}{exact_icon}")
        print(f"   Expected:  {', '.join(sorted(test['expected_categories']))}")
        print(f"   Predicted: {pred.strip()}")
    
    print()
    print("-" * 70)
    print("NOVEL INPUT RESULTS")
    print("-" * 70)
    print(f"Any Match:   {novel_correct}/{len(NOVEL_TESTS)} = {novel_correct/len(NOVEL_TESTS):.1%}")
    print(f"Exact Match: {novel_exact}/{len(NOVEL_TESTS)} = {novel_exact/len(NOVEL_TESTS):.1%}")
    print()


if __name__ == "__main__":
    asyncio.run(test_model())

