"""
Test the SFT model on various inputs.

Tests:
1. Examples from training dataset
2. Examples from test dataset  
3. Novel inputs the model has never seen
"""

import asyncio
import json
import os
from dotenv import load_dotenv

load_dotenv()

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Configuration
SFT_CHECKPOINT = "tinker://4f4bae1f-5a95-5f53-a55a-a14f2872825c:train:0/sampler_weights/sft_step_0090"
BASE_MODEL = "meta-llama/Llama-3.1-8B"

VALID_CATEGORIES = {
    "company.brand_core", "company.strategic_signatures", "company.knowledge_artifacts",
    "company.business_priorities", "company.tools_config", "company.performance_context",
    "user.communication_style", "user.strategic_approach", "user.role_context",
    "user.workflow_patterns", "user.session_history", "user.interaction_preferences",
    "none"
}

# Novel test cases the model has never seen
NOVEL_INPUTS = [
    {
        "conversation": """USER: I prefer getting updates in bullet points, not long paragraphs. Keep it concise.
ASSISTANT: Understood! I'll format all future updates as bullet points for easy scanning.
USER: Perfect. Also, I check messages mainly in the morning, so schedule any non-urgent stuff for 8am.""",
        "expected": ["user.communication_style", "user.workflow_patterns"],
        "description": "Communication preferences + workflow timing"
    },
    {
        "conversation": """USER: Our brand voice is playful but professional. Think friendly neighbor who happens to be an expert.
ASSISTANT: Great analogy! So approachable expertise - warm but credible.
USER: Exactly. We never use corporate jargon. Everything should feel like a conversation.""",
        "expected": ["company.brand_core"],
        "description": "Brand voice definition"
    },
    {
        "conversation": """USER: What's the weather like today?
ASSISTANT: I don't have access to real-time weather data. You might want to check a weather app.
USER: Ok thanks.""",
        "expected": ["none"],
        "description": "Transactional/irrelevant conversation"
    },
    {
        "conversation": """USER: We use HubSpot for CRM, Mailchimp for emails, and Slack for team comms. Everything needs to sync.
ASSISTANT: Got it. I'll note those integrations. Any specific workflows between them?
USER: Yeah, new leads from HubSpot should trigger a Slack notification and add to a Mailchimp welcome sequence.""",
        "expected": ["company.tools_config"],
        "description": "Tool stack and integrations"
    },
    {
        "conversation": """USER: Last quarter's email campaign had 45% open rate but only 2% conversion. We need to fix the CTA.
ASSISTANT: That's a significant drop-off. The open rate suggests good subject lines but weak call-to-action.
USER: Right. Let's A/B test three different CTAs next month.""",
        "expected": ["company.performance_context"],
        "description": "Campaign metrics and learnings"
    },
    {
        "conversation": """USER: I'm the VP of Marketing, reporting directly to the CEO. I have final say on brand decisions.
ASSISTANT: Understood. So you have executive authority on brand strategy.
USER: Yes, but I like to get input from the creative team before making calls.""",
        "expected": ["user.role_context"],
        "description": "Role and authority"
    },
    {
        "conversation": """USER: We always prioritize data over intuition. Every decision needs numbers to back it up.
ASSISTANT: A data-driven approach. Do you have minimum thresholds for statistical significance?
USER: Yes, 95% confidence interval minimum. No exceptions.""",
        "expected": ["company.strategic_signatures"],
        "description": "Decision-making framework"
    },
    {
        "conversation": """USER: Q4 is all about the holiday push. Black Friday and Cyber Monday are our biggest days.
ASSISTANT: Seasonal focus noted. What's the budget allocation for Q4?
USER: 60% of annual marketing budget goes to Q4. It's make or break.""",
        "expected": ["company.business_priorities"],
        "description": "Quarterly priorities"
    },
    {
        "conversation": """USER: I want you to challenge my ideas more. Don't just agree with everything.
ASSISTANT: I appreciate that feedback. You want more critical analysis and pushback.
USER: Yes, play devil's advocate when needed. I learn better that way.""",
        "expected": ["user.interaction_preferences"],
        "description": "Coaching/feedback style"
    },
    {
        "conversation": """USER: Yesterday we discussed the influencer campaign. Did you save those notes?
ASSISTANT: Yes, I have the notes from our previous session about the influencer strategy.
USER: Good. Let's continue from where we left off - the micro-influencer targeting.""",
        "expected": ["user.session_history"],
        "description": "Reference to previous context"
    },
]


def parse_prediction(text: str) -> set:
    """Parse model output into category set."""
    if not text or not text.strip():
        return set()
    cats = [c.strip().lower() for c in text.split(",")]
    return {c for c in cats if c in VALID_CATEGORIES}


def compute_metrics(predicted: set, gold: set) -> dict:
    """Compute evaluation metrics."""
    if not predicted and not gold:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0, "exact_match": True, "any_match": True}
    if not predicted or not gold:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "exact_match": False, "any_match": False}
    
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0
    recall = tp / len(gold) if gold else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "exact_match": predicted == gold,
        "any_match": bool(predicted & gold)
    }


async def test_model():
    print("=" * 70)
    print("SFT MODEL EVALUATION")
    print("=" * 70)
    print(f"Checkpoint: {SFT_CHECKPOINT}")
    print()
    
    # Initialize
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=SFT_CHECKPOINT)
    tokenizer = get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(name="llama3", tokenizer=tokenizer)
    
    stop_sequences = renderer.get_stop_sequences()
    params = types.SamplingParams(max_tokens=100, temperature=0.1, stop=stop_sequences)
    
    # System prompt
    system_prompt = """You route marketing conversations into structured memory categories.

Available categories:
- company.brand_core: Voice, values, positioning, identity anchors
- company.strategic_signatures: Decision frameworks, strategic heuristics
- company.knowledge_artifacts: Docs, style guides, playbooks
- company.business_priorities: Quarterly/seasonal goals, active campaigns
- company.tools_config: Integrations, API keys, workflow settings
- company.performance_context: Campaign metrics, retrospectives, learnings
- user.communication_style: Tone, verbosity, format expectations
- user.strategic_approach: Personal priorities, success definitions
- user.role_context: Title, scope, decision authority
- user.workflow_patterns: Review cadence, collaboration norms
- user.session_history: Immediate context, recent asks
- user.interaction_preferences: Coaching style, feedback expectations
- none: Irrelevant, vague, or transactional content

Respond with comma-separated categories. Use 'none' only if no other category applies."""

    # =========================================================================
    # Test 1: Novel inputs
    # =========================================================================
    print("-" * 70)
    print("TEST 1: NOVEL INPUTS (Never seen during training)")
    print("-" * 70)
    
    novel_results = []
    
    for i, test_case in enumerate(NOVEL_INPUTS):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this conversation and determine which memory categories apply:\n\n{test_case['conversation']}"}
        ]
        
        prompt = renderer.build_generation_prompt(messages)
        result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
        response, _ = renderer.parse_response(result.sequences[0].tokens)
        predicted_text = response["content"]
        
        predicted = parse_prediction(predicted_text)
        gold = set(test_case["expected"])
        metrics = compute_metrics(predicted, gold)
        
        novel_results.append(metrics)
        
        status = "✓" if metrics["any_match"] else "✗"
        exact = "EXACT" if metrics["exact_match"] else ""
        
        print(f"\n[{i+1}] {test_case['description']}")
        print(f"    Expected:  {', '.join(sorted(gold))}")
        print(f"    Predicted: {predicted_text}")
        print(f"    {status} F1: {metrics['f1']:.2f} {exact}")
    
    # Summary for novel inputs
    avg_f1 = sum(r["f1"] for r in novel_results) / len(novel_results)
    any_match_rate = sum(1 for r in novel_results if r["any_match"]) / len(novel_results)
    exact_match_rate = sum(1 for r in novel_results if r["exact_match"]) / len(novel_results)
    
    print(f"\n{'='*50}")
    print(f"NOVEL INPUTS SUMMARY ({len(novel_results)} examples)")
    print(f"  Any Match:   {any_match_rate:.1%}")
    print(f"  Exact Match: {exact_match_rate:.1%}")
    print(f"  Avg F1:      {avg_f1:.2f}")
    
    # =========================================================================
    # Test 2: Training dataset examples
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 2: TRAINING DATASET EXAMPLES")
    print("-" * 70)
    
    with open("training/processed_data/train_data.json", "r") as f:
        train_data = json.load(f)
    
    # Sample 20 random examples
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(train_data)), min(20, len(train_data)))
    
    train_results = []
    
    for idx in sample_indices:
        item = train_data[idx]
        messages = item["messages"][:-1]  # Exclude assistant response
        gold = set(item["categories"])
        
        prompt = renderer.build_generation_prompt(messages)
        result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
        response, _ = renderer.parse_response(result.sequences[0].tokens)
        predicted_text = response["content"]
        
        predicted = parse_prediction(predicted_text)
        metrics = compute_metrics(predicted, gold)
        train_results.append(metrics)
    
    avg_f1 = sum(r["f1"] for r in train_results) / len(train_results)
    any_match_rate = sum(1 for r in train_results if r["any_match"]) / len(train_results)
    exact_match_rate = sum(1 for r in train_results if r["exact_match"]) / len(train_results)
    
    print(f"TRAINING SET SAMPLE ({len(train_results)} examples)")
    print(f"  Any Match:   {any_match_rate:.1%}")
    print(f"  Exact Match: {exact_match_rate:.1%}")
    print(f"  Avg F1:      {avg_f1:.2f}")
    
    # =========================================================================
    # Test 3: Test dataset examples
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 3: TEST DATASET EXAMPLES (Held out)")
    print("-" * 70)
    
    with open("training/processed_data/test_data.json", "r") as f:
        test_data = json.load(f)
    
    test_results = []
    
    for item in test_data[:50]:  # First 50 test examples
        messages = item["messages"][:-1]
        gold = set(item["categories"])
        
        prompt = renderer.build_generation_prompt(messages)
        result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
        response, _ = renderer.parse_response(result.sequences[0].tokens)
        predicted_text = response["content"]
        
        predicted = parse_prediction(predicted_text)
        metrics = compute_metrics(predicted, gold)
        test_results.append(metrics)
    
    avg_f1 = sum(r["f1"] for r in test_results) / len(test_results)
    any_match_rate = sum(1 for r in test_results if r["any_match"]) / len(test_results)
    exact_match_rate = sum(1 for r in test_results if r["exact_match"]) / len(test_results)
    
    print(f"TEST SET ({len(test_results)} examples)")
    print(f"  Any Match:   {any_match_rate:.1%}")
    print(f"  Exact Match: {exact_match_rate:.1%}")
    print(f"  Avg F1:      {avg_f1:.2f}")
    
    # =========================================================================
    # Overall Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Dataset':<20} {'Any Match':<12} {'Exact Match':<12} {'Avg F1':<10}")
    print("-" * 54)
    
    # Novel
    novel_any = sum(1 for r in novel_results if r["any_match"]) / len(novel_results)
    novel_exact = sum(1 for r in novel_results if r["exact_match"]) / len(novel_results)
    novel_f1 = sum(r["f1"] for r in novel_results) / len(novel_results)
    print(f"{'Novel Inputs':<20} {novel_any:<12.1%} {novel_exact:<12.1%} {novel_f1:<10.2f}")
    
    # Train
    train_any = sum(1 for r in train_results if r["any_match"]) / len(train_results)
    train_exact = sum(1 for r in train_results if r["exact_match"]) / len(train_results)
    train_f1 = sum(r["f1"] for r in train_results) / len(train_results)
    print(f"{'Train Sample':<20} {train_any:<12.1%} {train_exact:<12.1%} {train_f1:<10.2f}")
    
    # Test
    test_any = sum(1 for r in test_results if r["any_match"]) / len(test_results)
    test_exact = sum(1 for r in test_results if r["exact_match"]) / len(test_results)
    test_f1 = sum(r["f1"] for r in test_results) / len(test_results)
    print(f"{'Test Set':<20} {test_any:<12.1%} {test_exact:<12.1%} {test_f1:<10.2f}")
    
    print("\n" + "=" * 70)
    print("SFT EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_model())

