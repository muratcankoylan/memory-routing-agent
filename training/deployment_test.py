"""
Deployment Readiness Test for Memory Routing Model

This script tests the model on completely novel scenarios that are 
intentionally different from the training distribution:
- Different industries (healthcare, legal, finance, education)
- Different conversation styles (formal, casual, technical)
- Edge cases and ambiguous scenarios
- Multi-label scenarios
- Negative examples (should be "none")
"""

import asyncio
from dotenv import load_dotenv
load_dotenv()

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

VALID_CATEGORIES = {
    "company.brand_core", "company.strategic_signatures", "company.knowledge_artifacts",
    "company.business_priorities", "company.tools_config", "company.performance_context",
    "user.communication_style", "user.strategic_approach", "user.role_context",
    "user.workflow_patterns", "user.session_history", "user.interaction_preferences",
    "none"
}

# Completely novel test scenarios - different from training data
DEPLOYMENT_TESTS = [
    # ========== HEALTHCARE INDUSTRY ==========
    {
        "id": "healthcare_1",
        "industry": "Healthcare",
        "conversation": """USER: As the Chief Marketing Officer at MedTech Solutions, I need all patient testimonials to go through our legal team first. We can't risk HIPAA violations.
ASSISTANT: Understood. I'll flag all testimonial content for legal review before any publication.
USER: Good. Also, we never use fear-based messaging. Our brand is about empowerment and hope.""",
        "expected": ["company.brand_core", "company.strategic_signatures", "user.role_context"],
        "description": "Healthcare CMO defining brand voice and compliance rules"
    },
    {
        "id": "healthcare_2",
        "industry": "Healthcare",
        "conversation": """USER: Can you check if the printer on floor 3 is working?
ASSISTANT: I don't have access to printer status. You might want to check with IT.
USER: Never mind, I'll walk over.""",
        "expected": ["none"],
        "description": "Transactional office request - should be none"
    },
    
    # ========== LEGAL/LAW FIRM ==========
    {
        "id": "legal_1",
        "industry": "Legal",
        "conversation": """USER: Our firm's positioning is 'aggressive but ethical'. We fight hard for clients but never cross ethical lines.
ASSISTANT: That's a clear brand differentiator in the legal space.
USER: Exactly. All marketing copy must reflect this balance. No ambulance-chasing language.""",
        "expected": ["company.brand_core"],
        "description": "Law firm brand positioning"
    },
    {
        "id": "legal_2",
        "industry": "Legal",
        "conversation": """USER: I reviewed last quarter's client acquisition costs. We're spending $1,200 per lead but conversion is only 8%.
ASSISTANT: That's a high CAC with low conversion. What's the industry benchmark?
USER: Competitors are at $800 with 12% conversion. We need to optimize our funnel.""",
        "expected": ["company.performance_context"],
        "description": "Legal firm analyzing marketing performance"
    },
    
    # ========== EDUCATION/UNIVERSITY ==========
    {
        "id": "education_1",
        "industry": "Education",
        "conversation": """USER: I'm the Dean of Admissions. I prefer receiving weekly enrollment reports every Monday at 9am, formatted as executive summaries.
ASSISTANT: I'll schedule weekly summaries for Monday mornings.
USER: And please keep them under 2 pages. I don't have time for lengthy reports.""",
        "expected": ["user.role_context", "user.workflow_patterns", "user.communication_style"],
        "description": "University dean stating role and preferences"
    },
    {
        "id": "education_2",
        "industry": "Education",
        "conversation": """USER: We use Slate for CRM, Mailchimp for email campaigns, and Canvas for course delivery.
ASSISTANT: Got it. Should these systems sync with each other?
USER: Yes, new student inquiries in Slate should trigger a welcome email sequence in Mailchimp.""",
        "expected": ["company.tools_config"],
        "description": "University tech stack integration"
    },
    
    # ========== FINANCE/FINTECH ==========
    {
        "id": "finance_1",
        "industry": "Finance",
        "conversation": """USER: Our investment philosophy is conservative growth. We never chase high-risk opportunities.
ASSISTANT: So capital preservation is prioritized over aggressive returns?
USER: Correct. This should be reflected in all our client communications.""",
        "expected": ["company.strategic_signatures", "company.brand_core"],
        "description": "Financial firm investment philosophy"
    },
    {
        "id": "finance_2",
        "industry": "Finance",
        "conversation": """USER: What time is the team meeting tomorrow?
ASSISTANT: I don't have access to your calendar. You might want to check your Outlook.
USER: Right, thanks.""",
        "expected": ["none"],
        "description": "Simple calendar question - should be none"
    },
    
    # ========== EDGE CASES ==========
    {
        "id": "edge_1",
        "industry": "Mixed",
        "conversation": """USER: I've been thinking about how we approach decisions here. We always validate with data before committing, but we also trust gut instincts when data is inconclusive.
ASSISTANT: A balanced approach between analytical and intuitive decision-making.
USER: Yes, and personally, I tend to be more data-driven. I need numbers before I'm comfortable.""",
        "expected": ["company.strategic_signatures", "user.strategic_approach"],
        "description": "Mixed company and personal decision frameworks"
    },
    {
        "id": "edge_2",
        "industry": "Mixed",
        "conversation": """USER: Remember yesterday when we discussed the Q3 campaign strategy?
ASSISTANT: Yes, we talked about the social media push for the product launch.
USER: Let's continue from there. What were the key action items?""",
        "expected": ["user.session_history"],
        "description": "Reference to previous session"
    },
    {
        "id": "edge_3",
        "industry": "Retail",
        "conversation": """USER: I want you to push back on my ideas more. Don't just agree with everything I say.
ASSISTANT: I appreciate that feedback. You want more critical analysis.
USER: Yes, challenge my assumptions. That's how I learn best.""",
        "expected": ["user.interaction_preferences"],
        "description": "User defining coaching style preference"
    },
    
    # ========== AMBIGUOUS CASES ==========
    {
        "id": "ambiguous_1",
        "industry": "Tech",
        "conversation": """USER: The product launch is next month and we need all hands on deck.
ASSISTANT: What's the priority for the marketing team?
USER: Focus on the launch. Everything else can wait.""",
        "expected": ["company.business_priorities"],
        "description": "Short-term business priority"
    },
    {
        "id": "ambiguous_2",
        "industry": "Agency",
        "conversation": """USER: Hi, how are you today?
ASSISTANT: I'm doing well, thank you! How can I help you?
USER: Just checking in. Nothing specific right now.""",
        "expected": ["none"],
        "description": "Pure small talk - should be none"
    },
    
    # ========== FORMAL VS CASUAL ==========
    {
        "id": "formal_1",
        "industry": "Corporate",
        "conversation": """USER: Per our previous correspondence, I wish to establish the following communication protocols: all strategic recommendations shall be presented in memorandum format with executive summary.
ASSISTANT: Understood. I'll format all strategic communications as formal memos.
USER: Furthermore, please ensure all figures are verified before inclusion.""",
        "expected": ["user.communication_style"],
        "description": "Very formal communication preference"
    },
    {
        "id": "casual_1",
        "industry": "Startup",
        "conversation": """USER: yo, so basically we're all about being super transparent with customers, like no corporate BS
ASSISTANT: Got it, authenticity and directness are core to your brand.
USER: yeah exactly, keep it real, no jargon""",
        "expected": ["company.brand_core"],
        "description": "Casual startup brand voice"
    },
    
    # ========== MULTI-LABEL COMPLEX ==========
    {
        "id": "complex_1",
        "industry": "E-commerce",
        "conversation": """USER: I'm the VP of Marketing, and I believe in testing everything. Our brand is playful but professional. We use Klaviyo for email and Shopify for commerce.
ASSISTANT: That's a lot of important context. Let me note all of this.
USER: Yes, and I prefer weekly check-ins on Fridays.""",
        "expected": ["user.role_context", "user.strategic_approach", "company.brand_core", "company.tools_config", "user.workflow_patterns"],
        "description": "Multiple categories in one conversation"
    },
    {
        "id": "complex_2",
        "industry": "SaaS",
        "conversation": """USER: Last month's MRR grew 15% but churn increased to 4.2%. We need to focus on retention this quarter.
ASSISTANT: So Q4 priority is reducing churn rather than new acquisition?
USER: Exactly. All campaigns should emphasize customer success stories.""",
        "expected": ["company.performance_context", "company.business_priorities"],
        "description": "Performance leading to priority shift"
    },
]


def parse_prediction(text):
    if not text or not text.strip():
        return set()
    cats = [c.strip().lower() for c in text.split(",")]
    return {c for c in cats if c in VALID_CATEGORIES}


def compute_metrics(predicted, gold):
    if not predicted and not gold:
        return 1.0, True, True
    if not predicted or not gold:
        return 0.0, False, False
    
    tp = len(predicted & gold)
    prec = tp / len(predicted)
    rec = tp / len(gold)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    any_match = tp > 0
    exact_match = predicted == gold
    
    return f1, any_match, exact_match


async def run_deployment_test():
    print("=" * 70, flush=True)
    print("DEPLOYMENT READINESS TEST", flush=True)
    print("Llama-8B RL Model - Novel Scenarios", flush=True)
    print("=" * 70, flush=True)
    
    # Latest RL checkpoint
    checkpoint = "tinker://4f4bae1f-5a95-5f53-a55a-a14f2872825c:train:0/sampler_weights/rl_iter_012"
    
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=checkpoint)
    tokenizer = get_tokenizer("meta-llama/Llama-3.1-8B")
    renderer = renderers.get_renderer(name="llama3", tokenizer=tokenizer)
    stop = renderer.get_stop_sequences()
    params = types.SamplingParams(max_tokens=100, temperature=0.1, stop=stop)
    
    system = """You route marketing conversations into structured memory categories.

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

    results_by_industry = {}
    total_f1 = 0
    total_any = 0
    total_exact = 0
    
    print(f"\nRunning {len(DEPLOYMENT_TESTS)} test scenarios...\n", flush=True)
    
    for i, test in enumerate(DEPLOYMENT_TESTS):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Analyze this conversation and determine which memory categories apply:\n\n{test['conversation']}"}
        ]
        
        prompt = renderer.build_generation_prompt(messages)
        result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
        response, _ = renderer.parse_response(result.sequences[0].tokens)
        predicted = parse_prediction(response["content"])
        gold = set(test["expected"])
        
        f1, any_match, exact_match = compute_metrics(predicted, gold)
        total_f1 += f1
        total_any += int(any_match)
        total_exact += int(exact_match)
        
        # Track by industry
        industry = test["industry"]
        if industry not in results_by_industry:
            results_by_industry[industry] = {"f1": [], "any": [], "exact": []}
        results_by_industry[industry]["f1"].append(f1)
        results_by_industry[industry]["any"].append(any_match)
        results_by_industry[industry]["exact"].append(exact_match)
        
        status = "✓" if any_match else "✗"
        exact_str = "EXACT" if exact_match else ""
        
        print(f"[{i+1:2d}] {test['industry']:<12} | {test['description'][:40]:<40}", flush=True)
        print(f"     Expected: {sorted(gold)}", flush=True)
        print(f"     Got:      {sorted(predicted)}", flush=True)
        print(f"     {status} F1={f1:.2f} {exact_str}", flush=True)
        print("", flush=True)
    
    # Summary
    n = len(DEPLOYMENT_TESTS)
    print("=" * 70, flush=True)
    print("OVERALL RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(f"Total Tests:  {n}", flush=True)
    print(f"Any Match:    {total_any}/{n} ({total_any/n:.0%})", flush=True)
    print(f"Exact Match:  {total_exact}/{n} ({total_exact/n:.0%})", flush=True)
    print(f"Average F1:   {total_f1/n:.2f}", flush=True)
    
    print("\n" + "-" * 70, flush=True)
    print("RESULTS BY INDUSTRY", flush=True)
    print("-" * 70, flush=True)
    print(f"{'Industry':<15} {'Tests':<8} {'Any Match':<12} {'Exact':<12} {'Avg F1':<10}", flush=True)
    print("-" * 70, flush=True)
    
    for industry in sorted(results_by_industry.keys()):
        data = results_by_industry[industry]
        n_tests = len(data["f1"])
        any_rate = sum(data["any"]) / n_tests
        exact_rate = sum(data["exact"]) / n_tests
        avg_f1 = sum(data["f1"]) / n_tests
        print(f"{industry:<15} {n_tests:<8} {any_rate:<12.0%} {exact_rate:<12.0%} {avg_f1:<10.2f}", flush=True)
    
    # Deployment recommendation
    print("\n" + "=" * 70, flush=True)
    print("DEPLOYMENT RECOMMENDATION", flush=True)
    print("=" * 70, flush=True)
    
    overall_any = total_any / n
    overall_f1 = total_f1 / n
    
    if overall_any >= 0.90 and overall_f1 >= 0.80:
        print("✓ READY FOR DEPLOYMENT", flush=True)
        print(f"  - Any Match rate {overall_any:.0%} exceeds 90% threshold", flush=True)
        print(f"  - Average F1 {overall_f1:.2f} exceeds 0.80 threshold", flush=True)
    elif overall_any >= 0.80 and overall_f1 >= 0.70:
        print("⚠ CONDITIONAL DEPLOYMENT", flush=True)
        print("  - Model performs adequately but may need monitoring", flush=True)
        print("  - Consider additional training on weak categories", flush=True)
    else:
        print("✗ NOT READY FOR DEPLOYMENT", flush=True)
        print("  - Model needs additional training", flush=True)
        print("  - Review failed cases and augment training data", flush=True)


if __name__ == "__main__":
    asyncio.run(run_deployment_test())

