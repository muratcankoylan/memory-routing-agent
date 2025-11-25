"""
Final Benchmark: Memory Routing Model Comparison

Compares:
1. Our RL-trained Llama-8B model
2. Base Llama-8B (untrained)
3. Cohere Command-R-Plus (teacher model used for data generation)

All scenarios are marketing-specific and challenging.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import cohere
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

SYSTEM_PROMPT = """You route marketing conversations into structured memory categories.

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

# 50 Challenging Marketing Scenarios
MARKETING_BENCHMARK = [
    # ========== BRAND & POSITIONING (1-8) ==========
    {
        "id": "brand_1",
        "conversation": """USER: Our brand DNA is 'accessible luxury'. We're not Chanel, but we're definitely not Target either. Think Everlane meets Aritzia.
ASSISTANT: So premium quality at attainable price points, with a focus on transparency?
USER: Exactly. Every piece of content should feel elevated but not pretentious.""",
        "expected": ["company.brand_core"],
        "difficulty": "medium",
        "category_focus": "Brand positioning with comparisons"
    },
    {
        "id": "brand_2",
        "conversation": """USER: We've been using 'Innovation for Everyone' as our tagline for 5 years. It's core to who we are.
ASSISTANT: That's a strong positioning statement. Does it extend to your visual identity too?
USER: Yes, our whole design system is built around democratizing technology.""",
        "expected": ["company.brand_core"],
        "difficulty": "easy",
        "category_focus": "Tagline and identity"
    },
    {
        "id": "brand_3",
        "conversation": """USER: I hate when brands try too hard to be funny. Our voice should be warm and helpful, never sarcastic.
ASSISTANT: So authentic helpfulness over trendy humor?
USER: Right. We're the trusted friend, not the class clown.""",
        "expected": ["company.brand_core"],
        "difficulty": "medium",
        "category_focus": "Voice and tone definition"
    },
    {
        "id": "brand_4",
        "conversation": """USER: Our competitors use fear tactics - 'Don't miss out!' We never do that. We believe in empowerment.
ASSISTANT: Positive framing over scarcity messaging.
USER: Always. Even our sales emails should feel like opportunities, not threats.""",
        "expected": ["company.brand_core", "company.strategic_signatures"],
        "difficulty": "hard",
        "category_focus": "Brand values + decision framework"
    },
    {
        "id": "brand_5",
        "conversation": """USER: What font are we using for the Q4 campaign headers?
ASSISTANT: The brand guidelines specify Montserrat for headers.
USER: Thanks, just needed to confirm.""",
        "expected": ["none"],
        "difficulty": "easy",
        "category_focus": "Transactional brand query"
    },
    {
        "id": "brand_6",
        "conversation": """USER: We position ourselves as the 'anti-agency agency'. No jargon, no BS, just results.
ASSISTANT: Transparency and authenticity as differentiators.
USER: And we back it up with our pricing model - no hidden fees, ever.""",
        "expected": ["company.brand_core"],
        "difficulty": "medium",
        "category_focus": "Agency positioning"
    },
    {
        "id": "brand_7",
        "conversation": """USER: Our brand is built on three pillars: sustainability, transparency, and community.
ASSISTANT: Those are strong values. How do they manifest in your marketing?
USER: Every campaign must touch at least one pillar. It's non-negotiable.""",
        "expected": ["company.brand_core", "company.strategic_signatures"],
        "difficulty": "hard",
        "category_focus": "Brand pillars as decision framework"
    },
    {
        "id": "brand_8",
        "conversation": """USER: We're rebranding next quarter but I can't share details yet.
ASSISTANT: Understood. Let me know when you're ready to discuss.
USER: Will do.""",
        "expected": ["none"],
        "difficulty": "medium",
        "category_focus": "Vague future reference"
    },

    # ========== STRATEGIC DECISIONS (9-16) ==========
    {
        "id": "strategy_1",
        "conversation": """USER: We never launch a campaign without at least 3 weeks of testing. It's our golden rule.
ASSISTANT: So testing is a non-negotiable gate in your process?
USER: Absolutely. Even if stakeholders push back, we hold the line.""",
        "expected": ["company.strategic_signatures"],
        "difficulty": "easy",
        "category_focus": "Decision framework"
    },
    {
        "id": "strategy_2",
        "conversation": """USER: Our philosophy is 'measure twice, cut once'. We'd rather delay a launch than ship something half-baked.
ASSISTANT: Quality over speed.
USER: Every time. Our reputation depends on it.""",
        "expected": ["company.strategic_signatures"],
        "difficulty": "medium",
        "category_focus": "Strategic heuristic"
    },
    {
        "id": "strategy_3",
        "conversation": """USER: When in doubt, we default to the customer's perspective. What would they want?
ASSISTANT: Customer-centric decision making.
USER: It's saved us from many internal-focused mistakes.""",
        "expected": ["company.strategic_signatures"],
        "difficulty": "medium",
        "category_focus": "Decision principle"
    },
    {
        "id": "strategy_4",
        "conversation": """USER: I personally believe in testing everything. Data over gut instinct.
ASSISTANT: A data-driven approach.
USER: It's how I've always operated. Even small decisions get A/B tested.""",
        "expected": ["user.strategic_approach"],
        "difficulty": "medium",
        "category_focus": "Personal vs company strategy"
    },
    {
        "id": "strategy_5",
        "conversation": """USER: The company prioritizes brand safety above all. But I think we're too conservative sometimes.
ASSISTANT: So there's tension between company policy and your personal view?
USER: Yes. I'd take more calculated risks if it were up to me.""",
        "expected": ["company.strategic_signatures", "user.strategic_approach"],
        "difficulty": "hard",
        "category_focus": "Company vs personal strategy conflict"
    },
    {
        "id": "strategy_6",
        "conversation": """USER: We have a 'no surprises' policy with clients. They hear bad news from us first.
ASSISTANT: Proactive communication as a core principle.
USER: It's built trust with every client we've worked with.""",
        "expected": ["company.strategic_signatures"],
        "difficulty": "medium",
        "category_focus": "Client relationship principle"
    },
    {
        "id": "strategy_7",
        "conversation": """USER: Should we go with vendor A or vendor B for the email platform?
ASSISTANT: What are the key differences you're weighing?
USER: Just cost really. They're pretty similar otherwise.""",
        "expected": ["none"],
        "difficulty": "easy",
        "category_focus": "Transactional decision"
    },
    {
        "id": "strategy_8",
        "conversation": """USER: Our rule is simple: if we can't explain it to a 10-year-old, we don't publish it.
ASSISTANT: Clarity as a content standard.
USER: It keeps our messaging accessible to everyone.""",
        "expected": ["company.strategic_signatures", "company.brand_core"],
        "difficulty": "hard",
        "category_focus": "Content principle overlapping brand"
    },

    # ========== PERFORMANCE & METRICS (17-24) ==========
    {
        "id": "perf_1",
        "conversation": """USER: Last quarter's email campaign hit 42% open rate and 8% CTR. Best we've ever done.
ASSISTANT: Those are strong numbers. What drove the improvement?
USER: We switched to personalized subject lines based on past behavior.""",
        "expected": ["company.performance_context"],
        "difficulty": "easy",
        "category_focus": "Campaign metrics"
    },
    {
        "id": "perf_2",
        "conversation": """USER: Our CAC has been climbing - $180 last month, up from $120 in Q1.
ASSISTANT: That's a significant increase. Any hypothesis on the cause?
USER: iOS privacy changes are killing our Facebook targeting.""",
        "expected": ["company.performance_context"],
        "difficulty": "medium",
        "category_focus": "Acquisition metrics with context"
    },
    {
        "id": "perf_3",
        "conversation": """USER: The influencer campaign flopped. 2M impressions but only 50 conversions.
ASSISTANT: That's a 0.0025% conversion rate. What went wrong?
USER: Wrong audience. The influencer's followers weren't our target demo.""",
        "expected": ["company.performance_context"],
        "difficulty": "medium",
        "category_focus": "Campaign failure analysis"
    },
    {
        "id": "perf_4",
        "conversation": """USER: We learned that video content outperforms static by 3x on engagement.
ASSISTANT: That's a significant insight. Are you shifting your content mix?
USER: Yes, Q4 will be 70% video. It's a major strategic shift.""",
        "expected": ["company.performance_context", "company.business_priorities"],
        "difficulty": "hard",
        "category_focus": "Learning leading to priority"
    },
    {
        "id": "perf_5",
        "conversation": """USER: What was our email open rate last week?
ASSISTANT: I don't have access to that data in real-time.
USER: I'll check the dashboard.""",
        "expected": ["none"],
        "difficulty": "easy",
        "category_focus": "Data lookup request"
    },
    {
        "id": "perf_6",
        "conversation": """USER: Our NPS dropped from 72 to 58 after the website redesign.
ASSISTANT: That's a significant drop. Customer feedback?
USER: They hate the new navigation. We're rolling back next week.""",
        "expected": ["company.performance_context"],
        "difficulty": "medium",
        "category_focus": "Customer satisfaction metrics"
    },
    {
        "id": "perf_7",
        "conversation": """USER: The podcast is our best performing channel - 40% of qualified leads come from it.
ASSISTANT: That's impressive attribution. How are you tracking it?
USER: Custom UTM parameters and post-conversion surveys.""",
        "expected": ["company.performance_context"],
        "difficulty": "medium",
        "category_focus": "Channel performance"
    },
    {
        "id": "perf_8",
        "conversation": """USER: We've tested 47 different landing page variants this year.
ASSISTANT: That's rigorous testing. What's been the biggest winner?
USER: Social proof above the fold. Increased conversions by 34%.""",
        "expected": ["company.performance_context", "company.knowledge_artifacts"],
        "difficulty": "hard",
        "category_focus": "Testing insights becoming knowledge"
    },

    # ========== TOOLS & INTEGRATIONS (25-30) ==========
    {
        "id": "tools_1",
        "conversation": """USER: We use HubSpot for CRM, Klaviyo for email, and Shopify for commerce. Everything needs to sync.
ASSISTANT: What's your current integration setup?
USER: Zapier connects them, but we're moving to native integrations.""",
        "expected": ["company.tools_config"],
        "difficulty": "easy",
        "category_focus": "Tech stack overview"
    },
    {
        "id": "tools_2",
        "conversation": """USER: Our API key for the analytics platform is expiring next month.
ASSISTANT: I'll note that for renewal.
USER: Thanks. It's critical - everything breaks without it.""",
        "expected": ["company.tools_config"],
        "difficulty": "medium",
        "category_focus": "API configuration"
    },
    {
        "id": "tools_3",
        "conversation": """USER: We just migrated from Mailchimp to Klaviyo. The segmentation is so much better.
ASSISTANT: What segments are you using?
USER: RFM-based: high-value, at-risk, and dormant customers.""",
        "expected": ["company.tools_config"],
        "difficulty": "medium",
        "category_focus": "Tool migration and usage"
    },
    {
        "id": "tools_4",
        "conversation": """USER: Can you remind me how to export data from Google Analytics?
ASSISTANT: Go to Reports, select your date range, then click Export.
USER: Got it, thanks.""",
        "expected": ["none"],
        "difficulty": "easy",
        "category_focus": "How-to question"
    },
    {
        "id": "tools_5",
        "conversation": """USER: Our Slack is integrated with everything - Asana, HubSpot, even our custom dashboards.
ASSISTANT: Centralized notifications.
USER: It's our command center. All marketing alerts go there.""",
        "expected": ["company.tools_config"],
        "difficulty": "medium",
        "category_focus": "Workflow integrations"
    },
    {
        "id": "tools_6",
        "conversation": """USER: We're evaluating Notion vs Confluence for our marketing wiki.
ASSISTANT: What are your key requirements?
USER: Easy editing and good search. Our current docs are a mess.""",
        "expected": ["none"],
        "difficulty": "medium",
        "category_focus": "Tool evaluation (not decided)"
    },

    # ========== USER PREFERENCES & ROLE (31-40) ==========
    {
        "id": "user_1",
        "conversation": """USER: I'm the VP of Growth, reporting to the CMO. I own all acquisition channels.
ASSISTANT: That's a broad scope. What's your team size?
USER: 12 people across paid, organic, and partnerships.""",
        "expected": ["user.role_context"],
        "difficulty": "easy",
        "category_focus": "Role and scope"
    },
    {
        "id": "user_2",
        "conversation": """USER: I prefer bullet points over paragraphs. Get to the point quickly.
ASSISTANT: Noted. I'll keep responses concise and scannable.
USER: Perfect. Time is my scarcest resource.""",
        "expected": ["user.communication_style"],
        "difficulty": "easy",
        "category_focus": "Format preference"
    },
    {
        "id": "user_3",
        "conversation": """USER: Don't sugarcoat things. If something's not working, tell me directly.
ASSISTANT: Direct feedback, even when it's uncomfortable.
USER: Especially when it's uncomfortable. That's when I need it most.""",
        "expected": ["user.interaction_preferences"],
        "difficulty": "medium",
        "category_focus": "Feedback style"
    },
    {
        "id": "user_4",
        "conversation": """USER: I review all campaigns on Friday afternoons. That's when I have focus time.
ASSISTANT: I'll schedule campaign reviews for Fridays then.
USER: And send me a summary Thursday evening so I can prep.""",
        "expected": ["user.workflow_patterns"],
        "difficulty": "medium",
        "category_focus": "Review cadence"
    },
    {
        "id": "user_5",
        "conversation": """USER: Yesterday we discussed the influencer strategy. Can you pull up those notes?
ASSISTANT: Yes, we covered micro-influencer targeting and budget allocation.
USER: Right. Let's continue from the budget discussion.""",
        "expected": ["user.session_history"],
        "difficulty": "easy",
        "category_focus": "Session continuity"
    },
    {
        "id": "user_6",
        "conversation": """USER: I believe in 'strong opinions, loosely held'. I'll argue a point but change my mind with good data.
ASSISTANT: Intellectually flexible but not wishy-washy.
USER: Exactly. Challenge me when you think I'm wrong.""",
        "expected": ["user.strategic_approach", "user.interaction_preferences"],
        "difficulty": "hard",
        "category_focus": "Personal philosophy + interaction style"
    },
    {
        "id": "user_7",
        "conversation": """USER: How's your day going?
ASSISTANT: Going well, thanks! How can I help you today?
USER: Just checking in before our meeting.""",
        "expected": ["none"],
        "difficulty": "easy",
        "category_focus": "Small talk"
    },
    {
        "id": "user_8",
        "conversation": """USER: I'm a visual thinker. Show me charts and diagrams whenever possible.
ASSISTANT: I'll prioritize visual representations in my responses.
USER: And color-code things by priority if you can.""",
        "expected": ["user.communication_style"],
        "difficulty": "medium",
        "category_focus": "Visual preference"
    },
    {
        "id": "user_9",
        "conversation": """USER: I've been in marketing for 15 years, so skip the basics.
ASSISTANT: I'll assume expert-level knowledge.
USER: Good. I hate when tools explain obvious concepts.""",
        "expected": ["user.interaction_preferences", "user.role_context"],
        "difficulty": "hard",
        "category_focus": "Experience level + interaction"
    },
    {
        "id": "user_10",
        "conversation": """USER: My success metric is pipeline contribution. Everything else is vanity.
ASSISTANT: Revenue-focused measurement.
USER: If it doesn't move pipeline, I don't care about it.""",
        "expected": ["user.strategic_approach"],
        "difficulty": "medium",
        "category_focus": "Personal success definition"
    },

    # ========== BUSINESS PRIORITIES (41-46) ==========
    {
        "id": "priority_1",
        "conversation": """USER: Q4 is all about Black Friday and Cyber Monday. Everything else takes a back seat.
ASSISTANT: So BFCM is the top priority through year-end?
USER: 60% of our annual revenue happens in those two weeks.""",
        "expected": ["company.business_priorities"],
        "difficulty": "easy",
        "category_focus": "Seasonal priority"
    },
    {
        "id": "priority_2",
        "conversation": """USER: We're launching a new product line next month. All marketing resources are shifting to support it.
ASSISTANT: What's the launch timeline?
USER: Soft launch March 1st, full campaign March 15th.""",
        "expected": ["company.business_priorities"],
        "difficulty": "medium",
        "category_focus": "Product launch priority"
    },
    {
        "id": "priority_3",
        "conversation": """USER: The board wants us to focus on profitability over growth this year.
ASSISTANT: A shift from growth-at-all-costs.
USER: Yes. CAC payback under 12 months is now mandatory.""",
        "expected": ["company.business_priorities", "company.strategic_signatures"],
        "difficulty": "hard",
        "category_focus": "Strategic priority shift"
    },
    {
        "id": "priority_4",
        "conversation": """USER: We need to hit 10,000 email subscribers by end of quarter.
ASSISTANT: What's the current count?
USER: 6,500. So we need about 120 new subs per day.""",
        "expected": ["company.business_priorities"],
        "difficulty": "medium",
        "category_focus": "Specific goal"
    },
    {
        "id": "priority_5",
        "conversation": """USER: Retention is our #1 focus this year. We've been leaking customers.
ASSISTANT: Churn reduction over acquisition.
USER: We're pausing all acquisition campaigns until we fix the retention problem.""",
        "expected": ["company.business_priorities"],
        "difficulty": "medium",
        "category_focus": "Strategic focus"
    },
    {
        "id": "priority_6",
        "conversation": """USER: What should we prioritize next quarter?
ASSISTANT: That depends on your goals. What's most important to the business?
USER: I'm not sure yet. Still figuring it out.""",
        "expected": ["none"],
        "difficulty": "medium",
        "category_focus": "Undefined priority"
    },

    # ========== KNOWLEDGE ARTIFACTS (47-50) ==========
    {
        "id": "knowledge_1",
        "conversation": """USER: We have a 50-page brand book that every agency partner must follow.
ASSISTANT: Comprehensive brand guidelines.
USER: It covers everything from logo usage to photography style.""",
        "expected": ["company.knowledge_artifacts"],
        "difficulty": "easy",
        "category_focus": "Brand documentation"
    },
    {
        "id": "knowledge_2",
        "conversation": """USER: Our content playbook defines the exact process from ideation to publication.
ASSISTANT: A documented workflow.
USER: Every piece of content goes through the same 7-step process.""",
        "expected": ["company.knowledge_artifacts"],
        "difficulty": "medium",
        "category_focus": "Process documentation"
    },
    {
        "id": "knowledge_3",
        "conversation": """USER: We have templates for every email type - welcome, abandoned cart, win-back, you name it.
ASSISTANT: A comprehensive email template library.
USER: It's saved us hundreds of hours. New team members can start producing immediately.""",
        "expected": ["company.knowledge_artifacts"],
        "difficulty": "medium",
        "category_focus": "Template library"
    },
    {
        "id": "knowledge_4",
        "conversation": """USER: Our style guide says we never use exclamation marks in headlines.
ASSISTANT: A specific editorial rule.
USER: It's part of our understated brand voice.""",
        "expected": ["company.knowledge_artifacts", "company.brand_core"],
        "difficulty": "hard",
        "category_focus": "Style guide overlapping brand"
    },
]


def parse_prediction(text):
    """Parse model output into category set."""
    if not text or not text.strip():
        return set()
    # Handle various formats
    text = text.lower().strip()
    # Remove common prefixes
    for prefix in ["categories:", "category:", "the categories are:", "answer:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    cats = [c.strip() for c in text.split(",")]
    return {c for c in cats if c in VALID_CATEGORIES}


def compute_metrics(predicted, gold):
    """Compute F1, precision, recall."""
    if not predicted and not gold:
        return 1.0, 1.0, 1.0, True, True
    if not predicted or not gold:
        return 0.0, 0.0, 0.0, False, False
    
    tp = len(predicted & gold)
    precision = tp / len(predicted)
    recall = tp / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    any_match = tp > 0
    exact_match = predicted == gold
    
    return f1, precision, recall, any_match, exact_match


async def eval_tinker_model(name, checkpoint, model_name, renderer_name):
    """Evaluate a Tinker model."""
    print(f"\n{'='*60}", flush=True)
    print(f"Evaluating: {name}", flush=True)
    print(f"{'='*60}", flush=True)
    
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=checkpoint)
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)
    stop = renderer.get_stop_sequences()
    params = types.SamplingParams(max_tokens=100, temperature=0.1, stop=stop)
    
    results = []
    
    for i, test in enumerate(MARKETING_BENCHMARK):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this conversation and determine which memory categories apply:\n\n{test['conversation']}"}
        ]
        
        prompt = renderer.build_generation_prompt(messages)
        result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
        response, _ = renderer.parse_response(result.sequences[0].tokens)
        predicted = parse_prediction(response["content"])
        gold = set(test["expected"])
        
        f1, prec, rec, any_match, exact = compute_metrics(predicted, gold)
        
        results.append({
            "id": test["id"],
            "predicted": list(predicted),
            "gold": list(gold),
            "f1": f1,
            "any_match": any_match,
            "exact_match": exact,
            "difficulty": test["difficulty"]
        })
        
        status = "✓" if any_match else "✗"
        print(f"[{i+1:2d}] {status} {test['id']:<15} F1={f1:.2f}", flush=True)
    
    return results


async def eval_cohere_model():
    """Evaluate Cohere Command-R-Plus (teacher model)."""
    print(f"\n{'='*60}", flush=True)
    print(f"Evaluating: Cohere Command-R-Plus (Teacher)", flush=True)
    print(f"{'='*60}", flush=True)
    
    client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    
    results = []
    
    for i, test in enumerate(MARKETING_BENCHMARK):
        prompt = f"""{SYSTEM_PROMPT}

Analyze this conversation and determine which memory categories apply:

{test['conversation']}

Respond with comma-separated categories only. No explanation."""

        try:
            response = client.chat(
                model="command-r-plus-08-2024",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            
            # Extract text from response
            response_text = ""
            if hasattr(response.message, 'content') and response.message.content:
                for block in response.message.content:
                    if hasattr(block, 'text'):
                        response_text = block.text
                        break
            
            predicted = parse_prediction(response_text)
            gold = set(test["expected"])
            
            f1, prec, rec, any_match, exact = compute_metrics(predicted, gold)
            
            results.append({
                "id": test["id"],
                "predicted": list(predicted),
                "gold": list(gold),
                "f1": f1,
                "any_match": any_match,
                "exact_match": exact,
                "difficulty": test["difficulty"]
            })
            
            status = "✓" if any_match else "✗"
            print(f"[{i+1:2d}] {status} {test['id']:<15} F1={f1:.2f}", flush=True)
            
            # Rate limiting
            await asyncio.sleep(0.5)
            
        except Exception as e:
            print(f"[{i+1:2d}] ERROR {test['id']}: {e}", flush=True)
            results.append({
                "id": test["id"],
                "predicted": [],
                "gold": list(test["expected"]),
                "f1": 0.0,
                "any_match": False,
                "exact_match": False,
                "difficulty": test["difficulty"],
                "error": str(e)
            })
    
    return results


def compute_summary(results, name):
    """Compute summary statistics."""
    n = len(results)
    avg_f1 = sum(r["f1"] for r in results) / n
    any_match = sum(1 for r in results if r["any_match"]) / n
    exact_match = sum(1 for r in results if r["exact_match"]) / n
    
    # By difficulty
    by_diff = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            by_diff[diff] = {
                "count": len(subset),
                "f1": sum(r["f1"] for r in subset) / len(subset),
                "any_match": sum(1 for r in subset if r["any_match"]) / len(subset),
                "exact_match": sum(1 for r in subset if r["exact_match"]) / len(subset)
            }
    
    return {
        "name": name,
        "total": n,
        "avg_f1": avg_f1,
        "any_match": any_match,
        "exact_match": exact_match,
        "by_difficulty": by_diff
    }


async def main():
    print("=" * 70, flush=True)
    print("FINAL BENCHMARK: Memory Routing Model Comparison", flush=True)
    print("50 Challenging Marketing Scenarios", flush=True)
    print("=" * 70, flush=True)
    
    all_results = {}
    
    # 1. Our RL-trained model
    rl_results = await eval_tinker_model(
        name="Llama-8B + LoRA + RL (Ours)",
        checkpoint="tinker://4f4bae1f-5a95-5f53-a55a-a14f2872825c:train:0/sampler_weights/rl_iter_012",
        model_name="meta-llama/Llama-3.1-8B",
        renderer_name="llama3"
    )
    all_results["rl_model"] = rl_results
    
    # 2. Cohere Command-R-Plus (Teacher)
    cohere_results = await eval_cohere_model()
    all_results["cohere"] = cohere_results
    
    # Compute summaries
    summaries = {
        "rl_model": compute_summary(rl_results, "Llama-8B + LoRA + RL (Ours)"),
        "cohere": compute_summary(cohere_results, "Cohere Command-R-Plus (104B)")
    }
    
    # Print comparison
    print("\n" + "=" * 70, flush=True)
    print("BENCHMARK RESULTS", flush=True)
    print("=" * 70, flush=True)
    
    print(f"\n{'Model':<35} {'Any Match':<12} {'Exact':<12} {'Avg F1':<10}", flush=True)
    print("-" * 70, flush=True)
    
    for key, summary in summaries.items():
        print(f"{summary['name']:<35} {summary['any_match']:<12.0%} {summary['exact_match']:<12.0%} {summary['avg_f1']:<10.2f}", flush=True)
    
    print("\n" + "-" * 70, flush=True)
    print("RESULTS BY DIFFICULTY", flush=True)
    print("-" * 70, flush=True)
    
    for diff in ["easy", "medium", "hard"]:
        print(f"\n{diff.upper()}:", flush=True)
        for key, summary in summaries.items():
            if diff in summary["by_difficulty"]:
                d = summary["by_difficulty"][diff]
                print(f"  {summary['name']:<33} Any={d['any_match']:.0%} Exact={d['exact_match']:.0%} F1={d['f1']:.2f} (n={d['count']})", flush=True)
    
    # Save results
    output = {
        "benchmark_date": datetime.now().isoformat(),
        "num_scenarios": len(MARKETING_BENCHMARK),
        "summaries": summaries,
        "detailed_results": all_results
    }
    
    os.makedirs("training/benchmarks", exist_ok=True)
    output_path = f"training/benchmarks/final_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}", flush=True)
    
    # Key findings
    print("\n" + "=" * 70, flush=True)
    print("KEY FINDINGS", flush=True)
    print("=" * 70, flush=True)
    
    rl_f1 = summaries["rl_model"]["avg_f1"]
    cohere_f1 = summaries["cohere"]["avg_f1"]
    
    if rl_f1 > cohere_f1:
        improvement = ((rl_f1 - cohere_f1) / cohere_f1) * 100
        print(f"✓ Our 8B model OUTPERFORMS the 104B teacher by {improvement:.1f}% on F1", flush=True)
    else:
        gap = ((cohere_f1 - rl_f1) / cohere_f1) * 100
        print(f"  Our 8B model is within {gap:.1f}% of the 104B teacher on F1", flush=True)
    
    print(f"\nModel Sizes:", flush=True)
    print(f"  - Llama-8B + LoRA: ~8B parameters (LoRA adds ~0.1B)", flush=True)
    print(f"  - Cohere Command-R-Plus: ~104B parameters", flush=True)
    print(f"  - Size ratio: 13x smaller", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

