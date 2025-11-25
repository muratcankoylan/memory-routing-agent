"""
Quick test of the RL model.
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

BASE_MODEL = "meta-llama/Llama-3.1-8B"

# Run with both SFT and RL (most iterations)
RL_CHECKPOINT = "tinker://398393e1-7182-555d-aa1b-7ddf23892338:train:0/sampler_weights/rl_iter_005"

# SFT from the same run
SFT_CHECKPOINT = "tinker://398393e1-7182-555d-aa1b-7ddf23892338:train:0/sampler_weights/sft_final_sampler"

VALID_CATEGORIES = {
    "company.brand_core", "company.strategic_signatures", "company.knowledge_artifacts",
    "company.business_priorities", "company.tools_config", "company.performance_context",
    "user.communication_style", "user.strategic_approach", "user.role_context",
    "user.workflow_patterns", "user.session_history", "user.interaction_preferences",
    "none"
}


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


async def test_model(checkpoint: str, name: str, test_examples: list):
    """Test a model on examples."""
    print(f"\n{'='*60}")
    print(f"TESTING: {name}")
    print(f"Checkpoint: {checkpoint}")
    print(f"{'='*60}")
    
    service_client = tinker.ServiceClient()
    tokenizer = get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(name="llama3", tokenizer=tokenizer)
    
    sampling_client = service_client.create_sampling_client(model_path=checkpoint)
    stop_sequences = renderer.get_stop_sequences()
    
    results = []
    
    for i, example in enumerate(test_examples):
        messages = example.get("messages", [])
        gold = example.get("categories", [])
        
        # Build prompt with system message (matching training format)
        conversation_text = ""
        for m in messages:
            role = m["role"].upper()
            conversation_text += f"{role}: {m['content']}\n"
        
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Conversation:\n{conversation_text}"}
        ]
        
        prompt = renderer.build_generation_prompt(prompt_messages)
        params = types.SamplingParams(max_tokens=100, temperature=0.1, stop=stop_sequences)
        
        result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
        response, success = renderer.parse_response(result.sequences[0].tokens)
        predicted = response["content"] if success else ""
        
        # Parse prediction
        predicted_set = set([c.strip().lower() for c in predicted.split(",") 
                           if c.strip().lower() in VALID_CATEGORIES])
        gold_set = set([c.lower() for c in gold])
        
        any_match = len(predicted_set & gold_set) > 0 if gold_set else (len(predicted_set) == 0)
        exact_match = predicted_set == gold_set
        
        results.append({
            "any_match": any_match,
            "exact_match": exact_match,
            "predicted": predicted,
            "gold": gold
        })
        
        # Show first 5 examples
        if i < 5:
            print(f"\nExample {i+1}:")
            print(f"  Gold: {gold}")
            print(f"  Pred: {predicted}")
            print(f"  Match: {'Yes' if any_match else 'No'}")
    
    # Summary
    any_match_rate = sum(r["any_match"] for r in results) / len(results) if results else 0
    exact_match_rate = sum(r["exact_match"] for r in results) / len(results) if results else 0
    
    print(f"\n--- Results ({len(results)} examples) ---")
    print(f"Any Match:   {any_match_rate:.1%}")
    print(f"Exact Match: {exact_match_rate:.1%}")
    
    return {"any_match": any_match_rate, "exact_match": exact_match_rate}


async def main():
    # First, preprocess data
    print("=" * 60)
    print("LOADING TEST DATA")
    print("=" * 60)
    
    data = []
    with open("synthetic_data/training_dataset_1000.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            messages = []
            for turn in item.get("conversation", []):
                if isinstance(turn, dict):
                    messages.append({"role": turn["role"], "content": turn["content"]})
            
            # Extract categories - handle nested labels structure
            labels = item.get("labels", {})
            if isinstance(labels, dict):
                categories = labels.get("categories", [])
            elif isinstance(labels, list):
                categories = labels
            else:
                categories = []
            
            if not categories:
                # Parse from scenario_id
                scenario_id = item.get("scenario_id", "")
                if "." in scenario_id:
                    cat = scenario_id.split("_")[0]
                    categories = [cat]
            
            data.append({
                "messages": messages,
                "categories": categories
            })
    
    print(f"Total examples: {len(data)}")
    
    # Use last 50 as test
    test_data = data[-50:]
    print(f"Test examples: {len(test_data)}")
    
    # Test RL model
    rl_results = await test_model(RL_CHECKPOINT, "RL Model (5 iters)", test_data)
    
    # Test SFT model for comparison
    sft_results = await test_model(SFT_CHECKPOINT, "SFT Model", test_data)
    
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"SFT Any Match:   {sft_results['any_match']:.1%}")
    print(f"RL Any Match:    {rl_results['any_match']:.1%}")
    print(f"Improvement:     {(rl_results['any_match'] - sft_results['any_match'])*100:+.1f}pp")


if __name__ == "__main__":
    asyncio.run(main())

