"""Quick evaluation script - shows results as they come in."""

import asyncio
import json
import os
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

def parse_prediction(text):
    if not text or not text.strip():
        return set()
    cats = [c.strip().lower() for c in text.split(",")]
    return {c for c in cats if c in VALID_CATEGORIES}

def compute_f1(predicted, gold):
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    tp = len(predicted & gold)
    prec = tp / len(predicted)
    rec = tp / len(gold)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

async def eval_model(name, checkpoint, model_name, renderer_name, test_data, n=10):
    print(f"\n{'='*60}", flush=True)
    print(f"EVALUATING: {name}", flush=True)
    print(f"Checkpoint: {checkpoint}", flush=True)
    print(f"{'='*60}", flush=True)
    
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=checkpoint)
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)
    stop = renderer.get_stop_sequences()
    params = types.SamplingParams(max_tokens=100, temperature=0.1, stop=stop)
    
    correct = exact = total_f1 = 0
    
    for i, item in enumerate(test_data[:n]):
        messages = item["messages"][:-1]
        gold = set(item["categories"])
        
        prompt = renderer.build_generation_prompt(messages)
        result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
        response, _ = renderer.parse_response(result.sequences[0].tokens)
        predicted = parse_prediction(response["content"])
        
        f1 = compute_f1(predicted, gold)
        total_f1 += f1
        if predicted & gold:
            correct += 1
        if predicted == gold:
            exact += 1
        
        status = "✓" if predicted & gold else "✗"
        ex = "EXACT" if predicted == gold else ""
        print(f"[{i+1:2d}] {status} Gold: {sorted(gold)} | Pred: {sorted(predicted)} | F1={f1:.2f} {ex}", flush=True)
    
    print(f"\n--- SUMMARY ({n} examples) ---", flush=True)
    print(f"Any Match:   {correct}/{n} ({correct/n:.0%})", flush=True)
    print(f"Exact Match: {exact}/{n} ({exact/n:.0%})", flush=True)
    print(f"Avg F1:      {total_f1/n:.2f}", flush=True)
    
    return {"any": correct/n, "exact": exact/n, "f1": total_f1/n}

async def main():
    # Load test data
    with open("training/processed_data/test_data.json", "r") as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test examples", flush=True)
    
    # Evaluate Llama-8B RL (latest)
    llama_result = await eval_model(
        name="Llama-8B RL (iter 12)",
        checkpoint="tinker://4f4bae1f-5a95-5f53-a55a-a14f2872825c:train:0/sampler_weights/rl_iter_012",
        model_name="meta-llama/Llama-3.1-8B",
        renderer_name="llama3",
        test_data=test_data,
        n=15
    )
    
    # Evaluate Qwen3-32B SFT
    qwen_result = await eval_model(
        name="Qwen3-32B SFT (step 30)",
        checkpoint="tinker://b7be2502-e321-59ee-9477-f3fd8a52ab4e:train:0/sampler_weights/sft_step_0030",
        model_name="Qwen/Qwen3-32B",
        renderer_name="qwen3",
        test_data=test_data,
        n=15
    )
    
    # Comparison
    print(f"\n{'='*60}", flush=True)
    print("COMPARISON", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Model':<25} {'Any Match':<12} {'Exact':<12} {'F1':<10}", flush=True)
    print("-" * 60, flush=True)
    print(f"{'Llama-8B RL':<25} {llama_result['any']:<12.0%} {llama_result['exact']:<12.0%} {llama_result['f1']:<10.2f}", flush=True)
    print(f"{'Qwen3-32B SFT':<25} {qwen_result['any']:<12.0%} {qwen_result['exact']:<12.0%} {qwen_result['f1']:<10.2f}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())

