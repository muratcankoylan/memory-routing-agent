"""
Benchmark: Memory Routing Model Evaluation

This script evaluates our trained model against:
1. Base model (untrained Llama-3.1-8B)
2. Our SFT model
3. Our RL model

We measure:
- Classification metrics (F1, precision, recall)
- Task-specific metrics (temporal alignment, scope parity)
- Efficiency (tokens generated, latency)
"""

import asyncio
import json
import time
import os
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    base_model: str = "meta-llama/Llama-3.1-8B"
    renderer_name: str = "llama3"
    test_data_path: str = "training/processed_data/test_data.json"
    output_dir: str = "training/benchmarks"
    
    # Model checkpoints to evaluate
    sft_checkpoint: str = ""
    rl_checkpoint: str = ""


VALID_CATEGORIES = {
    "company.brand_core", "company.strategic_signatures", "company.knowledge_artifacts",
    "company.business_priorities", "company.tools_config", "company.performance_context",
    "user.communication_style", "user.strategic_approach", "user.role_context",
    "user.workflow_patterns", "user.session_history", "user.interaction_preferences",
    "none"
}

CATEGORY_PERSISTENCE = {
    "company.brand_core": "long", "company.strategic_signatures": "long",
    "company.knowledge_artifacts": "long", "company.business_priorities": "short",
    "company.tools_config": "medium", "company.performance_context": "rolling",
    "user.communication_style": "long", "user.strategic_approach": "long",
    "user.role_context": "medium", "user.workflow_patterns": "medium",
    "user.session_history": "short", "user.interaction_preferences": "evolving",
    "none": "short"
}

SYSTEM_PROMPT = """You route marketing conversations into structured memory categories.

Available categories:
- company.brand_core: Voice, values, positioning
- company.strategic_signatures: Decision frameworks
- company.knowledge_artifacts: Docs, style guides
- company.business_priorities: Quarterly goals, campaigns
- company.tools_config: Integrations, settings
- company.performance_context: Campaign metrics
- user.communication_style: Tone, format expectations
- user.strategic_approach: Personal priorities
- user.role_context: Title, scope
- user.workflow_patterns: Review cadence
- user.session_history: Recent context
- user.interaction_preferences: Coaching style
- none: Irrelevant or transactional

Respond with comma-separated categories only. No explanations."""


def parse_prediction(text: str) -> set:
    """Parse model output into category set."""
    if not text:
        return set()
    
    categories = set()
    for part in text.split(","):
        cat = part.strip().lower()
        if cat in VALID_CATEGORIES:
            categories.add(cat)
    
    # Remove "none" if mixed with others
    if "none" in categories and len(categories) > 1:
        categories.discard("none")
    
    return categories


def compute_metrics(predicted: set, gold: set) -> Dict[str, float]:
    """Compute all evaluation metrics for a single example."""
    metrics = {}
    
    # Basic classification
    tp = len(predicted & gold)
    metrics["precision"] = tp / len(predicted) if predicted else 0
    metrics["recall"] = tp / len(gold) if gold else 0
    metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"]) if (metrics["precision"] + metrics["recall"]) > 0 else 0
    metrics["exact_match"] = float(predicted == gold)
    metrics["any_match"] = float(tp > 0)
    
    # Temporal alignment
    def majority_persistence(cats):
        if not cats:
            return "medium"
        persis = [CATEGORY_PERSISTENCE.get(c, "medium") for c in cats]
        return Counter(persis).most_common(1)[0][0]
    
    pred_pers = majority_persistence(predicted)
    gold_pers = majority_persistence(gold)
    metrics["temporal_match"] = float(pred_pers == gold_pers)
    
    # Scope parity
    def get_scope(cats):
        scopes = set()
        for c in cats:
            if c.startswith("company."):
                scopes.add("company")
            elif c.startswith("user."):
                scopes.add("user")
        if len(scopes) == 2:
            return "mixed"
        return scopes.pop() if scopes else "none"
    
    metrics["scope_match"] = float(get_scope(predicted) == get_scope(gold))
    
    # Efficiency
    n = len(predicted)
    metrics["n_categories"] = n
    metrics["efficiency"] = 1.0 if n <= 3 else (0.7 if n == 4 else 0.4)
    
    return metrics


async def evaluate_model(
    service_client, 
    tokenizer, 
    renderer, 
    checkpoint: str,
    test_data: List[Dict],
    model_name: str
) -> Tuple[Dict, List[Dict]]:
    """Evaluate a single model checkpoint."""
    from tinker import types
    
    print(f"\nEvaluating: {model_name}")
    print(f"Checkpoint: {checkpoint}")
    
    sampling_client = service_client.create_sampling_client(model_path=checkpoint)
    stop_sequences = renderer.get_stop_sequences()
    
    results = []
    latencies = []
    
    for i, example in enumerate(test_data):
        gold = set([c.lower() for c in example.get("categories", [])])
        messages = example.get("messages", [])
        prompt_messages = [m for m in messages if m.get("role") != "assistant"]
        
        if not prompt_messages:
            continue
        
        prompt = renderer.build_generation_prompt(prompt_messages)
        params = types.SamplingParams(max_tokens=50, temperature=0.1, stop=stop_sequences)
        
        start_time = time.time()
        result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
        latency = time.time() - start_time
        latencies.append(latency)
        
        response, success = renderer.parse_response(result.sequences[0].tokens)
        predicted_text = response["content"] if success else ""
        predicted = parse_prediction(predicted_text)
        
        metrics = compute_metrics(predicted, gold)
        metrics["gold"] = list(gold)
        metrics["predicted"] = list(predicted)
        metrics["predicted_text"] = predicted_text
        metrics["latency"] = latency
        metrics["format_valid"] = bool(predicted) or predicted_text.strip().lower() == "none"
        
        results.append(metrics)
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(test_data)}")
    
    # Aggregate
    aggregate = {
        "model_name": model_name,
        "checkpoint": checkpoint,
        "n_examples": len(results),
        "f1": np.mean([r["f1"] for r in results]),
        "precision": np.mean([r["precision"] for r in results]),
        "recall": np.mean([r["recall"] for r in results]),
        "exact_match": np.mean([r["exact_match"] for r in results]),
        "any_match": np.mean([r["any_match"] for r in results]),
        "temporal_match": np.mean([r["temporal_match"] for r in results]),
        "scope_match": np.mean([r["scope_match"] for r in results]),
        "efficiency": np.mean([r["efficiency"] for r in results]),
        "format_valid": np.mean([r["format_valid"] for r in results]),
        "mean_latency": np.mean(latencies),
        "p95_latency": np.percentile(latencies, 95),
    }
    
    return aggregate, results


async def run_benchmark(config: BenchmarkConfig):
    """Run full benchmark suite."""
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from dotenv import load_dotenv
    from datetime import datetime
    
    load_dotenv()
    
    print("=" * 70)
    print("MEMORY ROUTING BENCHMARK")
    print("=" * 70)
    
    # Setup
    os.makedirs(config.output_dir, exist_ok=True)
    service_client = tinker.ServiceClient()
    tokenizer = get_tokenizer(config.base_model)
    renderer = renderers.get_renderer(name=config.renderer_name, tokenizer=tokenizer)
    
    # Load test data
    with open(config.test_data_path, "r") as f:
        test_data = json.load(f)
    
    print(f"Test examples: {len(test_data)}")
    
    # Models to evaluate
    models = []
    
    if config.sft_checkpoint:
        models.append(("SFT Model (Llama-3.1-8B + LoRA)", config.sft_checkpoint))
    
    if config.rl_checkpoint:
        models.append(("RL Model (Llama-3.1-8B + LoRA)", config.rl_checkpoint))
    
    # Run evaluations
    all_results = {}
    
    for model_name, checkpoint in models:
        aggregate, details = await evaluate_model(
            service_client, tokenizer, renderer, checkpoint, test_data, model_name
        )
        all_results[model_name] = {
            "aggregate": aggregate,
            "details": details
        }
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    print(f"\n{'Metric':<20} ", end="")
    for model_name in all_results:
        short_name = model_name.split(" (")[0]
        print(f"{short_name:<15} ", end="")
    print()
    print("-" * 70)
    
    metrics_to_show = [
        ("F1 Score", "f1"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("Exact Match", "exact_match"),
        ("Any Match", "any_match"),
        ("Temporal Match", "temporal_match"),
        ("Scope Match", "scope_match"),
        ("Format Valid", "format_valid"),
        ("Mean Latency", "mean_latency"),
    ]
    
    for display_name, key in metrics_to_show:
        print(f"{display_name:<20} ", end="")
        for model_name in all_results:
            value = all_results[model_name]["aggregate"][key]
            if key == "mean_latency":
                print(f"{value:.3f}s         ", end="")
            else:
                print(f"{value:.1%}          ", end="")
        print()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(config.output_dir, f"benchmark_{timestamp}.json")
    
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "base_model": config.base_model,
                "test_examples": len(test_data),
            },
            "results": {k: v["aggregate"] for k, v in all_results.items()},
            "details": {k: v["details"] for k, v in all_results.items()}
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    return all_results


async def main():
    import sys
    
    config = BenchmarkConfig()
    
    # Parse command line args
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            if hasattr(config, key):
                setattr(config, key, value)
    
    await run_benchmark(config)


if __name__ == "__main__":
    asyncio.run(main())

