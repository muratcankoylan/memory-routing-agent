"""
Evaluate the SFT model and run RL continuation.

This script:
1. Evaluates the SFT checkpoint from our full_pipeline run
2. Continues RL training from the SFT state checkpoint
3. Evaluates the final RL model
"""

import asyncio
import json
import os
import time
import numpy as np
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Configuration
BASE_MODEL = "meta-llama/Llama-3.1-8B"
LORA_RANK = 32

# Checkpoints from our full_pipeline SFT run
SFT_STATE_CHECKPOINT = "tinker://398393e1-7182-555d-aa1b-7ddf23892338:train:0/weights/sft_final"
SFT_SAMPLER_CHECKPOINT = "tinker://398393e1-7182-555d-aa1b-7ddf23892338:train:0/sampler_weights/sft_final_sampler"

# RL Configuration
RL_ITERATIONS = 10
RL_BATCH_SIZE = 16
RL_GROUP_SIZE = 4
RL_LR = 1e-5
RL_TEMPERATURE = 0.7

# Data paths
TRAIN_DATA_PATH = "training/processed_data/train_data.json"
TEST_DATA_PATH = "training/processed_data/test_data.json"

# Memory taxonomy
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


def compute_reward(predicted_text: str, gold_categories: list) -> tuple:
    """Compute reward with detailed breakdown."""
    info = {"format_valid": True, "r_f1": 0, "r_temp": 0, "r_parity": 0, "r_eff": 0}
    
    if not predicted_text or not predicted_text.strip():
        info["format_valid"] = False
        return -1.0, info
    
    predicted = set([c.strip().lower() for c in predicted_text.split(",") 
                     if c.strip().lower() in VALID_CATEGORIES])
    
    if not predicted:
        info["format_valid"] = False
        return -1.0, info
    
    if "none" in predicted and len(predicted) > 1:
        predicted.discard("none")
    
    gold = set([c.lower() for c in gold_categories])
    
    # F1 Score
    if predicted and gold:
        tp = len(predicted & gold)
        precision = tp / len(predicted)
        recall = tp / len(gold)
        info["r_f1"] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    elif not predicted and not gold:
        info["r_f1"] = 1.0
    
    # Temporal alignment
    def majority_persistence(cats):
        if not cats:
            return "medium"
        persis = [CATEGORY_PERSISTENCE.get(c, "medium") for c in cats]
        return Counter(persis).most_common(1)[0][0]
    
    if majority_persistence(predicted) == majority_persistence(gold):
        info["r_temp"] = 1.0
    
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
    
    if get_scope(predicted) == get_scope(gold):
        info["r_parity"] = 1.0
    
    # Efficiency
    n = len(predicted)
    info["r_eff"] = 1.0 if n <= 3 else (0.7 if n == 4 else 0.4)
    
    r_total = 0.6 * info["r_f1"] + 0.2 * info["r_temp"] + 0.1 * info["r_parity"] + 0.1 * info["r_eff"]
    return r_total, info


async def evaluate_model(service_client, checkpoint, tokenizer, renderer, test_data, name, n_samples=100):
    """Evaluate a model checkpoint."""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {name}")
    print(f"{'='*60}")
    
    sampling_client = service_client.create_sampling_client(model_path=checkpoint)
    stop_sequences = renderer.get_stop_sequences()
    
    results = []
    
    for i, example in enumerate(test_data[:n_samples]):
        gold = example.get("categories", [])
        messages = example.get("messages", [])
        prompt_messages = [m for m in messages if m.get("role") != "assistant"]
        
        if not prompt_messages:
            continue
        
        prompt = renderer.build_generation_prompt(prompt_messages)
        params = types.SamplingParams(max_tokens=50, temperature=0.1, stop=stop_sequences)
        
        result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
        response, success = renderer.parse_response(result.sequences[0].tokens)
        predicted_text = response["content"] if success else ""
        
        predicted_set = set([c.strip().lower() for c in predicted_text.split(",") 
                           if c.strip().lower() in VALID_CATEGORIES])
        gold_set = set([c.lower() for c in gold])
        
        reward, info = compute_reward(predicted_text, gold)
        
        results.append({
            "any_match": len(predicted_set & gold_set) > 0,
            "exact_match": predicted_set == gold_set,
            "precision": len(predicted_set & gold_set) / len(predicted_set) if predicted_set else 0,
            "recall": len(predicted_set & gold_set) / len(gold_set) if gold_set else 0,
            "reward": reward,
            "format_valid": info["format_valid"]
        })
        
        if (i + 1) % 25 == 0:
            any_match = np.mean([r["any_match"] for r in results])
            print(f"  Progress: {i+1}/{n_samples}, Any Match: {any_match:.1%}")
    
    metrics = {
        "any_match": np.mean([r["any_match"] for r in results]),
        "exact_match": np.mean([r["exact_match"] for r in results]),
        "precision": np.mean([r["precision"] for r in results]),
        "recall": np.mean([r["recall"] for r in results]),
        "mean_reward": np.mean([r["reward"] for r in results]),
        "format_valid": np.mean([r["format_valid"] for r in results]),
    }
    metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"]) if (metrics["precision"] + metrics["recall"]) > 0 else 0
    
    print(f"\nResults for {name}:")
    print(f"  Any Match:   {metrics['any_match']:.1%}")
    print(f"  Exact Match: {metrics['exact_match']:.1%}")
    print(f"  F1 Score:    {metrics['f1']:.1%}")
    print(f"  Mean Reward: {metrics['mean_reward']:.3f}")
    
    return metrics


async def run_rl_phase(service_client, training_client, tokenizer, renderer, train_data):
    """Run RL training phase."""
    print(f"\n{'='*60}")
    print("PHASE 2: REINFORCEMENT LEARNING")
    print(f"{'='*60}")
    
    print(f"Loading SFT state from: {SFT_STATE_CHECKPOINT}")
    await training_client.load_state_async(SFT_STATE_CHECKPOINT)
    print("SFT weights loaded successfully!")
    
    stop_sequences = renderer.get_stop_sequences()
    metrics_log = []
    
    for iteration in range(RL_ITERATIONS):
        iter_start = time.time()
        print(f"\n--- RL Iteration {iteration + 1}/{RL_ITERATIONS} ---")
        
        # Save current weights for sampling
        save_future = await training_client.save_weights_for_sampler_async(
            name=f"rl_iter_{iteration:03d}"
        )
        save_result = await save_future.result_async()
        sampling_client = service_client.create_sampling_client(model_path=save_result.path)
        
        # Sample batch
        batch_indices = np.random.choice(len(train_data), size=RL_BATCH_SIZE, replace=False)
        
        all_rollouts = []
        all_rewards = []
        
        for idx in batch_indices:
            example = train_data[idx]
            gold_categories = example.get("categories", [])
            messages = example.get("messages", [])
            prompt_messages = [m for m in messages if m.get("role") != "assistant"]
            
            if not prompt_messages:
                continue
            
            prompt = renderer.build_generation_prompt(prompt_messages)
            params = types.SamplingParams(
                max_tokens=50, temperature=RL_TEMPERATURE, stop=stop_sequences
            )
            
            result = sampling_client.sample(
                prompt=prompt, sampling_params=params, num_samples=RL_GROUP_SIZE
            ).result()
            
            for seq in result.sequences:
                response, success = renderer.parse_response(seq.tokens)
                predicted = response["content"] if success else ""
                reward, _ = compute_reward(predicted, gold_categories)
                
                all_rollouts.append({
                    "prompt": prompt,
                    "tokens": seq.tokens,
                    "logprobs": seq.logprobs or [],
                    "predicted": predicted,
                    "gold": gold_categories
                })
                all_rewards.append(reward)
        
        # Compute advantages
        rewards_arr = np.array(all_rewards)
        mean_reward = rewards_arr.mean()
        std_reward = rewards_arr.std() + 1e-8
        advantages = (rewards_arr - mean_reward) / std_reward
        
        # Build training data
        training_data = []
        for i, rollout in enumerate(all_rollouts):
            if not rollout["logprobs"]:
                continue
            
            prompt_tokens = rollout["prompt"].to_ints()
            gen_tokens = rollout["tokens"]
            logprobs = rollout["logprobs"]
            adv = advantages[i]
            
            n_prompt = len(prompt_tokens) - 1
            n_gen = len(gen_tokens)
            
            if len(logprobs) != n_gen:
                continue
            
            full_input = prompt_tokens + gen_tokens[:-1] if n_gen > 1 else prompt_tokens
            full_target = prompt_tokens[1:] + gen_tokens
            full_logprobs = [0.0] * n_prompt + logprobs
            full_advantages = [0.0] * n_prompt + [adv] * n_gen
            
            if len(full_target) != len(full_input) or len(full_logprobs) != len(full_input):
                continue
            
            training_data.append(types.Datum(
                model_input=types.ModelInput.from_ints(full_input),
                loss_fn_inputs=dict(
                    target_tokens=full_target,
                    logprobs=full_logprobs,
                    advantages=full_advantages
                )
            ))
        
        # Update model
        if training_data:
            fwd_future = await training_client.forward_backward_async(
                training_data, loss_fn="importance_sampling"
            )
            optim_future = await training_client.optim_step_async(
                types.AdamParams(learning_rate=RL_LR, beta1=0.9, beta2=0.95, eps=1e-8)
            )
            await fwd_future.result_async()
            await optim_future.result_async()
        
        iter_time = time.time() - iter_start
        accuracy = sum(1 for r in all_rewards if r > 0) / len(all_rewards) if all_rewards else 0
        
        metrics = {
            "iteration": iteration,
            "mean_reward": float(mean_reward),
            "accuracy": accuracy,
            "num_rollouts": len(all_rollouts),
            "time": iter_time
        }
        metrics_log.append(metrics)
        
        print(f"  Reward: {mean_reward:.3f}, Accuracy: {accuracy:.1%}, Time: {iter_time:.1f}s")
    
    # Save final checkpoint
    print("\nSaving final RL checkpoint...")
    final_future = await training_client.save_weights_for_sampler_async(name="rl_final")
    final_result = await final_future.result_async()
    rl_checkpoint = final_result.path
    
    print(f"RL checkpoint: {rl_checkpoint}")
    
    return rl_checkpoint, metrics_log


async def main():
    print("=" * 70)
    print("MEMORY ROUTING AGENT - EVALUATION & RL CONTINUATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now()}")
    print(f"Base Model: {BASE_MODEL}")
    print(f"SFT State Checkpoint: {SFT_STATE_CHECKPOINT}")
    
    # Initialize
    service_client = tinker.ServiceClient()
    tokenizer = get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(name="llama3", tokenizer=tokenizer)
    
    # Load data
    with open(TRAIN_DATA_PATH, "r") as f:
        train_data = json.load(f)
    with open(TEST_DATA_PATH, "r") as f:
        test_data = json.load(f)
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Evaluate SFT model
    sft_metrics = await evaluate_model(
        service_client, SFT_SAMPLER_CHECKPOINT, tokenizer, renderer, test_data, "SFT Model", n_samples=100
    )
    
    # Create training client for RL
    training_client = await service_client.create_lora_training_client_async(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
    )
    
    # Run RL phase
    rl_checkpoint, rl_metrics = await run_rl_phase(
        service_client, training_client, tokenizer, renderer, train_data
    )
    
    # Evaluate RL model
    rl_eval_metrics = await evaluate_model(
        service_client, rl_checkpoint, tokenizer, renderer, test_data, "RL Model", n_samples=100
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"\nSFT Model:")
    print(f"  Checkpoint: {SFT_SAMPLER_CHECKPOINT}")
    print(f"  Any Match: {sft_metrics['any_match']:.1%}")
    print(f"  F1 Score:  {sft_metrics['f1']:.1%}")
    
    print(f"\nRL Model:")
    print(f"  Checkpoint: {rl_checkpoint}")
    print(f"  Any Match: {rl_eval_metrics['any_match']:.1%}")
    print(f"  F1 Score:  {rl_eval_metrics['f1']:.1%}")
    
    improvement = rl_eval_metrics['any_match'] - sft_metrics['any_match']
    print(f"\nImprovement: {improvement:+.1%}")
    
    # Save results
    results = {
        "sft_checkpoint": SFT_SAMPLER_CHECKPOINT,
        "rl_checkpoint": rl_checkpoint,
        "sft_metrics": sft_metrics,
        "rl_metrics": rl_eval_metrics,
        "rl_training_log": rl_metrics
    }
    
    results_path = f"training/experiments/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())

