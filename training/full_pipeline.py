"""
Full Training Pipeline: SFT -> RL -> Evaluation

This script implements the complete training pipeline for the Memory Routing Agent
following best practices from Tinker documentation and ML research.

Key insights from the codebase analysis:
1. SFT must save with save_state() for RL to continue from those weights
2. RL uses importance_sampling loss with proper advantage normalization
3. Evaluation should compare against baseline (untrained) and larger models

Architecture decisions:
- Base model: Llama-3.1-8B (good balance of capability and efficiency)
- LoRA rank 32 (sufficient for classification, per Tinker docs)
- SFT: 100 steps with early stopping, then RL: 15 iterations
"""

import asyncio
import json
import time
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter
from datetime import datetime


@dataclass
class PipelineConfig:
    """Configuration for the full training pipeline."""
    
    # Model
    base_model: str = "meta-llama/Llama-3.1-8B"
    lora_rank: int = 32
    renderer_name: str = "llama3"
    
    # SFT Phase
    sft_steps: int = 100
    sft_batch_size: int = 64
    sft_lr: Optional[float] = None  # Auto from get_lr()
    sft_eval_every: int = 10
    sft_early_stopping_patience: int = 5
    
    # RL Phase
    rl_iterations: int = 15
    rl_batch_size: int = 32
    rl_group_size: int = 8
    rl_lr: float = 1e-5
    rl_temperature: float = 0.7
    rl_kl_threshold: float = 0.01
    
    # Data
    train_data_path: str = "training/processed_data/train_data.json"
    test_data_path: str = "training/processed_data/test_data.json"
    
    # Output
    experiment_name: str = "memory_routing_v1"
    output_dir: str = "training/experiments"


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


def compute_reward(predicted_text: str, gold_categories: List[str]) -> Tuple[float, Dict]:
    """
    Compute reward with detailed breakdown.
    
    R_total = 0.6 * R_F1 + 0.2 * R_temp + 0.1 * R_parity + 0.1 * R_eff
    """
    info = {"format_valid": True, "r_f1": 0, "r_temp": 0, "r_parity": 0, "r_eff": 0}
    
    # Parse prediction
    if not predicted_text or not predicted_text.strip():
        info["format_valid"] = False
        return -1.0, info
    
    predicted = set([c.strip().lower() for c in predicted_text.split(",") 
                     if c.strip().lower() in VALID_CATEGORIES])
    
    if not predicted:
        info["format_valid"] = False
        return -1.0, info
    
    # Remove "none" if mixed with others
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
    
    pred_pers = majority_persistence(predicted)
    gold_pers = majority_persistence(gold)
    
    if pred_pers == gold_pers:
        info["r_temp"] = 1.0
    elif (pred_pers, gold_pers) in [("long", "medium"), ("medium", "long"), 
                                     ("medium", "short"), ("short", "medium")]:
        info["r_temp"] = 0.5
    
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
    info["r_eff"] = 1.0 if n <= 3 else (0.7 if n == 4 else (0.4 if n == 5 else 0.0))
    
    # Total
    r_total = 0.6 * info["r_f1"] + 0.2 * info["r_temp"] + 0.1 * info["r_parity"] + 0.1 * info["r_eff"]
    
    return r_total, info


async def run_sft_phase(config: PipelineConfig, service_client, tokenizer, renderer):
    """
    Phase 1: Supervised Fine-Tuning
    
    Key principles:
    - Use cross_entropy loss for next-token prediction
    - Monitor train/test loss for overfitting
    - Save full state checkpoint for RL continuation
    """
    import tinker
    from tinker import types
    from tinker_cookbook.hyperparam_utils import get_lr
    
    print("\n" + "=" * 70)
    print("PHASE 1: SUPERVISED FINE-TUNING")
    print("=" * 70)
    
    # Load data
    with open(config.train_data_path, "r") as f:
        train_data_raw = json.load(f)
    with open(config.test_data_path, "r") as f:
        test_data_raw = json.load(f)
    
    print(f"Train: {len(train_data_raw)}, Test: {len(test_data_raw)}")
    
    # Get learning rate
    lr = config.sft_lr or get_lr(config.base_model)
    print(f"Learning rate: {lr:.2e}")
    
    # Create training client
    training_client = await service_client.create_lora_training_client_async(
        base_model=config.base_model,
        rank=config.lora_rank,
    )
    
    # Convert data to Datum objects
    def to_datum(item):
        messages = item.get("messages", [])
        tokens, weights = renderer.build_supervised_example(messages)
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        if hasattr(weights, 'tolist'):
            weights = weights.tolist()
        
        return types.Datum(
            model_input=types.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs=dict(target_tokens=tokens[1:], weights=weights[1:])
        )
    
    train_data = [to_datum(item) for item in train_data_raw]
    test_data = [to_datum(item) for item in test_data_raw[:50]]  # Subset for eval
    
    # Training loop
    metrics_log = []
    best_test_loss = float('inf')
    no_improvement = 0
    
    for step in range(config.sft_steps):
        step_start = time.time()
        
        # Create batch
        batch_idx = (step * config.sft_batch_size) % len(train_data)
        batch = train_data[batch_idx:batch_idx + config.sft_batch_size]
        if len(batch) < config.sft_batch_size:
            batch = batch + train_data[:config.sft_batch_size - len(batch)]
        
        # Forward-backward
        fwd_future = await training_client.forward_backward_async(batch, loss_fn="cross_entropy")
        optim_future = await training_client.optim_step_async(
            types.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)
        )
        
        fwd_result = await fwd_future.result_async()
        await optim_future.result_async()
        
        # Compute train loss
        logprobs = np.concatenate([o['logprobs'].tolist() for o in fwd_result.loss_fn_outputs])
        weights = np.concatenate([d.loss_fn_inputs['weights'].tolist() for d in batch])
        train_loss = -np.dot(logprobs, weights) / max(weights.sum(), 1)
        
        step_time = time.time() - step_start
        
        # Evaluation
        test_loss = None
        if step % config.sft_eval_every == 0 or step == config.sft_steps - 1:
            eval_future = await training_client.forward_backward_async(test_data, loss_fn="cross_entropy")
            eval_result = await eval_future.result_async()
            test_logprobs = np.concatenate([o['logprobs'].tolist() for o in eval_result.loss_fn_outputs])
            test_weights = np.concatenate([d.loss_fn_inputs['weights'].tolist() for d in test_data])
            test_loss = -np.dot(test_logprobs, test_weights) / max(test_weights.sum(), 1)
            
            # Early stopping check
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                no_improvement = 0
            else:
                no_improvement += 1
            
            print(f"Step {step:3d}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}, time={step_time:.1f}s")
            
            if no_improvement >= config.sft_early_stopping_patience:
                print(f"Early stopping at step {step}")
                break
        else:
            print(f"Step {step:3d}: train_loss={train_loss:.4f}, time={step_time:.1f}s")
        
        metrics_log.append({
            "step": step, "train_loss": float(train_loss), 
            "test_loss": float(test_loss) if test_loss else None,
            "time": step_time
        })
    
    # Save final checkpoint (full state for RL)
    print("\nSaving final SFT checkpoint...")
    state_future = await training_client.save_state_async(name="sft_final")
    state_result = await state_future.result_async()
    sft_checkpoint = state_result.path
    
    # Also save sampler weights for inference
    sampler_future = await training_client.save_weights_for_sampler_async(name="sft_final_sampler")
    sampler_result = await sampler_future.result_async()
    sampler_checkpoint = sampler_result.path
    
    print(f"SFT State checkpoint: {sft_checkpoint}")
    print(f"SFT Sampler checkpoint: {sampler_checkpoint}")
    
    return training_client, sft_checkpoint, sampler_checkpoint, metrics_log


async def run_rl_phase(config: PipelineConfig, service_client, training_client, 
                       sft_checkpoint: str, tokenizer, renderer):
    """
    Phase 2: Reinforcement Learning
    
    Key principles:
    - Load SFT weights to continue training
    - Use importance_sampling loss for policy gradient
    - Group rollouts for variance reduction
    - Monitor KL divergence for stability
    """
    import tinker
    from tinker import types
    
    print("\n" + "=" * 70)
    print("PHASE 2: REINFORCEMENT LEARNING")
    print("=" * 70)
    
    # Load training data
    with open(config.train_data_path, "r") as f:
        train_data = json.load(f)
    
    print(f"Training examples: {len(train_data)}")
    print(f"RL iterations: {config.rl_iterations}")
    print(f"Batch size: {config.rl_batch_size}, Group size: {config.rl_group_size}")
    
    # Load SFT weights into training client
    print(f"\nLoading SFT checkpoint: {sft_checkpoint}")
    await training_client.load_state_async(sft_checkpoint)
    
    stop_sequences = renderer.get_stop_sequences()
    metrics_log = []
    
    for iteration in range(config.rl_iterations):
        iter_start = time.time()
        print(f"\n--- Iteration {iteration + 1}/{config.rl_iterations} ---")
        
        # Save current weights for sampling
        save_future = await training_client.save_weights_for_sampler_async(
            name=f"rl_iter_{iteration:03d}"
        )
        save_result = await save_future.result_async()
        sampling_client = service_client.create_sampling_client(model_path=save_result.path)
        
        # Sample batch
        batch_indices = np.random.choice(len(train_data), size=config.rl_batch_size, replace=False)
        
        all_rollouts = []
        all_rewards = []
        reward_infos = []
        
        for idx in batch_indices:
            example = train_data[idx]
            gold_categories = example.get("categories", [])
            messages = example.get("messages", [])
            prompt_messages = [m for m in messages if m.get("role") != "assistant"]
            
            if not prompt_messages:
                continue
            
            prompt = renderer.build_generation_prompt(prompt_messages)
            params = types.SamplingParams(
                max_tokens=50, temperature=config.rl_temperature, stop=stop_sequences
            )
            
            result = sampling_client.sample(
                prompt=prompt, sampling_params=params, num_samples=config.rl_group_size
            ).result()
            
            for seq in result.sequences:
                response, success = renderer.parse_response(seq.tokens)
                predicted = response["content"] if success else ""
                reward, info = compute_reward(predicted, gold_categories)
                
                all_rollouts.append({
                    "prompt": prompt,
                    "tokens": seq.tokens,
                    "logprobs": seq.logprobs or [],
                    "predicted": predicted,
                    "gold": gold_categories
                })
                all_rewards.append(reward)
                reward_infos.append(info)
        
        # Compute advantages (normalized)
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
                types.AdamParams(learning_rate=config.rl_lr, beta1=0.9, beta2=0.95, eps=1e-8)
            )
            await fwd_future.result_async()
            await optim_future.result_async()
        
        # Metrics
        iter_time = time.time() - iter_start
        accuracy = sum(1 for r in all_rewards if r > 0) / len(all_rewards) if all_rewards else 0
        format_valid_rate = sum(1 for info in reward_infos if info["format_valid"]) / len(reward_infos)
        
        metrics = {
            "iteration": iteration,
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "accuracy": accuracy,
            "format_valid_rate": format_valid_rate,
            "num_rollouts": len(all_rollouts),
            "time": iter_time
        }
        metrics_log.append(metrics)
        
        print(f"  Reward: {mean_reward:.3f} ± {std_reward:.3f}, Acc: {accuracy:.1%}, Format: {format_valid_rate:.1%}")
    
    # Save final checkpoint
    print("\nSaving final RL checkpoint...")
    final_future = await training_client.save_weights_for_sampler_async(name="rl_final")
    final_result = await final_future.result_async()
    rl_checkpoint = final_result.path
    
    print(f"RL checkpoint: {rl_checkpoint}")
    
    return rl_checkpoint, metrics_log


async def run_evaluation(config: PipelineConfig, service_client, checkpoint: str, 
                         tokenizer, renderer, name: str = "model"):
    """
    Comprehensive evaluation on test set.
    """
    import tinker
    from tinker import types
    
    print(f"\n--- Evaluating: {name} ---")
    
    # Load test data
    with open(config.test_data_path, "r") as f:
        test_data = json.load(f)
    
    sampling_client = service_client.create_sampling_client(model_path=checkpoint)
    stop_sequences = renderer.get_stop_sequences()
    
    results = []
    
    for i, example in enumerate(test_data):
        gold = example.get("categories", [])
        messages = example.get("messages", [])
        prompt_messages = [m for m in messages if m.get("role") != "assistant"]
        
        if not prompt_messages:
            continue
        
        prompt = renderer.build_generation_prompt(prompt_messages)
        params = types.SamplingParams(max_tokens=50, temperature=0.1, stop=stop_sequences)
        
        result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
        response, success = renderer.parse_response(result.sequences[0].tokens)
        predicted = response["content"] if success else ""
        
        reward, info = compute_reward(predicted, gold)
        
        predicted_set = set([c.strip().lower() for c in predicted.split(",") if c.strip().lower() in VALID_CATEGORIES])
        gold_set = set([c.lower() for c in gold])
        
        results.append({
            "gold": gold,
            "predicted": predicted,
            "reward": reward,
            "exact_match": predicted_set == gold_set,
            "any_match": len(predicted_set & gold_set) > 0,
            "precision": len(predicted_set & gold_set) / len(predicted_set) if predicted_set else 0,
            "recall": len(predicted_set & gold_set) / len(gold_set) if gold_set else 0,
            "format_valid": info["format_valid"]
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Evaluated {i + 1}/{len(test_data)}")
    
    # Aggregate metrics
    n = len(results)
    metrics = {
        "name": name,
        "n_examples": n,
        "mean_reward": np.mean([r["reward"] for r in results]),
        "exact_match": np.mean([r["exact_match"] for r in results]),
        "any_match": np.mean([r["any_match"] for r in results]),
        "precision": np.mean([r["precision"] for r in results]),
        "recall": np.mean([r["recall"] for r in results]),
        "format_valid": np.mean([r["format_valid"] for r in results]),
    }
    metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"]) if (metrics["precision"] + metrics["recall"]) > 0 else 0
    
    print(f"  Any Match: {metrics['any_match']:.1%}")
    print(f"  Exact Match: {metrics['exact_match']:.1%}")
    print(f"  F1: {metrics['f1']:.1%}")
    print(f"  Mean Reward: {metrics['mean_reward']:.3f}")
    
    return metrics, results


async def main():
    """Run the full training pipeline."""
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from dotenv import load_dotenv
    
    load_dotenv()
    
    config = PipelineConfig()
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(config.output_dir, f"{config.experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    print("=" * 70)
    print("MEMORY ROUTING AGENT - FULL TRAINING PIPELINE")
    print("=" * 70)
    print(f"Experiment: {config.experiment_name}")
    print(f"Output: {exp_dir}")
    print(f"Base model: {config.base_model}")
    print(f"LoRA rank: {config.lora_rank}")
    
    # Initialize
    service_client = tinker.ServiceClient()
    tokenizer = get_tokenizer(config.base_model)
    renderer = renderers.get_renderer(name=config.renderer_name, tokenizer=tokenizer)
    
    # Phase 1: SFT
    training_client, sft_state_ckpt, sft_sampler_ckpt, sft_metrics = await run_sft_phase(
        config, service_client, tokenizer, renderer
    )
    
    # Evaluate SFT model
    sft_eval, _ = await run_evaluation(
        config, service_client, sft_sampler_ckpt, tokenizer, renderer, "SFT Model"
    )
    
    # Phase 2: RL
    rl_checkpoint, rl_metrics = await run_rl_phase(
        config, service_client, training_client, sft_state_ckpt, tokenizer, renderer
    )
    
    # Evaluate RL model
    rl_eval, _ = await run_evaluation(
        config, service_client, rl_checkpoint, tokenizer, renderer, "RL Model"
    )
    
    # Save results
    results = {
        "config": {
            "base_model": config.base_model,
            "lora_rank": config.lora_rank,
            "sft_steps": config.sft_steps,
            "rl_iterations": config.rl_iterations,
        },
        "checkpoints": {
            "sft_state": sft_state_ckpt,
            "sft_sampler": sft_sampler_ckpt,
            "rl_final": rl_checkpoint,
        },
        "sft_metrics": sft_metrics,
        "rl_metrics": rl_metrics,
        "evaluation": {
            "sft": sft_eval,
            "rl": rl_eval,
        }
    }
    
    results_path = os.path.join(exp_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {results_path}")
    print(f"\nFinal Model: {rl_checkpoint}")
    print(f"\nComparison:")
    print(f"  SFT  - F1: {sft_eval['f1']:.1%}, Any Match: {sft_eval['any_match']:.1%}")
    print(f"  RL   - F1: {rl_eval['f1']:.1%}, Any Match: {rl_eval['any_match']:.1%}")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())

