"""
Memory Routing Agent - Training with Qwen3-32B

Using a larger model (32B vs 8B) to see if increased capacity improves:
1. Multi-label detection
2. Category disambiguation (e.g., company.strategic_signatures vs user.strategic_approach)
3. Edge case handling

Qwen3-32B is a "Hybrid" model that can do thinking/non-thinking modes.
Per Tinker docs, it's a dense 32B parameter model.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Set, Optional
from dotenv import load_dotenv
import numpy as np

load_dotenv()

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.hyperparam_utils import get_lr

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    # Model - Qwen3-32B (4x larger than Llama-8B)
    base_model: str = "Qwen/Qwen3-32B"
    lora_rank: int = 32
    
    # SFT Config
    sft_steps: int = 100
    sft_batch_size: int = 16  # Smaller batch for larger model
    sft_eval_every: int = 10
    sft_early_stopping_patience: int = 5
    sft_min_steps: int = 30
    
    # RL Config
    rl_iterations: int = 20
    rl_groups_per_batch: int = 32  # Reduced for larger model
    rl_group_size: int = 16  # Reduced for larger model
    rl_learning_rate: float = 1e-5  # Lower LR for larger model
    rl_temperature: float = 0.7
    rl_max_tokens: int = 100
    rl_kl_threshold: float = 0.01
    
    # Reward weights
    reward_f1_weight: float = 0.6
    reward_temp_weight: float = 0.2
    reward_parity_weight: float = 0.1
    reward_efficiency_weight: float = 0.1
    
    # Paths
    train_data: str = "training/processed_data/train_data.json"
    test_data: str = "training/processed_data/test_data.json"
    log_dir: str = field(default_factory=lambda: f"training/logs/qwen32b_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


# =============================================================================
# MEMORY TAXONOMY
# =============================================================================

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

CATEGORY_SCOPE = {cat: cat.split(".")[0] if "." in cat else "none" for cat in VALID_CATEGORIES}


# =============================================================================
# LOGGING
# =============================================================================

class TrainingLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.sft_file = open(os.path.join(log_dir, "sft_metrics.jsonl"), "w")
        self.rl_file = open(os.path.join(log_dir, "rl_metrics.jsonl"), "w")
        self.start_time = time.time()
    
    def log_sft(self, step: int, metrics: Dict[str, Any]):
        metrics["step"] = step
        metrics["elapsed_time"] = time.time() - self.start_time
        self.sft_file.write(json.dumps(metrics) + "\n")
        self.sft_file.flush()
        
        test_loss_str = f"{metrics.get('test_loss', 0):.4f}" if isinstance(metrics.get('test_loss'), float) else "N/A"
        print(f"[SFT {step:3d}] "
              f"Loss: {metrics.get('train_loss', 0):.4f} | "
              f"Test: {test_loss_str} | "
              f"Time: {metrics.get('step_time', 0):.1f}s", flush=True)
    
    def log_rl(self, iteration: int, metrics: Dict[str, Any]):
        metrics["iteration"] = iteration
        metrics["elapsed_time"] = time.time() - self.start_time
        self.rl_file.write(json.dumps(metrics) + "\n")
        self.rl_file.flush()
        
        print(f"[RL {iteration:3d}] "
              f"Reward: {metrics.get('mean_reward', 0):.3f} (±{metrics.get('std_reward', 0):.3f}) | "
              f"Acc: {metrics.get('accuracy', 0):.1%} | "
              f"KL_v1: {metrics.get('kl_v1', 0):.4f} | "
              f"KL_v2: {metrics.get('kl_v2', 0):.4f} | "
              f"Active: {metrics.get('active_groups', 0)}/{metrics.get('total_groups', 0)} | "
              f"Time: {metrics.get('iter_time', 0):.1f}s", flush=True)
        
        if metrics.get('kl_v2', 0) > 0.01:
            print(f"WARNING: KL_v2 {metrics.get('kl_v2', 0):.4f} exceeds threshold 0.01", flush=True)
    
    def close(self):
        self.sft_file.close()
        self.rl_file.close()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(config: TrainingConfig) -> Tuple[List[Dict], List[Dict]]:
    """Load preprocessed training and test data."""
    with open(config.train_data, "r") as f:
        train_data = json.load(f)
    with open(config.test_data, "r") as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(train_data)} training examples, {len(test_data)} test examples", flush=True)
    return train_data, test_data


# =============================================================================
# REWARD COMPUTATION
# =============================================================================

def parse_prediction(text: str) -> Set[str]:
    """Parse model output into category set."""
    if not text or not text.strip():
        return set()
    cats = [c.strip().lower() for c in text.split(",")]
    return {c for c in cats if c in VALID_CATEGORIES}


def compute_reward(predicted: Set[str], gold: Set[str], config: TrainingConfig) -> Tuple[float, Dict[str, float]]:
    """
    Compute composite reward per PRD Section 4:
    R_total = 0.6*R_F1 + 0.2*R_temp + 0.1*R_parity + 0.1*R_eff
    """
    # R_F1: F1 score
    if not predicted and not gold:
        r_f1 = 1.0
    elif not predicted or not gold:
        r_f1 = 0.0
    else:
        tp = len(predicted & gold)
        precision = tp / len(predicted) if predicted else 0
        recall = tp / len(gold) if gold else 0
        r_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # R_temp: Temporal alignment
    if not predicted:
        r_temp = 0.0
    else:
        persistence_matches = sum(
            1 for cat in predicted 
            if cat in gold and CATEGORY_PERSISTENCE.get(cat, "") == CATEGORY_PERSISTENCE.get(cat, "")
        )
        r_temp = persistence_matches / len(predicted) if predicted else 0
    
    # R_parity: User/company scope balance
    pred_user = sum(1 for c in predicted if CATEGORY_SCOPE.get(c, "") == "user")
    pred_company = sum(1 for c in predicted if CATEGORY_SCOPE.get(c, "") == "company")
    gold_user = sum(1 for c in gold if CATEGORY_SCOPE.get(c, "") == "user")
    gold_company = sum(1 for c in gold if CATEGORY_SCOPE.get(c, "") == "company")
    
    pred_ratio = pred_user / (pred_user + pred_company) if (pred_user + pred_company) > 0 else 0.5
    gold_ratio = gold_user / (gold_user + gold_company) if (gold_user + gold_company) > 0 else 0.5
    r_parity = 1.0 - abs(pred_ratio - gold_ratio)
    
    # R_eff: Efficiency (penalize over-prediction)
    if not gold:
        r_eff = 1.0 if not predicted else 0.5
    else:
        r_eff = min(1.0, len(gold) / len(predicted)) if predicted else 0.0
    
    # Composite reward
    r_total = (
        config.reward_f1_weight * r_f1 +
        config.reward_temp_weight * r_temp +
        config.reward_parity_weight * r_parity +
        config.reward_efficiency_weight * r_eff
    )
    
    return r_total, {"r_f1": r_f1, "r_temp": r_temp, "r_parity": r_parity, "r_eff": r_eff}


# =============================================================================
# SFT TRAINING
# =============================================================================

async def run_sft(config: TrainingConfig, train_data: List[Dict], test_data: List[Dict], logger: TrainingLogger):
    """Run Supervised Fine-Tuning phase."""
    print("\n" + "="*70, flush=True)
    print("PHASE 1: SUPERVISED FINE-TUNING (Qwen3-32B)", flush=True)
    print("="*70, flush=True)
    
    # Get recommended LR for this model
    recommended_lr = get_lr(config.base_model)
    print(f"Model: {config.base_model}", flush=True)
    print(f"Recommended LR: {recommended_lr:.6e}", flush=True)
    print(f"LoRA Rank: {config.lora_rank}", flush=True)
    print(f"Steps: {config.sft_steps}", flush=True)
    print(f"Batch Size: {config.sft_batch_size}", flush=True)
    
    # Initialize clients
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=config.base_model,
        rank=config.lora_rank
    )
    
    # Get tokenizer and renderer for Qwen3
    tokenizer = get_tokenizer(config.base_model)
    renderer = renderers.get_renderer(name="qwen3", tokenizer=tokenizer)
    
    # Convert data to Datum objects
    def convert_to_datum(item: Dict) -> types.Datum:
        messages = item["messages"]
        tokens, weights = renderer.build_supervised_example(messages)
        
        # Convert tensors to lists (Tinker requires Python lists, not torch.Tensor)
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        if hasattr(weights, 'tolist'):
            weights = weights.tolist()
        
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        weights = weights[1:]
        
        return types.Datum(
            model_input=types.ModelInput.from_ints(input_tokens),
            loss_fn_inputs={
                "target_tokens": target_tokens,
                "weights": weights
            }
        )
    
    print("Converting data to Datum objects...", flush=True)
    train_datums = [convert_to_datum(item) for item in train_data]
    test_datums = [convert_to_datum(item) for item in test_data[:50]]
    
    # Shuffle training data indices
    train_indices = list(range(len(train_datums)))
    np.random.shuffle(train_indices)
    
    best_test_loss = float('inf')
    patience_counter = 0
    best_checkpoint = None
    
    # Training loop
    for step in range(config.sft_steps):
        step_start = time.time()
        
        # Get batch
        epoch = step * config.sft_batch_size // len(train_datums)
        if step * config.sft_batch_size % len(train_datums) < config.sft_batch_size:
            np.random.shuffle(train_indices)
        
        batch_start = (step * config.sft_batch_size) % len(train_datums)
        batch_indices = train_indices[batch_start:batch_start + config.sft_batch_size]
        if len(batch_indices) < config.sft_batch_size:
            batch_indices = batch_indices + train_indices[:config.sft_batch_size - len(batch_indices)]
        
        batch = [train_datums[i] for i in batch_indices]
        
        # Forward-backward
        fwd_bwd_future = await training_client.forward_backward_async(batch, loss_fn="cross_entropy")
        
        # Optimizer step
        adam_params = types.AdamParams(learning_rate=recommended_lr)
        optim_future = await training_client.optim_step_async(adam_params)
        
        # Wait for results
        fwd_bwd_result = await fwd_bwd_future.result_async()
        await optim_future.result_async()
        
        # Compute train loss
        train_loss = sum(
            -np.dot(output["logprobs"].tolist(), batch[i].loss_fn_inputs["weights"].tolist()) /
            max(sum(batch[i].loss_fn_inputs["weights"].tolist()), 1)
            for i, output in enumerate(fwd_bwd_result.loss_fn_outputs)
        ) / len(batch)
        
        step_time = time.time() - step_start
        
        metrics = {
            "train_loss": train_loss,
            "step_time": step_time,
            "epoch": epoch,
            "learning_rate": recommended_lr
        }
        
        # Periodic evaluation
        if step % config.sft_eval_every == 0:
            test_fwd_future = await training_client.forward_backward_async(test_datums, loss_fn="cross_entropy")
            test_result = await test_fwd_future.result_async()
            
            test_loss = sum(
                -np.dot(output["logprobs"].tolist(), test_datums[i].loss_fn_inputs["weights"].tolist()) /
                max(sum(test_datums[i].loss_fn_inputs["weights"].tolist()), 1)
                for i, output in enumerate(test_result.loss_fn_outputs)
            ) / len(test_datums)
            
            metrics["test_loss"] = test_loss
            
            # Save checkpoint
            checkpoint_future = await training_client.save_weights_for_sampler_async(name=f"sft_step_{step:04d}")
            checkpoint_result = await checkpoint_future.result_async()
            metrics["checkpoint"] = checkpoint_result.path
            
            # Early stopping check
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_checkpoint = checkpoint_result.path
                patience_counter = 0
                metrics["is_best"] = True
            else:
                patience_counter += 1
                metrics["is_best"] = False
            
            metrics["patience_counter"] = patience_counter
            
            if step >= config.sft_min_steps and patience_counter >= config.sft_early_stopping_patience:
                print(f"Early stopping at step {step} (no improvement for {patience_counter} evals)", flush=True)
                logger.log_sft(step, metrics)
                break
        
        logger.log_sft(step, metrics)
    
    # Save final state for RL
    print("\nSaving final SFT state for RL...", flush=True)
    final_state_future = await training_client.save_state_async(name="sft_final")
    final_state = await final_state_future.result_async()
    print(f"SFT state saved: {final_state.path}", flush=True)
    
    # Also save sampler weights
    final_sampler_future = await training_client.save_weights_for_sampler_async(name="sft_final_sampler")
    final_sampler = await final_sampler_future.result_async()
    print(f"SFT sampler weights: {final_sampler.path}", flush=True)
    
    # Quick evaluation
    print("\nEvaluating SFT model...", flush=True)
    sampling_client = service_client.create_sampling_client(model_path=final_sampler.path)
    stop_sequences = renderer.get_stop_sequences()
    params = types.SamplingParams(max_tokens=100, temperature=0.1, stop=stop_sequences)
    
    correct = 0
    exact = 0
    for item in test_data[:50]:
        messages = item["messages"][:-1]
        gold = set(item["categories"])
        
        prompt = renderer.build_generation_prompt(messages)
        result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
        response, _ = renderer.parse_response(result.sequences[0].tokens)
        predicted = parse_prediction(response["content"])
        
        if predicted & gold:
            correct += 1
        if predicted == gold:
            exact += 1
    
    any_match = correct / 50
    exact_match = exact / 50
    f1_approx = (any_match + exact_match) / 2
    
    print(f"SFT: Any={any_match:.1%}, Exact={exact_match:.1%}, F1≈{f1_approx:.1%}", flush=True)
    
    return training_client, final_state.path, final_sampler.path


# =============================================================================
# RL TRAINING
# =============================================================================

async def run_rl(
    config: TrainingConfig,
    training_client: tinker.TrainingClient,
    sft_state_path: str,
    sft_sampler_path: str,
    train_data: List[Dict],
    test_data: List[Dict],
    logger: TrainingLogger
):
    """Run Reinforcement Learning phase."""
    print("\n" + "="*70, flush=True)
    print("PHASE 2: REINFORCEMENT LEARNING (Qwen3-32B)", flush=True)
    print("="*70, flush=True)
    
    print(f"Loading SFT state: {sft_state_path}", flush=True)
    print(f"Iterations: {config.rl_iterations}", flush=True)
    print(f"Groups per batch: {config.rl_groups_per_batch}", flush=True)
    print(f"Group size: {config.rl_group_size}", flush=True)
    print(f"Total rollouts per iteration: {config.rl_groups_per_batch * config.rl_group_size}", flush=True)
    print(f"Learning rate: {config.rl_learning_rate:.2e}", flush=True)
    print(f"KL threshold: {config.rl_kl_threshold}", flush=True)
    
    # Load SFT state
    training_client.load_state(sft_state_path)
    
    service_client = tinker.ServiceClient()
    tokenizer = get_tokenizer(config.base_model)
    renderer = renderers.get_renderer(name="qwen3", tokenizer=tokenizer)
    stop_sequences = renderer.get_stop_sequences()
    
    best_reward = -float('inf')
    best_checkpoint = None
    
    for iteration in range(config.rl_iterations):
        iter_start = time.time()
        
        # Save current weights for sampling
        sampler_future = await training_client.save_weights_for_sampler_async(name=f"rl_iter_{iteration:03d}")
        sampler_result = await sampler_future.result_async()
        sampling_client = service_client.create_sampling_client(model_path=sampler_result.path)
        
        # Sample batch of problems
        batch_indices = np.random.choice(len(train_data), config.rl_groups_per_batch, replace=False)
        
        all_datums = []
        all_rewards = []
        all_old_logprobs = []
        all_new_logprobs = []
        reward_components = {"r_f1": [], "r_temp": [], "r_parity": [], "r_eff": []}
        correct_count = 0
        total_count = 0
        
        # Generate rollouts for each problem
        for group_idx, data_idx in enumerate(batch_indices):
            item = train_data[data_idx]
            messages = item["messages"][:-1]
            gold = set(item["categories"])
            
            prompt = renderer.build_generation_prompt(messages)
            params = types.SamplingParams(
                max_tokens=config.rl_max_tokens,
                temperature=config.rl_temperature,
                stop=stop_sequences,
                logprobs=True
            )
            
            # Generate group_size samples
            result = sampling_client.sample(
                prompt=prompt,
                sampling_params=params,
                num_samples=config.rl_group_size
            ).result()
            
            group_rewards = []
            group_datums = []
            group_old_lps = []
            
            for seq in result.sequences:
                response, _ = renderer.parse_response(seq.tokens)
                predicted = parse_prediction(response["content"])
                
                reward, components = compute_reward(predicted, gold, config)
                group_rewards.append(reward)
                
                for key in reward_components:
                    reward_components[key].append(components[key])
                
                if predicted & gold:
                    correct_count += 1
                total_count += 1
                
                # Build datum for training
                input_tokens = prompt.to_ints() + seq.tokens
                target_tokens = seq.tokens
                old_logprobs = seq.logprobs if seq.logprobs else [0.0] * len(seq.tokens)
                
                group_datums.append({
                    "input_tokens": input_tokens,
                    "target_tokens": target_tokens,
                    "old_logprobs": old_logprobs,
                    "reward": reward
                })
                group_old_lps.extend(old_logprobs)
            
            # Compute advantages (center within group)
            group_mean = np.mean(group_rewards)
            group_std = np.std(group_rewards) + 1e-8
            
            # Only include groups with reward variance
            if np.std(group_rewards) > 0.01:
                for datum_dict in group_datums:
                    advantage = (datum_dict["reward"] - group_mean) / group_std
                    advantages = [advantage] * len(datum_dict["target_tokens"])
                    
                    datum = types.Datum(
                        model_input=types.ModelInput.from_ints(datum_dict["input_tokens"]),
                        loss_fn_inputs={
                            "target_tokens": datum_dict["target_tokens"],
                            "logprobs": datum_dict["old_logprobs"],
                            "advantages": advantages
                        }
                    )
                    all_datums.append(datum)
                    all_rewards.append(datum_dict["reward"])
                    all_old_logprobs.extend(datum_dict["old_logprobs"])
        
        # Skip if no training data
        if not all_datums:
            print(f"[RL {iteration:3d}] No active groups, skipping...", flush=True)
            continue
        
        # Forward-backward with importance sampling
        fwd_bwd_future = await training_client.forward_backward_async(all_datums, loss_fn="importance_sampling")
        
        # Optimizer step
        adam_params = types.AdamParams(learning_rate=config.rl_learning_rate)
        optim_future = await training_client.optim_step_async(adam_params)
        
        # Wait for results
        fwd_bwd_result = await fwd_bwd_future.result_async()
        await optim_future.result_async()
        
        # Compute KL divergence
        all_new_logprobs = []
        for output in fwd_bwd_result.loss_fn_outputs:
            all_new_logprobs.extend(output["logprobs"].tolist())
        
        # KL estimators per Tinker docs
        log_ratios = np.array(all_new_logprobs) - np.array(all_old_logprobs[:len(all_new_logprobs)])
        kl_v1 = np.mean(log_ratios)  # Can be negative
        kl_v2 = np.mean((np.exp(log_ratios) - 1) - log_ratios)  # Always non-negative
        
        iter_time = time.time() - iter_start
        
        mean_reward = np.mean(all_rewards)
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_checkpoint = sampler_result.path
        
        metrics = {
            "mean_reward": mean_reward,
            "std_reward": np.std(all_rewards),
            "accuracy": correct_count / total_count if total_count > 0 else 0,
            "kl_v1": kl_v1,
            "kl_v2": kl_v2,
            "total_groups": config.rl_groups_per_batch,
            "active_groups": len(all_datums) // config.rl_group_size,
            "num_training_examples": len(all_datums),
            "iter_time": iter_time,
            "checkpoint": sampler_result.path,
            "mean_r_f1": np.mean(reward_components["r_f1"]),
            "mean_r_temp": np.mean(reward_components["r_temp"]),
            "mean_r_parity": np.mean(reward_components["r_parity"]),
            "mean_r_eff": np.mean(reward_components["r_eff"]),
        }
        
        logger.log_rl(iteration, metrics)
    
    # Save final RL weights
    print("\nSaving final RL weights...", flush=True)
    final_future = await training_client.save_weights_for_sampler_async(name="rl_final")
    final_result = await final_future.result_async()
    print(f"RL final weights: {final_result.path}", flush=True)
    print(f"Best checkpoint: {best_checkpoint} (reward: {best_reward:.3f})", flush=True)
    
    return final_result.path, best_checkpoint


# =============================================================================
# MAIN
# =============================================================================

async def main():
    config = TrainingConfig()
    
    print("="*70, flush=True)
    print("MEMORY ROUTING AGENT - QWEN3-32B TRAINING", flush=True)
    print("="*70, flush=True)
    print(f"Model: {config.base_model}", flush=True)
    print(f"Log directory: {config.log_dir}", flush=True)
    
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(config.log_dir, "config.json"), "w") as f:
        json.dump({k: v for k, v in config.__dict__.items() if not k.startswith("_")}, f, indent=2)
    
    logger = TrainingLogger(config.log_dir)
    
    try:
        # Load data
        train_data, test_data = load_data(config)
        
        # Run SFT
        training_client, sft_state_path, sft_sampler_path = await run_sft(
            config, train_data, test_data, logger
        )
        
        # Run RL
        rl_final_path, rl_best_path = await run_rl(
            config, training_client, sft_state_path, sft_sampler_path,
            train_data, test_data, logger
        )
        
        print("\n" + "="*70, flush=True)
        print("TRAINING COMPLETE", flush=True)
        print("="*70, flush=True)
        print(f"SFT checkpoint: {sft_sampler_path}", flush=True)
        print(f"RL final: {rl_final_path}", flush=True)
        print(f"RL best: {rl_best_path}", flush=True)
        print(f"Logs: {config.log_dir}", flush=True)
        
    finally:
        logger.close()


if __name__ == "__main__":
    asyncio.run(main())

