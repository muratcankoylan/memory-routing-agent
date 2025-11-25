"""
Fixed Training Script for Memory Routing Agent.

Key fixes based on Tinker docs:
1. Proper advantage computation (centered within groups)
2. Correct tensor alignment for importance_sampling loss
3. Proper group-based rollout collection
4. KL divergence monitoring
"""

import asyncio
import json
import os
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.hyperparam_utils import get_lr
import numpy as np

# Configuration
BASE_MODEL = "meta-llama/Llama-3.1-8B"
LORA_RANK = 32

# SFT Config
SFT_STEPS = 50
SFT_BATCH_SIZE = 32

# RL Config - Following Tinker's recommendations
RL_ITERATIONS = 20  # More iterations for better convergence
RL_GROUPS_PER_BATCH = 32  # Number of unique problems per batch
RL_GROUP_SIZE = 4  # Number of rollouts per problem (for advantage computation)
RL_LR = 2e-5  # Slightly higher LR as recommended
RL_TEMPERATURE = 0.7
RL_MAX_TOKENS = 100

# Paths
TRAIN_DATA = "training/processed_data/train_data.json"
TEST_DATA = "training/processed_data/test_data.json"
LOG_DIR = "training/logs/run_" + datetime.now().strftime("%Y%m%d_%H%M%S")

# Categories
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


@dataclass
class Rollout:
    """Single rollout from a problem."""
    prompt_tokens: List[int]
    gen_tokens: List[int]
    logprobs: List[float]
    reward: float
    predicted: str
    gold: List[str]


@dataclass
class RolloutGroup:
    """Group of rollouts for the same problem."""
    problem_id: int
    rollouts: List[Rollout]
    
    def get_rewards(self) -> List[float]:
        return [r.reward for r in self.rollouts]
    
    def is_constant_reward(self) -> bool:
        """Check if all rewards are the same (no learning signal)."""
        rewards = self.get_rewards()
        return len(set(rewards)) == 1


class TrainingLogger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.sft_log = open(os.path.join(log_dir, "sft_metrics.jsonl"), "w")
        self.rl_log = open(os.path.join(log_dir, "rl_metrics.jsonl"), "w")
        self.start_time = time.time()
        
    def log_sft(self, step, metrics):
        metrics["step"] = step
        metrics["elapsed_time"] = time.time() - self.start_time
        self.sft_log.write(json.dumps(metrics) + "\n")
        self.sft_log.flush()
        
        test_loss = metrics.get('test_loss')
        test_str = f"{test_loss:.4f}" if isinstance(test_loss, (int, float)) else "N/A"
        print(f"[SFT {step:3d}] "
              f"Loss: {metrics.get('train_loss', 0):.4f} | "
              f"Test: {test_str} | "
              f"Acc: {metrics.get('accuracy', 'N/A')} | "
              f"Time: {metrics.get('step_time', 0):.1f}s")
    
    def log_rl(self, iteration, metrics):
        metrics["iteration"] = iteration
        metrics["elapsed_time"] = time.time() - self.start_time
        self.rl_log.write(json.dumps(metrics) + "\n")
        self.rl_log.flush()
        
        print(f"[RL {iteration:3d}] "
              f"Reward: {metrics.get('mean_reward', 0):.3f} (±{metrics.get('std_reward', 0):.3f}) | "
              f"Acc: {metrics.get('accuracy', 0):.1%} | "
              f"KL: {metrics.get('kl_divergence', 0):.4f} | "
              f"Groups: {metrics.get('active_groups', 0)} | "
              f"Time: {metrics.get('iter_time', 0):.1f}s")
    
    def close(self):
        self.sft_log.close()
        self.rl_log.close()


def compute_reward(predicted_text: str, gold_categories: List[str]) -> Tuple[float, Dict]:
    """Compute F1-based reward for RL."""
    if not predicted_text or not predicted_text.strip():
        return -1.0, {"format_valid": False, "predicted": set(), "gold": set(gold_categories)}
    
    predicted = set([c.strip().lower() for c in predicted_text.split(",") 
                     if c.strip().lower() in VALID_CATEGORIES])
    
    if not predicted:
        return -1.0, {"format_valid": False, "predicted": set(), "gold": set(gold_categories)}
    
    gold = set([c.lower() for c in gold_categories])
    
    # F1 Score as reward
    if predicted and gold:
        tp = len(predicted & gold)
        precision = tp / len(predicted) if predicted else 0
        recall = tp / len(gold) if gold else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        f1 = 1.0 if not predicted and not gold else 0.0
    
    return f1, {"format_valid": True, "f1": f1, "predicted": predicted, "gold": gold}


def compute_group_advantages(groups: List[RolloutGroup]) -> List[List[float]]:
    """
    Compute advantages by centering rewards within each group.
    This is the correct way per Tinker docs.
    """
    all_advantages = []
    
    for group in groups:
        rewards = np.array(group.get_rewards())
        
        # Center rewards within the group (subtract mean)
        mean_reward = rewards.mean()
        
        # Optionally normalize by std (helps with stability)
        std_reward = rewards.std()
        if std_reward > 1e-8:
            advantages = (rewards - mean_reward) / std_reward
        else:
            advantages = rewards - mean_reward
        
        all_advantages.append(advantages.tolist())
    
    return all_advantages


def build_rl_datum(rollout: Rollout, advantage: float) -> types.Datum:
    """
    Build a Datum for importance_sampling loss.
    
    Per Tinker docs, importance_sampling requires:
    - target_tokens: array[(N,), int] - Target token IDs from sampler
    - logprobs: array[(N,), float] - Reference log probabilities from sampler
    - advantages: array[(N,), float] - Advantage values
    
    All must have shape (N,) where N = model_input.length
    """
    prompt_tokens = rollout.prompt_tokens
    gen_tokens = rollout.gen_tokens
    sampler_logprobs = rollout.logprobs
    
    # The model input is prompt + generated tokens (excluding last token)
    # We predict the next token at each position
    n_prompt = len(prompt_tokens)
    n_gen = len(gen_tokens)
    
    # Full sequence: prompt tokens + generated tokens
    full_tokens = prompt_tokens + gen_tokens
    
    # Model input: all tokens except the last one
    input_tokens = full_tokens[:-1]
    
    # Target tokens: all tokens except the first one (what we're predicting)
    target_tokens = full_tokens[1:]
    
    # For logprobs and advantages:
    # - Prompt tokens: 0.0 (we don't have logprobs for them, and don't want to update on them)
    # - Generated tokens: actual logprobs and advantages
    
    # Number of positions in the input
    n_input = len(input_tokens)
    
    # Logprobs: 0 for prompt positions, actual logprobs for generation positions
    # The logprobs from sampler correspond to gen_tokens
    # After shifting, we need logprobs for positions n_prompt-1 onwards
    full_logprobs = [0.0] * (n_prompt - 1) + sampler_logprobs
    
    # Advantages: 0 for prompt positions, actual advantage for generation positions  
    full_advantages = [0.0] * (n_prompt - 1) + [advantage] * n_gen
    
    # Ensure all arrays have the same length as input_tokens
    assert len(target_tokens) == n_input, f"target_tokens length mismatch: {len(target_tokens)} vs {n_input}"
    assert len(full_logprobs) == n_input, f"logprobs length mismatch: {len(full_logprobs)} vs {n_input}"
    assert len(full_advantages) == n_input, f"advantages length mismatch: {len(full_advantages)} vs {n_input}"
    
    return types.Datum(
        model_input=types.ModelInput.from_ints(input_tokens),
        loss_fn_inputs=dict(
            target_tokens=target_tokens,
            logprobs=full_logprobs,
            advantages=full_advantages
        )
    )


async def collect_rollouts(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    train_data: List[Dict],
    groups_per_batch: int,
    group_size: int
) -> List[RolloutGroup]:
    """Collect rollouts organized by problem groups."""
    
    stop_sequences = renderer.get_stop_sequences()
    params = types.SamplingParams(
        max_tokens=RL_MAX_TOKENS,
        temperature=RL_TEMPERATURE,
        stop=stop_sequences
    )
    
    # Sample random problems
    problem_indices = np.random.choice(len(train_data), size=groups_per_batch, replace=False)
    
    rollout_groups = []
    
    for problem_idx in problem_indices:
        example = train_data[problem_idx]
        gold = example.get("categories", [])
        messages = example.get("messages", [])
        
        # Build prompt (exclude assistant response)
        prompt_messages = messages[:-1] if messages else []
        if not prompt_messages:
            continue
            
        prompt = renderer.build_generation_prompt(prompt_messages)
        prompt_tokens = prompt.to_ints()
        
        # Generate group_size rollouts for this problem
        result = sampling_client.sample(
            prompt=prompt,
            sampling_params=params,
            num_samples=group_size
        ).result()
        
        rollouts = []
        for seq in result.sequences:
            response, success = renderer.parse_response(seq.tokens)
            predicted = response["content"] if success else ""
            reward, info = compute_reward(predicted, gold)
            
            # Only include if we have logprobs
            if seq.logprobs and len(seq.logprobs) == len(seq.tokens):
                rollouts.append(Rollout(
                    prompt_tokens=prompt_tokens,
                    gen_tokens=seq.tokens,
                    logprobs=seq.logprobs,
                    reward=reward,
                    predicted=predicted,
                    gold=gold
                ))
        
        if rollouts:
            rollout_groups.append(RolloutGroup(
                problem_id=problem_idx,
                rollouts=rollouts
            ))
    
    return rollout_groups


def filter_constant_reward_groups(groups: List[RolloutGroup]) -> List[RolloutGroup]:
    """
    Remove groups where all rollouts have the same reward.
    These provide no learning signal (gradient is zero).
    """
    return [g for g in groups if not g.is_constant_reward()]


async def run_sft(
    service_client: tinker.ServiceClient,
    training_client: tinker.TrainingClient,
    tokenizer,
    renderer: renderers.Renderer,
    train_data: List[Dict],
    test_data: List[Dict],
    logger: TrainingLogger
) -> Tuple[str, str]:
    """Run SFT phase."""
    print("\n" + "=" * 70)
    print("PHASE 1: SUPERVISED FINE-TUNING")
    print("=" * 70)
    
    lr = get_lr(BASE_MODEL)
    print(f"Learning rate: {lr:.2e}")
    print(f"Steps: {SFT_STEPS}, Batch size: {SFT_BATCH_SIZE}")
    print()
    
    # Convert to Datum
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
    
    train_datums = [to_datum(item) for item in train_data]
    test_datums = [to_datum(item) for item in test_data[:50]]
    
    for step in range(SFT_STEPS):
        step_start = time.time()
        
        # Get batch
        batch_idx = (step * SFT_BATCH_SIZE) % len(train_datums)
        batch = train_datums[batch_idx:batch_idx + SFT_BATCH_SIZE]
        if len(batch) < SFT_BATCH_SIZE:
            batch = batch + train_datums[:SFT_BATCH_SIZE - len(batch)]
        
        # Forward-backward
        fwd_future = await training_client.forward_backward_async(batch, loss_fn="cross_entropy")
        optim_future = await training_client.optim_step_async(
            types.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)
        )
        
        fwd_result = await fwd_future.result_async()
        await optim_future.result_async()
        
        # Compute loss
        logprobs = np.concatenate([o['logprobs'].tolist() for o in fwd_result.loss_fn_outputs])
        weights = np.concatenate([d.loss_fn_inputs['weights'].tolist() for d in batch])
        train_loss = -np.dot(logprobs, weights) / max(weights.sum(), 1)
        
        step_time = time.time() - step_start
        metrics = {"train_loss": float(train_loss), "step_time": step_time}
        
        # Evaluate every 10 steps
        if step % 10 == 0 or step == SFT_STEPS - 1:
            eval_future = await training_client.forward_backward_async(test_datums, loss_fn="cross_entropy")
            eval_result = await eval_future.result_async()
            test_logprobs = np.concatenate([o['logprobs'].tolist() for o in eval_result.loss_fn_outputs])
            test_weights = np.concatenate([d.loss_fn_inputs['weights'].tolist() for d in test_datums])
            test_loss = -np.dot(test_logprobs, test_weights) / max(test_weights.sum(), 1)
            metrics["test_loss"] = float(test_loss)
            
            # Save checkpoint
            save_future = await training_client.save_weights_for_sampler_async(name=f"sft_step_{step:04d}")
            save_result = await save_future.result_async()
            metrics["checkpoint"] = save_result.path
        
        logger.log_sft(step, metrics)
    
    # Save final state (for RL to continue from)
    state_future = await training_client.save_state_async(name="sft_final")
    state_result = await state_future.result_async()
    
    sampler_future = await training_client.save_weights_for_sampler_async(name="sft_final_sampler")
    sampler_result = await sampler_future.result_async()
    
    print(f"\nSFT Complete. State checkpoint: {state_result.path}")
    
    return state_result.path, sampler_result.path


async def run_rl(
    service_client: tinker.ServiceClient,
    training_client: tinker.TrainingClient,
    sft_state_path: str,
    tokenizer,
    renderer: renderers.Renderer,
    train_data: List[Dict],
    test_data: List[Dict],
    logger: TrainingLogger
) -> str:
    """Run RL phase with proper advantage computation."""
    print("\n" + "=" * 70)
    print("PHASE 2: REINFORCEMENT LEARNING")
    print("=" * 70)
    
    # Load SFT weights
    print(f"Loading SFT state from: {sft_state_path}")
    await training_client.load_state_async(sft_state_path)
    
    print(f"Iterations: {RL_ITERATIONS}")
    print(f"Groups per batch: {RL_GROUPS_PER_BATCH}")
    print(f"Group size: {RL_GROUP_SIZE}")
    print(f"Learning rate: {RL_LR:.2e}")
    print()
    
    for iteration in range(RL_ITERATIONS):
        iter_start = time.time()
        
        # 1. Save current weights for sampling
        save_future = await training_client.save_weights_for_sampler_async(name=f"rl_iter_{iteration:03d}")
        save_result = await save_future.result_async()
        sampling_client = service_client.create_sampling_client(model_path=save_result.path)
        
        # 2. Collect rollouts organized by problem groups
        rollout_groups = await collect_rollouts(
            sampling_client, renderer, train_data,
            RL_GROUPS_PER_BATCH, RL_GROUP_SIZE
        )
        
        # 3. Filter out constant-reward groups (no learning signal)
        active_groups = filter_constant_reward_groups(rollout_groups)
        
        # Collect metrics before filtering
        all_rewards = []
        for group in rollout_groups:
            all_rewards.extend(group.get_rewards())
        
        # 4. Compute advantages (centered within groups)
        group_advantages = compute_group_advantages(active_groups)
        
        # 5. Build training data
        training_data = []
        for group, advantages in zip(active_groups, group_advantages):
            for rollout, advantage in zip(group.rollouts, advantages):
                try:
                    datum = build_rl_datum(rollout, advantage)
                    training_data.append(datum)
                except AssertionError as e:
                    print(f"Warning: Skipping datum due to length mismatch: {e}")
                    continue
        
        # 6. Compute KL divergence estimate
        # Using the v1 estimator: E[log(p/q)] ≈ E[new_logprobs - old_logprobs]
        kl_samples = []
        if training_data:
            # We'll compute KL after the forward pass
            pass
        
        # 7. Update model
        if training_data:
            fwd_future = await training_client.forward_backward_async(
                training_data, loss_fn="importance_sampling"
            )
            optim_future = await training_client.optim_step_async(
                types.AdamParams(learning_rate=RL_LR, beta1=0.9, beta2=0.95, eps=1e-8)
            )
            
            fwd_result = await fwd_future.result_async()
            await optim_future.result_async()
            
            # Compute KL divergence from the forward pass
            # new_logprobs - old_logprobs
            for i, output in enumerate(fwd_result.loss_fn_outputs):
                new_logprobs = output['logprobs'].tolist()
                old_logprobs = training_data[i].loss_fn_inputs['logprobs'].tolist()
                # Only consider generation tokens (where old_logprobs != 0)
                for new_lp, old_lp in zip(new_logprobs, old_logprobs):
                    if old_lp != 0.0:
                        kl_samples.append(new_lp - old_lp)
        
        iter_time = time.time() - iter_start
        
        # Compute metrics
        mean_reward = np.mean(all_rewards) if all_rewards else 0
        std_reward = np.std(all_rewards) if all_rewards else 0
        accuracy = sum(1 for r in all_rewards if r > 0) / len(all_rewards) if all_rewards else 0
        kl_divergence = np.mean(kl_samples) if kl_samples else 0
        
        metrics = {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "accuracy": accuracy,
            "kl_divergence": float(kl_divergence),
            "total_groups": len(rollout_groups),
            "active_groups": len(active_groups),
            "num_training_examples": len(training_data),
            "iter_time": iter_time,
            "checkpoint": save_result.path
        }
        
        logger.log_rl(iteration, metrics)
        
        # Early stopping if KL divergence is too high
        if abs(kl_divergence) > 0.01:
            print(f"Warning: KL divergence {kl_divergence:.4f} exceeds threshold 0.01")
    
    # Save final checkpoint
    final_future = await training_client.save_weights_for_sampler_async(name="rl_final")
    final_result = await final_future.result_async()
    
    print(f"\nRL Complete. Final checkpoint: {final_result.path}")
    
    return final_result.path


async def evaluate_model(
    service_client: tinker.ServiceClient,
    model_path: str,
    renderer: renderers.Renderer,
    test_data: List[Dict],
    n_samples: int = 100
) -> Dict[str, float]:
    """Evaluate model on test data."""
    print(f"\nEvaluating: {model_path}")
    
    sampling_client = service_client.create_sampling_client(model_path=model_path)
    stop_sequences = renderer.get_stop_sequences()
    params = types.SamplingParams(max_tokens=100, temperature=0.1, stop=stop_sequences)
    
    correct_any = 0
    correct_exact = 0
    total_f1 = 0
    
    for item in test_data[:n_samples]:
        messages = item.get("messages", [])
        gold = item.get("categories", [])
        
        prompt_messages = messages[:-1] if messages else []
        if not prompt_messages:
            continue
        
        prompt = renderer.build_generation_prompt(prompt_messages)
        result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
        response, _ = renderer.parse_response(result.sequences[0].tokens)
        pred = response["content"]
        
        pred_set = set([c.strip().lower() for c in pred.split(",") 
                       if c.strip().lower() in VALID_CATEGORIES])
        gold_set = set([c.lower() for c in gold])
        
        if pred_set & gold_set:
            correct_any += 1
        if pred_set == gold_set:
            correct_exact += 1
        
        # F1
        if pred_set and gold_set:
            tp = len(pred_set & gold_set)
            precision = tp / len(pred_set)
            recall = tp / len(gold_set)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            total_f1 += f1
    
    n = min(n_samples, len(test_data))
    return {
        "any_match": correct_any / n,
        "exact_match": correct_exact / n,
        "f1": total_f1 / n
    }


async def main():
    print("=" * 70)
    print("MEMORY ROUTING AGENT - FIXED TRAINING PIPELINE")
    print("=" * 70)
    print(f"Log directory: {LOG_DIR}")
    print(f"Model: {BASE_MODEL}")
    print()
    
    # Initialize
    service_client = tinker.ServiceClient()
    tokenizer = get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(name="llama3", tokenizer=tokenizer)
    
    # Load data
    with open(TRAIN_DATA, "r") as f:
        train_data = json.load(f)
    with open(TEST_DATA, "r") as f:
        test_data = json.load(f)
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Create logger
    logger = TrainingLogger(LOG_DIR)
    
    # Create training client
    training_client = await service_client.create_lora_training_client_async(
        base_model=BASE_MODEL, rank=LORA_RANK
    )
    
    # Run SFT
    sft_state, sft_sampler = await run_sft(
        service_client, training_client, tokenizer, renderer,
        train_data, test_data, logger
    )
    
    # Evaluate SFT
    print("\n" + "-" * 70)
    sft_results = await evaluate_model(service_client, sft_sampler, renderer, test_data)
    print(f"SFT Results: Any={sft_results['any_match']:.1%}, Exact={sft_results['exact_match']:.1%}, F1={sft_results['f1']:.1%}")
    
    # Run RL
    rl_final = await run_rl(
        service_client, training_client, sft_state,
        tokenizer, renderer, train_data, test_data, logger
    )
    
    # Evaluate RL
    print("\n" + "-" * 70)
    rl_results = await evaluate_model(service_client, rl_final, renderer, test_data)
    print(f"RL Results: Any={rl_results['any_match']:.1%}, Exact={rl_results['exact_match']:.1%}, F1={rl_results['f1']:.1%}")
    
    logger.close()
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"Logs: {LOG_DIR}")
    print(f"SFT Checkpoint: {sft_sampler}")
    print(f"RL Checkpoint: {rl_final}")
    print()
    print("Performance Comparison:")
    print(f"{'Metric':<15} {'SFT':>10} {'RL':>10} {'Delta':>10}")
    print("-" * 45)
    for metric in ['any_match', 'exact_match', 'f1']:
        sft_val = sft_results[metric]
        rl_val = rl_results[metric]
        delta = rl_val - sft_val
        print(f"{metric:<15} {sft_val:>10.1%} {rl_val:>10.1%} {delta:>+10.1%}")


if __name__ == "__main__":
    asyncio.run(main())

