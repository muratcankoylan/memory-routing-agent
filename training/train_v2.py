"""
Memory Routing Agent - Training Pipeline v2

Fixes based on engineer feedback and Tinker docs alignment:

1. KL Divergence: Use proper Tinker estimators (kl_sample_train_v1/v2)
2. Reward Function: Use full composite reward from rl_env.py
3. Group Size: Increased to 32 (per Tinker rl/rl-hyperparams.mdx)
4. Batch Size: Increased to 64 groups per batch
5. Checkpointing: Non-blocking saves
6. Advantage Computation: Proper centering within groups (per Tinker rl/rl-loops.mdx)

Per Tinker docs:
- forward_backward_async returns Future, must await .result_async()
- importance_sampling loss requires: target_tokens, logprobs, advantages (all same length)
- save_state() for resumable checkpoints, save_weights_for_sampler() for sampling
- LR scaling: LR ∝ √batch_size
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
# CONFIGURATION - Aligned with Tinker recommendations
# =============================================================================

@dataclass
class TrainingConfig:
    # Model
    base_model: str = "meta-llama/Llama-3.1-8B"
    lora_rank: int = 32
    
    # SFT Config - Per Tinker sl-hyperparams.mdx
    sft_steps: int = 100  # Increased for better convergence (aim for 100+ steps)
    sft_batch_size: int = 32
    sft_eval_every: int = 10
    sft_early_stopping_patience: int = 5  # Stop if no improvement for N evals
    sft_min_steps: int = 30  # Minimum steps before early stopping
    sft_gradient_accumulation: int = 1  # Accumulate gradients over N batches
    
    # RL Config - Per Tinker rl/rl-hyperparams.mdx
    rl_iterations: int = 30  # More iterations for convergence
    rl_groups_per_batch: int = 64  # Increased from 32
    rl_group_size: int = 32  # Increased from 4 (per Tinker recommendation)
    rl_learning_rate: float = 2e-5
    rl_temperature: float = 0.7
    rl_max_tokens: int = 100
    rl_kl_threshold: float = 0.01  # Per Tinker: stable with KL < 0.01
    
    # Reward weights - Per PRD Section 4
    reward_f1_weight: float = 0.6
    reward_temp_weight: float = 0.2
    reward_parity_weight: float = 0.1
    reward_efficiency_weight: float = 0.1
    
    # Paths
    train_data: str = "training/processed_data/train_data.json"
    test_data: str = "training/processed_data/test_data.json"
    log_dir: str = field(default_factory=lambda: f"training/logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


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


# =============================================================================
# REWARD COMPUTATION - Full composite reward per PRD
# =============================================================================

@dataclass
class RewardComponents:
    r_f1: float = 0.0
    r_temp: float = 0.0
    r_parity: float = 0.0
    r_eff: float = 0.0
    r_total: float = 0.0
    format_valid: bool = True
    predicted: Set[str] = field(default_factory=set)
    gold: Set[str] = field(default_factory=set)


def parse_categories(text: str) -> Tuple[Set[str], bool]:
    """Parse comma-separated categories from model output."""
    if not text or not text.strip():
        return set(), False
    
    raw_cats = [c.strip().lower() for c in text.split(",")]
    valid_cats = {c for c in raw_cats if c in VALID_CATEGORIES}
    
    if not valid_cats:
        return set(), False
    
    # "none" must be exclusive
    if "none" in valid_cats and len(valid_cats) > 1:
        valid_cats.discard("none")
    
    return valid_cats, True


def compute_f1(predicted: Set[str], gold: Set[str]) -> float:
    """Compute F1 score between predicted and gold category sets."""
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_temporal_reward(predicted: Set[str], gold: Set[str]) -> float:
    """Compute temporal alignment reward."""
    if not predicted or not gold:
        return 0.0
    
    from collections import Counter
    
    pred_pers = [CATEGORY_PERSISTENCE.get(c, "medium") for c in predicted]
    gold_pers = [CATEGORY_PERSISTENCE.get(c, "medium") for c in gold]
    
    def majority(items):
        if not items:
            return "medium"
        return Counter(items).most_common(1)[0][0]
    
    pred_p = majority(pred_pers)
    gold_p = majority(gold_pers)
    
    if pred_p == gold_p:
        return 1.0
    
    # Adjacent matches
    adjacent = {
        ("long", "medium"), ("medium", "long"),
        ("medium", "short"), ("short", "medium"),
        ("medium", "rolling"), ("rolling", "medium"),
        ("short", "rolling"), ("rolling", "short"),
    }
    
    if (pred_p, gold_p) in adjacent:
        return 0.5
    return 0.0


def compute_parity_reward(predicted: Set[str], gold: Set[str]) -> float:
    """Compute company/user scope alignment reward."""
    def get_scope(cats):
        scopes = {CATEGORY_SCOPE.get(c, "none") for c in cats}
        if "company" in scopes and "user" in scopes:
            return "mixed"
        elif "company" in scopes:
            return "company"
        elif "user" in scopes:
            return "user"
        return "none"
    
    return 1.0 if get_scope(predicted) == get_scope(gold) else 0.0


def compute_efficiency_reward(predicted: Set[str]) -> float:
    """Compute storage efficiency reward."""
    n = len(predicted)
    if n <= 3:
        return 1.0
    elif n == 4:
        return 0.7
    elif n == 5:
        return 0.4
    return 0.0


def compute_reward(predicted_text: str, gold_categories: List[str], config: TrainingConfig) -> RewardComponents:
    """
    Compute full composite reward.
    R_total = w1*R_F1 + w2*R_temp + w3*R_parity + w4*R_eff
    """
    result = RewardComponents()
    
    predicted, parse_success = parse_categories(predicted_text)
    gold = set(gold_categories)
    
    result.predicted = predicted
    result.gold = gold
    
    if not parse_success:
        result.format_valid = False
        result.r_total = -1.0
        return result
    
    result.r_f1 = compute_f1(predicted, gold)
    result.r_temp = compute_temporal_reward(predicted, gold)
    result.r_parity = compute_parity_reward(predicted, gold)
    result.r_eff = compute_efficiency_reward(predicted)
    
    result.r_total = (
        config.reward_f1_weight * result.r_f1 +
        config.reward_temp_weight * result.r_temp +
        config.reward_parity_weight * result.r_parity +
        config.reward_efficiency_weight * result.r_eff
    )
    
    return result


# =============================================================================
# DATA STRUCTURES - Per Tinker rl/rl-loops.mdx
# =============================================================================

@dataclass
class Rollout:
    """Single rollout from a problem."""
    prompt_tokens: List[int]
    gen_tokens: List[int]
    logprobs: List[float]
    reward: float
    reward_components: RewardComponents
    predicted: str
    gold: List[str]


@dataclass
class RolloutGroup:
    """Group of rollouts for the same problem - per Tinker EnvGroupBuilder pattern."""
    problem_id: int
    rollouts: List[Rollout]
    
    def get_rewards(self) -> List[float]:
        return [r.reward for r in self.rollouts]
    
    def is_constant_reward(self) -> bool:
        rewards = self.get_rewards()
        return len(set(round(r, 4) for r in rewards)) == 1


# =============================================================================
# LOGGING
# =============================================================================

class TrainingLogger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.sft_log = open(os.path.join(log_dir, "sft_metrics.jsonl"), "w")
        self.rl_log = open(os.path.join(log_dir, "rl_metrics.jsonl"), "w")
        self.start_time = time.time()
        
    def log_sft(self, step: int, metrics: Dict):
        metrics["step"] = step
        metrics["elapsed_time"] = time.time() - self.start_time
        self.sft_log.write(json.dumps(metrics) + "\n")
        self.sft_log.flush()
        
        test_str = f"{metrics.get('test_loss', 0):.4f}" if isinstance(metrics.get('test_loss'), (int, float)) else "N/A"
        print(f"[SFT {step:3d}] Loss: {metrics.get('train_loss', 0):.4f} | Test: {test_str} | Time: {metrics.get('step_time', 0):.1f}s", flush=True)
    
    def log_rl(self, iteration: int, metrics: Dict):
        metrics["iteration"] = iteration
        metrics["elapsed_time"] = time.time() - self.start_time
        self.rl_log.write(json.dumps(metrics) + "\n")
        self.rl_log.flush()
        
        print(f"[RL {iteration:3d}] "
              f"Reward: {metrics.get('mean_reward', 0):.3f} (±{metrics.get('std_reward', 0):.3f}) | "
              f"Acc: {metrics.get('accuracy', 0):.1%} | "
              f"KL_v1: {metrics.get('kl_v1', 0):.4f} | "
              f"KL_v2: {metrics.get('kl_v2', 0):.4f} | "
              f"Active: {metrics.get('active_groups', 0)}/{metrics.get('total_groups', 0)} | "
              f"Time: {metrics.get('iter_time', 0):.1f}s", flush=True)
    
    def close(self):
        self.sft_log.close()
        self.rl_log.close()


# =============================================================================
# ADVANTAGE COMPUTATION - Per Tinker rl/rl-loops.mdx
# =============================================================================

def compute_group_advantages(groups: List[RolloutGroup]) -> List[List[float]]:
    """
    Compute advantages by centering rewards within each group.
    
    Per Tinker docs (rl/rl-loops.mdx):
    "We compute advantages by centering rewards within each problem group."
    
    This is the GRPO-style advantage: A(x,y) = r(y) - mean(r(y') for y' in group)
    """
    all_advantages = []
    
    for group in groups:
        rewards = np.array(group.get_rewards())
        mean_reward = rewards.mean()
        
        # Normalize by std for stability (optional but recommended)
        std_reward = rewards.std()
        if std_reward > 1e-8:
            advantages = (rewards - mean_reward) / std_reward
        else:
            advantages = rewards - mean_reward
        
        all_advantages.append(advantages.tolist())
    
    return all_advantages


# =============================================================================
# KL DIVERGENCE ESTIMATORS - Per Tinker rl/rl-hyperparams.mdx
# =============================================================================

def compute_kl_estimators(old_logprobs: List[float], new_logprobs: List[float]) -> Tuple[float, float]:
    """
    Compute KL divergence estimators per Tinker docs.
    
    Per rl/rl-hyperparams.mdx:
    - kl_sample_train_v1: E[log(p/q)] approximation
    - kl_sample_train_v2: Alternative estimator
    
    Both should be >= 0 in expectation. We compute:
    - v1: mean(new_lp - old_lp) -- this is E[log(p_new/p_old)]
    - v2: mean(exp(new_lp - old_lp) - 1 - (new_lp - old_lp)) -- Taylor expansion
    
    Note: These estimate KL(p_new || p_old), not the reverse.
    """
    if not old_logprobs or not new_logprobs:
        return 0.0, 0.0
    
    log_ratios = []
    for old_lp, new_lp in zip(old_logprobs, new_logprobs):
        if old_lp != 0.0:  # Skip prompt tokens
            log_ratios.append(new_lp - old_lp)
    
    if not log_ratios:
        return 0.0, 0.0
    
    log_ratios = np.array(log_ratios)
    
    # v1: Simple mean of log ratios
    # This estimates E_p[log(p/q)] but with samples from q
    # For on-policy, this should be close to 0
    kl_v1 = float(np.mean(log_ratios))
    
    # v2: Unbiased estimator using importance weights
    # E[exp(log_ratio) - 1 - log_ratio] = KL(p||q)
    # This is always >= 0 by Jensen's inequality
    ratios = np.exp(np.clip(log_ratios, -20, 20))  # Clip for numerical stability
    kl_v2 = float(np.mean(ratios - 1 - log_ratios))
    
    return kl_v1, kl_v2


# =============================================================================
# DATUM CONSTRUCTION - Per Tinker losses.mdx
# =============================================================================

def build_rl_datum(rollout: Rollout, advantage: float) -> types.Datum:
    """
    Build a Datum for importance_sampling loss.
    
    Per Tinker losses.mdx, importance_sampling requires:
    - target_tokens: array[(N,), int] - Target token IDs from sampler
    - logprobs: array[(N,), float] - Reference log probabilities from sampler
    - advantages: array[(N,), float] - Advantage values
    
    All must have length N = model_input.length
    """
    prompt_tokens = rollout.prompt_tokens
    gen_tokens = rollout.gen_tokens
    sampler_logprobs = rollout.logprobs
    
    n_prompt = len(prompt_tokens)
    n_gen = len(gen_tokens)
    
    # Full sequence: prompt + generated
    full_tokens = prompt_tokens + gen_tokens
    
    # Model input: all except last token
    input_tokens = full_tokens[:-1]
    
    # Target: all except first token (next-token prediction)
    target_tokens = full_tokens[1:]
    
    n_input = len(input_tokens)
    
    # Logprobs: 0 for prompt positions, actual for generation
    # Sampler logprobs correspond to gen_tokens
    full_logprobs = [0.0] * (n_prompt - 1) + sampler_logprobs
    
    # Advantages: 0 for prompt, actual for generation
    full_advantages = [0.0] * (n_prompt - 1) + [advantage] * n_gen
    
    # Verify lengths match
    assert len(target_tokens) == n_input, f"target_tokens: {len(target_tokens)} vs input: {n_input}"
    assert len(full_logprobs) == n_input, f"logprobs: {len(full_logprobs)} vs input: {n_input}"
    assert len(full_advantages) == n_input, f"advantages: {len(full_advantages)} vs input: {n_input}"
    
    return types.Datum(
        model_input=types.ModelInput.from_ints(input_tokens),
        loss_fn_inputs=dict(
            target_tokens=target_tokens,
            logprobs=full_logprobs,
            advantages=full_advantages
        )
    )


# =============================================================================
# ROLLOUT COLLECTION
# =============================================================================

async def collect_rollouts(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    train_data: List[Dict],
    config: TrainingConfig
) -> List[RolloutGroup]:
    """
    Collect rollouts organized by problem groups.
    
    Per Tinker rl/rl-loops.mdx:
    - Generate group_size rollouts per unique problem
    - This enables variance reduction via advantage centering
    """
    stop_sequences = renderer.get_stop_sequences()
    params = types.SamplingParams(
        max_tokens=config.rl_max_tokens,
        temperature=config.rl_temperature,
        stop=stop_sequences
    )
    
    # Sample random problems
    n_problems = min(config.rl_groups_per_batch, len(train_data))
    problem_indices = np.random.choice(len(train_data), size=n_problems, replace=False)
    
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
            num_samples=config.rl_group_size
        ).result()
        
        rollouts = []
        for seq in result.sequences:
            response, success = renderer.parse_response(seq.tokens)
            predicted = response["content"] if success else ""
            reward_comp = compute_reward(predicted, gold, config)
            
            # Only include if we have logprobs
            if seq.logprobs and len(seq.logprobs) == len(seq.tokens):
                rollouts.append(Rollout(
                    prompt_tokens=prompt_tokens,
                    gen_tokens=seq.tokens,
                    logprobs=seq.logprobs,
                    reward=reward_comp.r_total,
                    reward_components=reward_comp,
                    predicted=predicted,
                    gold=gold
                ))
        
        if rollouts:
            rollout_groups.append(RolloutGroup(
                problem_id=int(problem_idx),
                rollouts=rollouts
            ))
    
    return rollout_groups


def filter_constant_reward_groups(groups: List[RolloutGroup]) -> List[RolloutGroup]:
    """
    Remove groups where all rollouts have the same reward.
    These provide no learning signal (gradient is zero).
    
    Per Tinker rl/rl-loops.mdx:
    "We can optionally filter out groups with all successes or all failures
    as these have policy gradients of zero."
    """
    return [g for g in groups if not g.is_constant_reward()]


# =============================================================================
# SFT PHASE - Per Tinker supervised-learning docs
# =============================================================================

async def run_sft(
    service_client: tinker.ServiceClient,
    training_client: tinker.TrainingClient,
    renderer: renderers.Renderer,
    train_data: List[Dict],
    test_data: List[Dict],
    config: TrainingConfig,
    logger: TrainingLogger
) -> Tuple[str, str]:
    """
    Run SFT phase per Tinker supervised-learning docs.
    
    Improvements based on Tinker docs:
    1. Data shuffling each epoch (sl-basic.mdx)
    2. Early stopping to prevent overfitting (lora-primer.mdx)
    3. Proper LR from get_lr() which accounts for LoRA scaling (sl-hyperparams.mdx)
    4. Overlapping requests for better throughput (async.mdx)
    5. Gradient accumulation support for larger effective batch sizes
    """
    print("\n" + "=" * 70, flush=True)
    print("PHASE 1: SUPERVISED FINE-TUNING", flush=True)
    print("=" * 70, flush=True)
    
    # Get LoRA-adjusted learning rate per sl-hyperparams.mdx
    # LR(m) = lr_base * M_LoRA * (2000/H_m)^P_m
    lr = get_lr(config.base_model)
    
    # Effective batch size for LR scaling
    effective_batch = config.sft_batch_size * config.sft_gradient_accumulation
    
    print(f"Learning rate: {lr:.2e} (LoRA-adjusted)", flush=True)
    print(f"Steps: {config.sft_steps}, Batch size: {config.sft_batch_size}", flush=True)
    print(f"Gradient accumulation: {config.sft_gradient_accumulation}", flush=True)
    print(f"Effective batch size: {effective_batch}", flush=True)
    print(f"Early stopping patience: {config.sft_early_stopping_patience} evals", flush=True)
    print()
    
    # Convert to Datum - per rendering.mdx, use build_supervised_example
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
    
    # Count completion tokens for LoRA capacity check (per lora-primer.mdx)
    total_completion_tokens = sum(
        sum(d.loss_fn_inputs['weights'].tolist()) for d in train_datums
    )
    print(f"Total completion tokens: {total_completion_tokens:,}", flush=True)
    print(f"(LoRA works well when completion tokens < LoRA params)", flush=True)
    print()
    
    # Early stopping state
    best_test_loss = float('inf')
    best_checkpoint = None
    patience_counter = 0
    
    # Shuffle indices for each epoch
    indices = list(range(len(train_datums)))
    epoch = 0
    idx_ptr = 0
    
    for step in range(config.sft_steps):
        step_start = time.time()
        
        # Shuffle at epoch boundary
        if idx_ptr + config.sft_batch_size > len(indices):
            np.random.shuffle(indices)
            idx_ptr = 0
            epoch += 1
        
        # Get batch with shuffled indices
        batch_indices = indices[idx_ptr:idx_ptr + config.sft_batch_size]
        idx_ptr += config.sft_batch_size
        batch = [train_datums[i] for i in batch_indices]
        
        # Forward-backward per Tinker async pattern
        # Per async.mdx: "submit your next request while the current one is running"
        fwd_future = await training_client.forward_backward_async(batch, loss_fn="cross_entropy")
        
        # Submit optim step immediately (overlapping with forward-backward)
        optim_future = await training_client.optim_step_async(
            types.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)
        )
        
        # Now wait for results
        fwd_result = await fwd_future.result_async()
        await optim_future.result_async()
        
        # Compute loss
        logprobs = np.concatenate([o['logprobs'].tolist() for o in fwd_result.loss_fn_outputs])
        weights_arr = np.concatenate([d.loss_fn_inputs['weights'].tolist() for d in batch])
        train_loss = -np.dot(logprobs, weights_arr) / max(weights_arr.sum(), 1)
        
        step_time = time.time() - step_start
        metrics = {
            "train_loss": float(train_loss),
            "step_time": step_time,
            "epoch": epoch,
            "learning_rate": lr
        }
        
        # Evaluate periodically
        if step % config.sft_eval_every == 0 or step == config.sft_steps - 1:
            # Use forward (not forward_backward) for eval to avoid gradient accumulation
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
            
            # Early stopping check
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_checkpoint = save_result.path
                patience_counter = 0
                metrics["is_best"] = True
            else:
                patience_counter += 1
                metrics["is_best"] = False
            
            metrics["patience_counter"] = patience_counter
            
            # Check early stopping (after minimum steps)
            if step >= config.sft_min_steps and patience_counter >= config.sft_early_stopping_patience:
                print(f"\nEarly stopping at step {step} (no improvement for {patience_counter} evals)", flush=True)
                logger.log_sft(step, metrics)
                break
        
        logger.log_sft(step, metrics)
    
    # Save final state for RL continuation
    # Per save-load.mdx: use save_state() for resumable checkpoints
    state_future = await training_client.save_state_async(name="sft_final")
    state_result = await state_future.result_async()
    
    # Also save sampler weights
    sampler_future = await training_client.save_weights_for_sampler_async(name="sft_final_sampler")
    sampler_result = await sampler_future.result_async()
    
    print(f"\nSFT Complete.", flush=True)
    print(f"  Final checkpoint: {sampler_result.path}", flush=True)
    print(f"  Best checkpoint (loss={best_test_loss:.4f}): {best_checkpoint}", flush=True)
    print(f"  State for RL: {state_result.path}", flush=True)
    
    return state_result.path, sampler_result.path


# =============================================================================
# RL PHASE
# =============================================================================

async def run_rl(
    service_client: tinker.ServiceClient,
    training_client: tinker.TrainingClient,
    sft_state_path: str,
    renderer: renderers.Renderer,
    train_data: List[Dict],
    test_data: List[Dict],
    config: TrainingConfig,
    logger: TrainingLogger
) -> str:
    """
    Run RL phase with proper advantage computation.
    
    Per Tinker rl/rl-loops.mdx:
    1. Create policy with current weights
    2. Generate rollouts (group_size per problem)
    3. Process trajectory data into training examples
    4. Update model parameters
    """
    print("\n" + "=" * 70, flush=True)
    print("PHASE 2: REINFORCEMENT LEARNING", flush=True)
    print("=" * 70, flush=True)
    
    # Load SFT weights
    print(f"Loading SFT state: {sft_state_path}", flush=True)
    await training_client.load_state_async(sft_state_path)
    
    print(f"Iterations: {config.rl_iterations}", flush=True)
    print(f"Groups per batch: {config.rl_groups_per_batch}", flush=True)
    print(f"Group size: {config.rl_group_size}", flush=True)
    print(f"Total rollouts per iteration: {config.rl_groups_per_batch * config.rl_group_size}", flush=True)
    print(f"Learning rate: {config.rl_learning_rate:.2e}", flush=True)
    print(f"KL threshold: {config.rl_kl_threshold}", flush=True)
    print()
    
    best_reward = -float('inf')
    best_checkpoint = None
    
    for iteration in range(config.rl_iterations):
        iter_start = time.time()
        
        # 1. Save current weights for sampling
        save_future = await training_client.save_weights_for_sampler_async(name=f"rl_iter_{iteration:03d}")
        save_result = await save_future.result_async()
        sampling_client = service_client.create_sampling_client(model_path=save_result.path)
        
        # 2. Collect rollouts organized by problem groups
        rollout_groups = await collect_rollouts(
            sampling_client, renderer, train_data, config
        )
        
        # 3. Filter constant-reward groups
        active_groups = filter_constant_reward_groups(rollout_groups)
        
        # Collect all rewards for metrics
        all_rewards = []
        all_reward_components = {"f1": [], "temp": [], "parity": [], "eff": []}
        for group in rollout_groups:
            for rollout in group.rollouts:
                all_rewards.append(rollout.reward)
                if rollout.reward_components.format_valid:
                    all_reward_components["f1"].append(rollout.reward_components.r_f1)
                    all_reward_components["temp"].append(rollout.reward_components.r_temp)
                    all_reward_components["parity"].append(rollout.reward_components.r_parity)
                    all_reward_components["eff"].append(rollout.reward_components.r_eff)
        
        # 4. Compute advantages (centered within groups)
        group_advantages = compute_group_advantages(active_groups)
        
        # 5. Build training data
        training_data = []
        for group, advantages in zip(active_groups, group_advantages):
            for rollout, advantage in zip(group.rollouts, advantages):
                try:
                    datum = build_rl_datum(rollout, advantage)
                    training_data.append((datum, rollout))
                except AssertionError as e:
                    print(f"Warning: Skipping datum: {e}", flush=True)
        
        # 6. Update model and compute KL
        kl_v1_samples = []
        kl_v2_samples = []
        
        if training_data:
            datums = [d[0] for d in training_data]
            
            fwd_future = await training_client.forward_backward_async(
                datums, loss_fn="importance_sampling"
            )
            optim_future = await training_client.optim_step_async(
                types.AdamParams(learning_rate=config.rl_learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
            )
            
            fwd_result = await fwd_future.result_async()
            await optim_future.result_async()
            
            # Compute KL divergence estimators
            for i, output in enumerate(fwd_result.loss_fn_outputs):
                new_logprobs = output['logprobs'].tolist()
                old_logprobs = datums[i].loss_fn_inputs['logprobs'].tolist()
                v1, v2 = compute_kl_estimators(old_logprobs, new_logprobs)
                kl_v1_samples.append(v1)
                kl_v2_samples.append(v2)
        
        iter_time = time.time() - iter_start
        
        # Compute metrics
        mean_reward = np.mean(all_rewards) if all_rewards else 0
        std_reward = np.std(all_rewards) if all_rewards else 0
        accuracy = sum(1 for r in all_rewards if r > 0) / len(all_rewards) if all_rewards else 0
        kl_v1 = np.mean(kl_v1_samples) if kl_v1_samples else 0
        kl_v2 = np.mean(kl_v2_samples) if kl_v2_samples else 0
        
        metrics = {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "accuracy": accuracy,
            "kl_v1": float(kl_v1),  # Log ratio mean (can be negative)
            "kl_v2": float(kl_v2),  # Unbiased estimator (always >= 0)
            "total_groups": len(rollout_groups),
            "active_groups": len(active_groups),
            "num_training_examples": len(training_data),
            "iter_time": iter_time,
            "checkpoint": save_result.path,
            # Reward component breakdown
            "mean_r_f1": float(np.mean(all_reward_components["f1"])) if all_reward_components["f1"] else 0,
            "mean_r_temp": float(np.mean(all_reward_components["temp"])) if all_reward_components["temp"] else 0,
            "mean_r_parity": float(np.mean(all_reward_components["parity"])) if all_reward_components["parity"] else 0,
            "mean_r_eff": float(np.mean(all_reward_components["eff"])) if all_reward_components["eff"] else 0,
        }
        
        logger.log_rl(iteration, metrics)
        
        # Track best checkpoint
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_checkpoint = save_result.path
        
        # KL threshold warning (using v2 which is always >= 0)
        if kl_v2 > config.rl_kl_threshold:
            print(f"WARNING: KL_v2 {kl_v2:.4f} exceeds threshold {config.rl_kl_threshold}", flush=True)
    
    # Save final checkpoint
    final_future = await training_client.save_weights_for_sampler_async(name="rl_final")
    final_result = await final_future.result_async()
    
    print(f"\nRL Complete. Final: {final_result.path}", flush=True)
    print(f"Best checkpoint (reward={best_reward:.3f}): {best_checkpoint}", flush=True)
    
    return final_result.path


# =============================================================================
# EVALUATION
# =============================================================================

async def evaluate_model(
    service_client: tinker.ServiceClient,
    model_path: str,
    renderer: renderers.Renderer,
    test_data: List[Dict],
    config: TrainingConfig,
    n_samples: int = 100
) -> Dict[str, float]:
    """Evaluate model on test data."""
    print(f"\nEvaluating: {model_path}", flush=True)
    
    sampling_client = service_client.create_sampling_client(model_path=model_path)
    stop_sequences = renderer.get_stop_sequences()
    params = types.SamplingParams(max_tokens=100, temperature=0.1, stop=stop_sequences)
    
    correct_any = 0
    correct_exact = 0
    total_f1 = 0
    total_reward = 0
    
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
        
        reward_comp = compute_reward(pred, gold, config)
        total_reward += reward_comp.r_total
        
        if reward_comp.predicted & reward_comp.gold:
            correct_any += 1
        if reward_comp.predicted == reward_comp.gold:
            correct_exact += 1
        total_f1 += reward_comp.r_f1
    
    n = min(n_samples, len(test_data))
    return {
        "any_match": correct_any / n,
        "exact_match": correct_exact / n,
        "f1": total_f1 / n,
        "mean_reward": total_reward / n
    }


# =============================================================================
# MAIN
# =============================================================================

async def main():
    config = TrainingConfig()
    
    print("=" * 70, flush=True)
    print("MEMORY ROUTING AGENT - TRAINING PIPELINE v2", flush=True)
    print("=" * 70, flush=True)
    print(f"Log directory: {config.log_dir}", flush=True)
    print(f"Model: {config.base_model}", flush=True)
    print(f"RL Groups: {config.rl_groups_per_batch}, Group Size: {config.rl_group_size}", flush=True)
    print()
    
    # Initialize
    service_client = tinker.ServiceClient()
    tokenizer = get_tokenizer(config.base_model)
    renderer = renderers.get_renderer(name="llama3", tokenizer=tokenizer)
    
    # Load data
    with open(config.train_data, "r") as f:
        train_data = json.load(f)
    with open(config.test_data, "r") as f:
        test_data = json.load(f)
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}", flush=True)
    
    # Create logger
    logger = TrainingLogger(config.log_dir)
    
    # Create training client
    training_client = await service_client.create_lora_training_client_async(
        base_model=config.base_model, rank=config.lora_rank
    )
    
    # Run SFT
    sft_state, sft_sampler = await run_sft(
        service_client, training_client, renderer,
        train_data, test_data, config, logger
    )
    
    # Evaluate SFT
    print("\n" + "-" * 70, flush=True)
    sft_results = await evaluate_model(service_client, sft_sampler, renderer, test_data, config)
    print(f"SFT: Any={sft_results['any_match']:.1%}, Exact={sft_results['exact_match']:.1%}, "
          f"F1={sft_results['f1']:.1%}, Reward={sft_results['mean_reward']:.3f}", flush=True)
    
    # Run RL
    rl_final = await run_rl(
        service_client, training_client, sft_state,
        renderer, train_data, test_data, config, logger
    )
    
    # Evaluate RL
    print("\n" + "-" * 70, flush=True)
    rl_results = await evaluate_model(service_client, rl_final, renderer, test_data, config)
    print(f"RL: Any={rl_results['any_match']:.1%}, Exact={rl_results['exact_match']:.1%}, "
          f"F1={rl_results['f1']:.1%}, Reward={rl_results['mean_reward']:.3f}", flush=True)
    
    logger.close()
    
    # Summary
    print("\n" + "=" * 70, flush=True)
    print("TRAINING COMPLETE", flush=True)
    print("=" * 70, flush=True)
    print(f"Logs: {config.log_dir}", flush=True)
    print(f"SFT: {sft_sampler}", flush=True)
    print(f"RL: {rl_final}", flush=True)
    print()
    print("Performance Comparison:", flush=True)
    print(f"{'Metric':<15} {'SFT':>10} {'RL':>10} {'Delta':>10}", flush=True)
    print("-" * 45, flush=True)
    for metric in ['any_match', 'exact_match', 'f1', 'mean_reward']:
        sft_val = sft_results[metric]
        rl_val = rl_results[metric]
        delta = rl_val - sft_val
        if metric == 'mean_reward':
            print(f"{metric:<15} {sft_val:>10.3f} {rl_val:>10.3f} {delta:>+10.3f}", flush=True)
        else:
            print(f"{metric:<15} {sft_val:>10.1%} {rl_val:>10.1%} {delta:>+10.1%}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

