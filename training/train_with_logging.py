"""
Training script with detailed real-time logging.

This script logs:
- SFT: loss, test loss, accuracy at checkpoints
- RL: reward, accuracy, KL divergence per iteration
"""

import asyncio
import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.hyperparam_utils import get_lr
import numpy as np
from collections import Counter

# Configuration
BASE_MODEL = "meta-llama/Llama-3.1-8B"
LORA_RANK = 32
SFT_STEPS = 50
SFT_BATCH_SIZE = 32
RL_ITERATIONS = 15
RL_BATCH_SIZE = 16
RL_GROUP_SIZE = 4
RL_LR = 1e-5

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
        
        # Print to console
        print(f"[SFT Step {step:3d}] "
              f"Loss: {metrics.get('train_loss', 0):.4f} | "
              f"Test: {metrics.get('test_loss', 'N/A')} | "
              f"Acc: {metrics.get('accuracy', 'N/A')} | "
              f"Time: {metrics.get('step_time', 0):.1f}s")
    
    def log_rl(self, iteration, metrics):
        metrics["iteration"] = iteration
        metrics["elapsed_time"] = time.time() - self.start_time
        self.rl_log.write(json.dumps(metrics) + "\n")
        self.rl_log.flush()
        
        # Print to console
        print(f"[RL Iter {iteration:3d}] "
              f"Reward: {metrics.get('mean_reward', 0):.3f} | "
              f"Acc: {metrics.get('accuracy', 0):.1%} | "
              f"Format: {metrics.get('format_valid', 0):.1%} | "
              f"Time: {metrics.get('iter_time', 0):.1f}s")
    
    def close(self):
        self.sft_log.close()
        self.rl_log.close()


def compute_reward(predicted_text, gold_categories):
    """Compute reward for RL."""
    if not predicted_text or not predicted_text.strip():
        return -1.0, {"format_valid": False}
    
    predicted = set([c.strip().lower() for c in predicted_text.split(",") 
                     if c.strip().lower() in VALID_CATEGORIES])
    
    if not predicted:
        return -1.0, {"format_valid": False}
    
    gold = set([c.lower() for c in gold_categories])
    
    # F1 Score
    if predicted and gold:
        tp = len(predicted & gold)
        precision = tp / len(predicted)
        recall = tp / len(gold)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        f1 = 1.0 if not predicted and not gold else 0.0
    
    return f1, {"format_valid": True, "f1": f1}


async def evaluate_accuracy(sampling_client, renderer, test_data, n_samples=20):
    """Quick accuracy evaluation."""
    stop_sequences = renderer.get_stop_sequences()
    correct = 0
    
    for item in test_data[:n_samples]:
        messages = item.get("messages", [])
        gold = item.get("categories", [])
        
        # Build prompt
        prompt_messages = messages[:-1]  # Exclude assistant response
        prompt = renderer.build_generation_prompt(prompt_messages)
        params = types.SamplingParams(max_tokens=100, temperature=0.1, stop=stop_sequences)
        
        result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
        response, _ = renderer.parse_response(result.sequences[0].tokens)
        pred = response["content"]
        
        pred_set = set([c.strip().lower() for c in pred.split(",") 
                       if c.strip().lower() in VALID_CATEGORIES])
        gold_set = set([c.lower() for c in gold])
        
        if pred_set & gold_set:
            correct += 1
    
    return correct / n_samples


async def run_sft(service_client, training_client, tokenizer, renderer, 
                  train_data, test_data, logger):
    """Run SFT with detailed logging."""
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
            # Test loss
            eval_future = await training_client.forward_backward_async(test_datums, loss_fn="cross_entropy")
            eval_result = await eval_future.result_async()
            test_logprobs = np.concatenate([o['logprobs'].tolist() for o in eval_result.loss_fn_outputs])
            test_weights = np.concatenate([d.loss_fn_inputs['weights'].tolist() for d in test_datums])
            test_loss = -np.dot(test_logprobs, test_weights) / max(test_weights.sum(), 1)
            metrics["test_loss"] = float(test_loss)
            
            # Save checkpoint and evaluate accuracy
            save_future = await training_client.save_weights_for_sampler_async(name=f"sft_step_{step:04d}")
            save_result = await save_future.result_async()
            
            sampling_client = service_client.create_sampling_client(model_path=save_result.path)
            accuracy = await evaluate_accuracy(sampling_client, renderer, test_data, n_samples=20)
            metrics["accuracy"] = f"{accuracy:.1%}"
            metrics["checkpoint"] = save_result.path
        
        logger.log_sft(step, metrics)
    
    # Save final state
    state_future = await training_client.save_state_async(name="sft_final")
    state_result = await state_future.result_async()
    
    sampler_future = await training_client.save_weights_for_sampler_async(name="sft_final_sampler")
    sampler_result = await sampler_future.result_async()
    
    print(f"\nSFT Complete. State: {state_result.path}")
    
    return state_result.path, sampler_result.path


async def run_rl(service_client, training_client, sft_state_path, 
                 tokenizer, renderer, train_data, test_data, logger):
    """Run RL with detailed logging."""
    print("\n" + "=" * 70)
    print("PHASE 2: REINFORCEMENT LEARNING")
    print("=" * 70)
    
    print(f"Loading SFT weights from: {sft_state_path}")
    await training_client.load_state_async(sft_state_path)
    
    print(f"Iterations: {RL_ITERATIONS}, Batch: {RL_BATCH_SIZE}, Group: {RL_GROUP_SIZE}")
    print()
    
    stop_sequences = renderer.get_stop_sequences()
    
    for iteration in range(RL_ITERATIONS):
        iter_start = time.time()
        
        # Save current weights for sampling
        save_future = await training_client.save_weights_for_sampler_async(name=f"rl_iter_{iteration:03d}")
        save_result = await save_future.result_async()
        sampling_client = service_client.create_sampling_client(model_path=save_result.path)
        
        # Sample batch
        batch_indices = np.random.choice(len(train_data), size=RL_BATCH_SIZE, replace=False)
        
        all_rewards = []
        format_valid_count = 0
        training_data = []
        
        for idx in batch_indices:
            example = train_data[idx]
            gold = example.get("categories", [])
            messages = example.get("messages", [])
            prompt_messages = messages[:-1]
            
            if not prompt_messages:
                continue
            
            prompt = renderer.build_generation_prompt(prompt_messages)
            params = types.SamplingParams(
                max_tokens=100, temperature=0.7, stop=stop_sequences
            )
            
            result = sampling_client.sample(
                prompt=prompt, sampling_params=params, num_samples=RL_GROUP_SIZE
            ).result()
            
            for seq in result.sequences:
                response, success = renderer.parse_response(seq.tokens)
                predicted = response["content"] if success else ""
                reward, info = compute_reward(predicted, gold)
                
                all_rewards.append(reward)
                if info["format_valid"]:
                    format_valid_count += 1
                
                # Build training example (simplified)
                if seq.logprobs and reward > -1:
                    prompt_tokens = prompt.to_ints()
                    gen_tokens = seq.tokens
                    logprobs = seq.logprobs
                    
                    n_prompt = len(prompt_tokens) - 1
                    n_gen = len(gen_tokens)
                    
                    if len(logprobs) == n_gen:
                        full_input = prompt_tokens + gen_tokens[:-1] if n_gen > 1 else prompt_tokens
                        full_target = prompt_tokens[1:] + gen_tokens
                        full_logprobs = [0.0] * n_prompt + logprobs
                        full_advantages = [0.0] * n_prompt + [reward] * n_gen
                        
                        if len(full_target) == len(full_input):
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
            # Normalize advantages
            rewards_arr = np.array(all_rewards)
            mean_r = rewards_arr.mean()
            std_r = rewards_arr.std() + 1e-8
            
            fwd_future = await training_client.forward_backward_async(
                training_data, loss_fn="importance_sampling"
            )
            optim_future = await training_client.optim_step_async(
                types.AdamParams(learning_rate=RL_LR, beta1=0.9, beta2=0.95, eps=1e-8)
            )
            await fwd_future.result_async()
            await optim_future.result_async()
        
        iter_time = time.time() - iter_start
        
        metrics = {
            "mean_reward": float(np.mean(all_rewards)),
            "std_reward": float(np.std(all_rewards)),
            "accuracy": sum(1 for r in all_rewards if r > 0) / len(all_rewards) if all_rewards else 0,
            "format_valid": format_valid_count / len(all_rewards) if all_rewards else 0,
            "num_rollouts": len(all_rewards),
            "num_training": len(training_data),
            "iter_time": iter_time,
            "checkpoint": save_result.path
        }
        
        logger.log_rl(iteration, metrics)
    
    # Save final
    final_future = await training_client.save_weights_for_sampler_async(name="rl_final")
    final_result = await final_future.result_async()
    
    print(f"\nRL Complete. Final: {final_result.path}")
    
    return final_result.path


async def main():
    print("=" * 70)
    print("MEMORY ROUTING AGENT - TRAINING WITH DETAILED LOGGING")
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
    
    # Run RL
    rl_final = await run_rl(
        service_client, training_client, sft_state,
        tokenizer, renderer, train_data, test_data, logger
    )
    
    logger.close()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Logs: {LOG_DIR}")
    print(f"SFT: {sft_sampler}")
    print(f"RL:  {rl_final}")


if __name__ == "__main__":
    asyncio.run(main())

