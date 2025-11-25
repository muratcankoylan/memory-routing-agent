"""
Reinforcement Learning Training for Memory Routing

This implements Stage 2 of the PRD: RL Optimization using Tinker's
importance_sampling loss function.

Per Tinker docs (rl.mdx):
- RL learns from trial and error with reward functions
- Use importance_sampling loss for policy gradient

Per Tinker docs (rl/rl-loops.mdx):
1. Create policy with current weights
2. Generate rollouts
3. Process trajectory data into training examples
4. Update model parameters

Per PRD Section 8:
- 25 iterations minimum
- Group size 8 for variance reduction
- KL divergence monitoring (<0.01 threshold)
"""

import asyncio
import json
import time
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import Counter

# Configuration
@dataclass
class RLConfig:
    # Model - start from SFT checkpoint
    sft_checkpoint: str = ""  # Will be set from command line
    base_model: str = "meta-llama/Llama-3.1-8B"
    lora_rank: int = 32
    renderer_name: str = "llama3"
    
    # Training
    num_iterations: int = 25
    batch_size: int = 32  # Number of unique conversations per iteration
    group_size: int = 8   # Rollouts per conversation for variance reduction
    learning_rate: float = 1e-5  # Lower than SFT per Tinker RL docs
    
    # Adam optimizer
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Sampling
    max_tokens: int = 50
    temperature: float = 0.7
    
    # Monitoring
    kl_threshold: float = 0.01  # Per PRD: warn if KL > 0.01
    checkpoint_every: int = 5
    
    # Paths
    train_data_path: str = "training/processed_data/train_data.json"
    log_path: str = "training/logs/rl"


# Import reward computation from rl_env
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_env import compute_reward, VALID_CATEGORIES


def build_routing_prompt(conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Build the routing prompt for a conversation."""
    system_content = """You route marketing conversations into structured memory categories.

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

Respond with comma-separated categories."""

    # Format the conversation
    conversation_text = ""
    for turn in conversation:
        if isinstance(turn, dict):
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            conversation_text += f"{role.upper()}: {content}\n"
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Conversation:\n{conversation_text.strip()}\n\nCategories?"}
    ]


async def run_rl_training(config: RLConfig):
    """
    Main RL training loop.
    
    Per Tinker docs (rl/rl-loops.mdx):
    1. Create policy with current weights
    2. Generate rollouts (sample from model)
    3. Compute rewards and advantages
    4. Update with importance_sampling loss
    """
    import tinker
    from tinker import types
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tinker_cookbook.hyperparam_utils import get_lr
    from dotenv import load_dotenv
    import numpy as np
    
    load_dotenv()
    os.makedirs(config.log_path, exist_ok=True)
    
    print("=" * 70)
    print("REINFORCEMENT LEARNING TRAINING")
    print("=" * 70)
    
    # Load training data
    print(f"\nLoading training data from {config.train_data_path}...")
    with open(config.train_data_path, "r") as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} examples")
    
    # Initialize Tinker
    print("\nInitializing Tinker...")
    service_client = tinker.ServiceClient()
    
    # For RL, we need to:
    # 1. Create a training client (fresh LoRA weights)
    # 2. Use the SFT checkpoint directly for sampling
    # Note: We cannot load sampler_weights into training client
    # The SFT checkpoint is a sampler checkpoint, not a full state checkpoint
    
    print(f"Creating training client (base: {config.base_model})...")
    training_client = await service_client.create_lora_training_client_async(
        base_model=config.base_model,
        rank=config.lora_rank,
    )
    
    # Note: For proper RL continuation from SFT, we would need:
    # 1. SFT to save with save_state() not save_weights_for_sampler()
    # 2. Then load_state() here
    # For now, we'll use the SFT checkpoint directly for initial sampling
    print(f"SFT checkpoint for reference: {config.sft_checkpoint}")
    print("Note: Using fresh LoRA weights for training, SFT checkpoint for initial sampling")
    
    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.base_model)
    renderer = renderers.get_renderer(name=config.renderer_name, tokenizer=tokenizer)
    stop_sequences = renderer.get_stop_sequences()
    
    print(f"""
RL Training Configuration:
--------------------------
SFT Checkpoint:  {config.sft_checkpoint}
Base Model:      {config.base_model}
LoRA Rank:       {config.lora_rank}
Iterations:      {config.num_iterations}
Batch Size:      {config.batch_size}
Group Size:      {config.group_size}
Learning Rate:   {config.learning_rate:.2e}
Temperature:     {config.temperature}
KL Threshold:    {config.kl_threshold}
""")
    
    metrics_log = []
    final_checkpoint = None
    
    for iteration in range(config.num_iterations):
        iter_start = time.time()
        print(f"\n{'='*70}")
        print(f"Iteration {iteration + 1}/{config.num_iterations}")
        print(f"{'='*70}")
        
        # === STEP 1: Get sampling client ===
        print("\n[1/4] Getting sampling client...")
        
        if iteration == 0:
            # First iteration: use SFT checkpoint
            sampling_path = config.sft_checkpoint
            print(f"  Using SFT checkpoint: {sampling_path}")
        else:
            # Subsequent iterations: save current weights
            save_future = await training_client.save_weights_for_sampler_async(
                name=f"rl_iter_{iteration:04d}"
            )
            save_result = await save_future.result_async()
            sampling_path = save_result.path
            print(f"  Saved new checkpoint: {sampling_path}")
        
        # Create sampling client
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)
        
        # === STEP 2: Generate rollouts ===
        print("[2/4] Generating rollouts...")
        
        # Sample batch of conversations
        batch_indices = np.random.choice(len(train_data), size=config.batch_size, replace=False)
        batch_examples = [train_data[i] for i in batch_indices]
        
        all_rollouts = []
        all_rewards = []
        all_gold_categories = []
        
        for example in batch_examples:
            # Get gold categories
            gold_categories = example.get("categories", [])
            if not gold_categories:
                gold_categories = example.get("labels", {}).get("categories", [])
            
            # Get messages - the data is already in messages format from preprocessing
            messages = example.get("messages", [])
            if not messages:
                continue
            
            # Remove the assistant response if present (we want to generate it)
            prompt_messages = [m for m in messages if m.get("role") != "assistant"]
            
            # Build prompt from the messages (already formatted)
            prompt = renderer.build_generation_prompt(prompt_messages)
            
            # Sample group_size rollouts
            params = types.SamplingParams(
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                stop=stop_sequences
            )
            
            result = sampling_client.sample(
                prompt=prompt,
                sampling_params=params,
                num_samples=config.group_size
            ).result()
            
            # Process each rollout
            for seq in result.sequences:
                response_message, success = renderer.parse_response(seq.tokens)
                predicted_text = response_message["content"] if success else ""
                
                # Compute reward
                reward_result = compute_reward(predicted_text, gold_categories)
                
                # Debug: print first few
                if len(all_rollouts) < 3:
                    print(f"  DEBUG: predicted='{predicted_text}', gold={gold_categories}, reward={reward_result.r_total:.3f}")
                
                all_rollouts.append({
                    "prompt": prompt,
                    "tokens": seq.tokens,
                    "logprobs": seq.logprobs if seq.logprobs else [],
                    "predicted": predicted_text,
                    "gold": gold_categories
                })
                all_rewards.append(reward_result.r_total)
                all_gold_categories.append(gold_categories)
        
        # === STEP 3: Compute advantages ===
        print("[3/4] Computing advantages...")
        
        rewards_array = np.array(all_rewards)
        mean_reward = rewards_array.mean()
        std_reward = rewards_array.std() + 1e-8
        
        # Normalize rewards to get advantages
        advantages = (rewards_array - mean_reward) / std_reward
        
        # === STEP 4: Update model ===
        print("[4/4] Updating model...")
        
        # Build training data
        # Per Tinker losses.mdx: importance_sampling needs target_tokens, logprobs, advantages
        # All arrays must have length N where model_input has length N
        training_data = []
        for i, rollout in enumerate(all_rollouts):
            if not rollout["logprobs"] or len(rollout["logprobs"]) == 0:
                continue
            
            tokens = rollout["tokens"]
            logprobs = rollout["logprobs"]
            advantage = advantages[i]
            
            # Get prompt tokens
            prompt_tokens = rollout["prompt"].to_ints()
            
            # For importance_sampling, per Tinker rl/train.py example:
            # - model_input: the input sequence (prompt + completion[:-1])
            # - target_tokens: what we predict (completion tokens)
            # - logprobs: sampling logprobs for target_tokens
            # - advantages: advantage values for each token
            
            n_gen = len(tokens)
            if n_gen < 1 or len(logprobs) != n_gen:
                continue
            
            # Full input sequence
            full_input = prompt_tokens + tokens[:-1] if n_gen > 1 else prompt_tokens
            n_input = len(full_input)
            
            # Target tokens (shifted by 1 for next-token prediction)
            # We need to include prompt targets too for proper alignment
            full_target = prompt_tokens[1:] + tokens if len(prompt_tokens) > 0 else tokens
            
            # Logprobs: 0 for prompt positions, actual logprobs for completion
            n_prompt = len(prompt_tokens) - 1 if len(prompt_tokens) > 0 else 0
            full_logprobs = [0.0] * n_prompt + logprobs
            
            # Advantages: 0 for prompt, actual advantage for completion
            full_advantages = [0.0] * n_prompt + [advantage] * n_gen
            
            # Verify all lengths match
            if len(full_target) != n_input or len(full_logprobs) != n_input or len(full_advantages) != n_input:
                # Length mismatch, skip
                continue
            
            datum = types.Datum(
                model_input=types.ModelInput.from_ints(full_input),
                loss_fn_inputs=dict(
                    target_tokens=full_target,
                    logprobs=full_logprobs,
                    advantages=full_advantages
                )
            )
            training_data.append(datum)
        
        if training_data:
            # Forward-backward with importance_sampling loss
            fwd_bwd_future = await training_client.forward_backward_async(
                training_data,
                loss_fn="importance_sampling"
            )
            
            # Optimizer step
            adam_params = types.AdamParams(
                learning_rate=config.learning_rate,
                beta1=config.beta1,
                beta2=config.beta2,
                eps=config.eps,
            )
            optim_future = await training_client.optim_step_async(adam_params)
            
            # Wait for results
            fwd_bwd_result = await fwd_bwd_future.result_async()
            optim_result = await optim_future.result_async()
            
            # Extract KL divergence from outputs
            kl_values = []
            for output in fwd_bwd_result.loss_fn_outputs:
                if "logprobs" in output:
                    new_logprobs = output["logprobs"].tolist()
                    # KL approximation
                    kl_values.extend(new_logprobs)
        
        # Compute metrics
        iter_time = time.time() - iter_start
        
        # Category prediction accuracy
        correct = 0
        total = len(all_rollouts)
        for rollout in all_rollouts:
            predicted_set = set([x.strip() for x in rollout["predicted"].split(",") if x.strip() in VALID_CATEGORIES])
            gold_set = set(rollout["gold"])
            if predicted_set.intersection(gold_set):
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        metrics = {
            "iteration": iteration,
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "accuracy": accuracy,
            "num_rollouts": len(all_rollouts),
            "num_training_examples": len(training_data),
            "iter_time": iter_time,
        }
        metrics_log.append(metrics)
        
        print(f"""
Iteration {iteration + 1} Results:
----------------------------------
Mean Reward:     {mean_reward:.4f}
Std Reward:      {std_reward:.4f}
Accuracy:        {accuracy:.2%}
Rollouts:        {len(all_rollouts)}
Training Data:   {len(training_data)}
Time:            {iter_time:.1f}s
""")
        
        # Checkpoint
        if (iteration + 1) % config.checkpoint_every == 0 or iteration == config.num_iterations - 1:
            ckpt_future = await training_client.save_weights_for_sampler_async(
                name=f"rl_final_{iteration:04d}"
            )
            ckpt_result = await ckpt_future.result_async()
            final_checkpoint = ckpt_result.path
            print(f"Checkpoint saved: {final_checkpoint}")
    
    # Save metrics
    metrics_path = os.path.join(config.log_path, "metrics.jsonl")
    with open(metrics_path, "w") as f:
        for m in metrics_log:
            f.write(json.dumps(m) + "\n")
    
    print(f"\n{'='*70}")
    print("RL TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Final checkpoint: {final_checkpoint}")
    print(f"Metrics saved to: {metrics_path}")
    
    return final_checkpoint, metrics_log


async def main():
    import sys
    
    config = RLConfig()
    
    # Parse command line args
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            if hasattr(config, key):
                current_value = getattr(config, key)
                if isinstance(current_value, int):
                    setattr(config, key, int(value))
                elif isinstance(current_value, float):
                    setattr(config, key, float(value))
                else:
                    setattr(config, key, value)
    
    if not config.sft_checkpoint:
        print("ERROR: sft_checkpoint is required")
        print("Usage: python rl_train.py sft_checkpoint=tinker://...")
        sys.exit(1)
    
    await run_rl_training(config)


if __name__ == "__main__":
    asyncio.run(main())

