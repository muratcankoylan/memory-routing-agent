"""
Supervised Fine-Tuning (SFT) for Memory Routing

This implements Stage 1 of the PRD: Prompt Distillation using Tinker's
cross_entropy loss function with LoRA fine-tuning.

Per Tinker docs (supervised-learning.mdx):
- SFT means maximizing log-probability of target tokens
- Use cross_entropy loss: -(weights * logp(target_tokens)).sum()

Per Tinker docs (lora-primer.mdx):
- LoRA requires larger LR than full fine-tuning (20-100x)
- Use get_lr() utility to get recommended LR
- Default rank 32 is suitable for classification tasks

Per Tinker docs (async.mdx):
- Use async methods for performance
- Double await pattern: await future, then await result_async()

Per PRD Section 7:
- 300-500 steps minimum
- Batch size 128
- Early stopping if test loss plateaus
- Checkpoint every 20 steps
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Configuration
@dataclass
class SFTConfig:
    # Model
    base_model: str = "meta-llama/Llama-3.1-8B"
    lora_rank: int = 32
    renderer_name: str = "llama3"
    
    # Training
    num_steps: int = 300
    batch_size: int = 128
    learning_rate: Optional[float] = None  # Will use get_lr() if None
    
    # Adam optimizer (per Tinker defaults)
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Checkpointing
    checkpoint_every: int = 20
    eval_every: int = 20
    
    # Early stopping
    early_stopping_patience: int = 5  # Stop if no improvement for this many evals
    
    # Paths
    train_data_path: str = "training/processed_data/train_data.json"
    test_data_path: str = "training/processed_data/test_data.json"
    log_path: str = "training/logs/sft"


@dataclass
class TrainingMetrics:
    step: int
    train_loss: float
    test_loss: Optional[float] = None
    learning_rate: float = 0.0
    batch_time: float = 0.0
    checkpoint_path: Optional[str] = None


def load_processed_data(path: str) -> List[Dict[str, Any]]:
    """Load preprocessed data from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def create_batch(data: List[Any], batch_size: int, step: int) -> List[Any]:
    """
    Create a batch of data for training.
    Cycles through data if step * batch_size exceeds data length.
    """
    start_idx = (step * batch_size) % len(data)
    end_idx = start_idx + batch_size
    
    if end_idx <= len(data):
        return data[start_idx:end_idx]
    else:
        # Wrap around
        batch = data[start_idx:]
        batch.extend(data[:end_idx - len(data)])
        return batch


async def run_sft_training(config: SFTConfig):
    """
    Main SFT training loop.
    
    Per Tinker docs (training-sampling.mdx):
    1. Create ServiceClient
    2. Create TrainingClient with base_model and LoRA config
    3. Loop: forward_backward -> optim_step
    4. Periodically save checkpoints and evaluate
    """
    import tinker
    from tinker import types
    from tinker_cookbook.hyperparam_utils import get_lr
    from tinker_cookbook import renderers, tokenizer_utils
    import numpy as np
    import os
    from dotenv import load_dotenv
    
    # Load API key from .env
    load_dotenv()
    
    os.makedirs(config.log_path, exist_ok=True)
    
    # Get learning rate if not specified
    if config.learning_rate is None:
        config.learning_rate = get_lr(config.base_model)
        print(f"Using recommended LR for {config.base_model}: {config.learning_rate:.2e}")
    
    # Load data
    print(f"Loading training data from {config.train_data_path}...")
    train_data_raw = load_processed_data(config.train_data_path)
    print(f"Loading test data from {config.test_data_path}...")
    test_data_raw = load_processed_data(config.test_data_path)
    
    print(f"Train examples: {len(train_data_raw)}")
    print(f"Test examples: {len(test_data_raw)}")
    
    # Initialize Tinker clients
    print(f"\nInitializing Tinker ServiceClient...")
    service_client = tinker.ServiceClient()
    
    print(f"Creating LoRA training client...")
    print(f"  Base model: {config.base_model}")
    print(f"  LoRA rank: {config.lora_rank}")
    
    training_client = await service_client.create_lora_training_client_async(
        base_model=config.base_model,
        rank=config.lora_rank,
    )
    
    # Get tokenizer from training client (avoids HF auth issues)
    tokenizer = training_client.get_tokenizer()
    renderer = renderers.get_renderer(name=config.renderer_name, tokenizer=tokenizer)
    
    # Convert raw data to Datum objects
    print("Converting data to Datum objects...")
    
    def convert_to_datum(item: Dict) -> types.Datum:
        """Convert preprocessed item back to Datum."""
        if "model_input" in item:
            # Already in Datum format
            return types.Datum(
                model_input=types.ModelInput.from_ints(item["model_input"]["chunks"][0]["tokens"]),
                loss_fn_inputs=item["loss_fn_inputs"]
            )
        else:
            # Mock format - need to re-tokenize
            messages = item["messages"]
            tokens, weights = renderer.build_supervised_example(messages)
            
            # Convert tensors to lists if needed
            if hasattr(tokens, 'tolist'):
                tokens = tokens.tolist()
            if hasattr(weights, 'tolist'):
                weights = weights.tolist()
            
            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]
            loss_weights = weights[1:]
            
            return types.Datum(
                model_input=types.ModelInput.from_ints(input_tokens),
                loss_fn_inputs=dict(
                    target_tokens=target_tokens,
                    weights=loss_weights
                )
            )
    
    train_data = [convert_to_datum(item) for item in train_data_raw]
    test_data = [convert_to_datum(item) for item in test_data_raw]
    
    print(f"Converted {len(train_data)} train, {len(test_data)} test examples")
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting SFT Training")
    print(f"{'='*60}")
    print(f"Steps: {config.num_steps}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate:.2e}")
    print(f"Checkpoint every: {config.checkpoint_every} steps")
    print(f"Eval every: {config.eval_every} steps")
    print(f"{'='*60}\n")
    
    metrics_log = []
    best_test_loss = float('inf')
    no_improvement_count = 0
    final_checkpoint_path = None
    
    for step in range(config.num_steps):
        step_start = time.time()
        
        # Create batch
        batch = create_batch(train_data, config.batch_size, step)
        
        # Forward-backward pass
        # Per Tinker docs: submit forward_backward, then optim_step
        # Can overlap by submitting both before waiting
        fwd_bwd_future = await training_client.forward_backward_async(
            batch,
            loss_fn="cross_entropy"
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
        # Per Tinker async.mdx: must await result_async() to get actual values
        fwd_bwd_result = await fwd_bwd_future.result_async()
        optim_result = await optim_future.result_async()
        
        # Compute train loss
        # Per Tinker losses.mdx: cross_entropy outputs logprobs
        logprobs = np.concatenate([
            output['logprobs'].tolist() 
            for output in fwd_bwd_result.loss_fn_outputs
        ])
        weights = np.concatenate([
            datum.loss_fn_inputs['weights'].tolist() 
            for datum in batch
        ])
        train_loss = -np.dot(logprobs, weights) / max(weights.sum(), 1)
        
        step_time = time.time() - step_start
        
        # Create metrics
        metrics = TrainingMetrics(
            step=step,
            train_loss=train_loss,
            learning_rate=config.learning_rate,
            batch_time=step_time
        )
        
        # Periodic evaluation
        if step % config.eval_every == 0 or step == config.num_steps - 1:
            # Evaluate on test set (sample a batch)
            test_batch = create_batch(test_data, min(config.batch_size, len(test_data)), 0)
            
            # Forward only (no backward) for evaluation
            eval_future = await training_client.forward_backward_async(
                test_batch,
                loss_fn="cross_entropy"
            )
            eval_result = await eval_future.result_async()
            
            test_logprobs = np.concatenate([
                output['logprobs'].tolist() 
                for output in eval_result.loss_fn_outputs
            ])
            test_weights = np.concatenate([
                datum.loss_fn_inputs['weights'].tolist() 
                for datum in test_batch
            ])
            test_loss = -np.dot(test_logprobs, test_weights) / max(test_weights.sum(), 1)
            metrics.test_loss = test_loss
            
            # Early stopping check
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= config.early_stopping_patience:
                print(f"\nEarly stopping at step {step} (no improvement for {config.early_stopping_patience} evals)")
                break
        
        # Periodic checkpointing
        if step % config.checkpoint_every == 0 or step == config.num_steps - 1:
            # Save both sampler weights (for inference) and full state (for RL continuation)
            # Per Tinker save-load.mdx: save_state for resuming training
            
            # Sampler weights for inference
            sampler_future = await training_client.save_weights_for_sampler_async(
                name=f"sft_step_{step:04d}"
            )
            sampler_result = await sampler_future.result_async()
            metrics.checkpoint_path = sampler_result.path
            
            # Full state for RL continuation (only at final step to save storage)
            if step == config.num_steps - 1:
                state_future = await training_client.save_state_async(
                    name=f"sft_final_state"
                )
                state_result = await state_future.result_async()
                final_checkpoint_path = state_result.path
                print(f"  Full state checkpoint: {final_checkpoint_path}")
            else:
                final_checkpoint_path = sampler_result.path
        
        metrics_log.append(metrics)
        
        # Print progress
        test_str = f", test_loss={metrics.test_loss:.4f}" if metrics.test_loss else ""
        ckpt_str = f", checkpoint={metrics.checkpoint_path}" if metrics.checkpoint_path else ""
        print(f"Step {step:4d}/{config.num_steps}: train_loss={train_loss:.4f}{test_str}, time={step_time:.1f}s{ckpt_str}")
    
    # Save metrics log
    metrics_path = os.path.join(config.log_path, "metrics.jsonl")
    with open(metrics_path, "w") as f:
        for m in metrics_log:
            f.write(json.dumps({
                "step": m.step,
                "train_loss": m.train_loss,
                "test_loss": m.test_loss,
                "learning_rate": m.learning_rate,
                "batch_time": m.batch_time,
                "checkpoint_path": m.checkpoint_path
            }) + "\n")
    
    print(f"\n{'='*60}")
    print(f"SFT Training Complete")
    print(f"{'='*60}")
    print(f"Final train loss: {metrics_log[-1].train_loss:.4f}")
    print(f"Best test loss: {best_test_loss:.4f}")
    print(f"Final checkpoint: {final_checkpoint_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"{'='*60}")
    
    return final_checkpoint_path, metrics_log


async def main():
    """Entry point for SFT training."""
    import sys
    
    config = SFTConfig()
    
    # Parse command line args
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            if hasattr(config, key):
                # Type conversion
                current_value = getattr(config, key)
                if isinstance(current_value, int):
                    setattr(config, key, int(value))
                elif isinstance(current_value, float):
                    setattr(config, key, float(value))
                else:
                    setattr(config, key, value)
    
    await run_sft_training(config)


if __name__ == "__main__":
    asyncio.run(main())

