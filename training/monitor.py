"""
Training Monitor - Check progress and evaluate completed models.
"""

import asyncio
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
import numpy as np

BASE_MODEL = "meta-llama/Llama-3.1-8B"

VALID_CATEGORIES = {
    "company.brand_core", "company.strategic_signatures", "company.knowledge_artifacts",
    "company.business_priorities", "company.tools_config", "company.performance_context",
    "user.communication_style", "user.strategic_approach", "user.role_context",
    "user.workflow_patterns", "user.session_history", "user.interaction_preferences",
    "none"
}


def list_training_runs():
    """List all training runs and their checkpoints."""
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    
    runs = rest_client.list_training_runs().result()
    
    print("=" * 70)
    print("TRAINING RUNS")
    print("=" * 70)
    
    for run in runs.training_runs[:10]:
        ckpts = rest_client.list_checkpoints(run.training_run_id).result()
        
        # Categorize checkpoints
        sft_ckpts = [c for c in ckpts.checkpoints if 'sft' in c.checkpoint_id]
        rl_ckpts = [c for c in ckpts.checkpoints if 'rl_' in c.checkpoint_id]
        
        print(f"\nRun: {run.training_run_id}")
        print(f"  Last request: {run.last_request_time}")
        print(f"  SFT checkpoints: {len(sft_ckpts)}")
        print(f"  RL checkpoints: {len(rl_ckpts)}")
        
        if rl_ckpts:
            # Find the latest RL checkpoint
            latest = sorted(rl_ckpts, key=lambda x: x.time)[-1]
            print(f"  Latest RL: {latest.checkpoint_id}")
            
            # Check if it's a final checkpoint
            if 'final' in latest.checkpoint_id:
                print(f"  STATUS: RL COMPLETE")
                print(f"  Final checkpoint: tinker://{run.training_run_id}/{latest.checkpoint_id}")


async def quick_eval(checkpoint_path: str, n_samples: int = 20):
    """Quick evaluation of a checkpoint."""
    service_client = tinker.ServiceClient()
    tokenizer = get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(name="llama3", tokenizer=tokenizer)
    
    # Load test data
    with open("training/processed_data/test_data.json", "r") as f:
        test_data = json.load(f)
    
    print(f"\nEvaluating: {checkpoint_path}")
    print(f"Samples: {n_samples}")
    
    sampling_client = service_client.create_sampling_client(model_path=checkpoint_path)
    stop_sequences = renderer.get_stop_sequences()
    
    correct = 0
    total = 0
    
    for example in test_data[:n_samples]:
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
        
        if predicted_set & gold_set:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Any Match Accuracy: {accuracy:.1%} ({correct}/{total})")
    
    return accuracy


def find_best_checkpoint():
    """Find the best completed RL checkpoint."""
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    
    runs = rest_client.list_training_runs().result()
    
    best_rl_checkpoint = None
    best_sft_checkpoint = None
    
    for run in runs.training_runs:
        ckpts = rest_client.list_checkpoints(run.training_run_id).result()
        
        for ckpt in ckpts.checkpoints:
            if 'rl_final' in ckpt.checkpoint_id:
                path = f"tinker://{run.training_run_id}/{ckpt.checkpoint_id}"
                if best_rl_checkpoint is None or ckpt.time > best_rl_checkpoint[1]:
                    best_rl_checkpoint = (path, ckpt.time)
            
            if 'sft_final_sampler' in ckpt.checkpoint_id:
                path = f"tinker://{run.training_run_id}/{ckpt.checkpoint_id}"
                if best_sft_checkpoint is None or ckpt.time > best_sft_checkpoint[1]:
                    best_sft_checkpoint = (path, ckpt.time)
    
    return best_sft_checkpoint, best_rl_checkpoint


async def main():
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        # Evaluate mode
        sft_ckpt, rl_ckpt = find_best_checkpoint()
        
        print("=" * 70)
        print("CHECKPOINT EVALUATION")
        print("=" * 70)
        
        if sft_ckpt:
            print(f"\nBest SFT: {sft_ckpt[0]}")
            await quick_eval(sft_ckpt[0], n_samples=50)
        
        if rl_ckpt:
            print(f"\nBest RL: {rl_ckpt[0]}")
            await quick_eval(rl_ckpt[0], n_samples=50)
    else:
        # List mode
        list_training_runs()
        
        sft_ckpt, rl_ckpt = find_best_checkpoint()
        
        print("\n" + "=" * 70)
        print("BEST CHECKPOINTS")
        print("=" * 70)
        
        if sft_ckpt:
            print(f"\nSFT: {sft_ckpt[0]}")
            print(f"     Time: {sft_ckpt[1]}")
        
        if rl_ckpt:
            print(f"\nRL:  {rl_ckpt[0]}")
            print(f"     Time: {rl_ckpt[1]}")
        
        print("\nTo evaluate, run: python training/monitor.py eval")


if __name__ == "__main__":
    asyncio.run(main())

