"""
Upload Memory Routing Agent to HuggingFace Hub

This script uploads:
1. Model card (README.md)
2. Training dataset
3. Benchmark dataset
4. Training configuration
"""

import os
import json
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder, login

load_dotenv()

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "muratcankoylan/memory-routing-agent"
DATASET_REPO_ID = "muratcankoylan/memory-routing-marketing"

# Login first
if HF_TOKEN:
    print(f"Logging in with token (first 10 chars): {HF_TOKEN[:10]}...")
    login(token=HF_TOKEN)
else:
    print("ERROR: HF_TOKEN not found in .env file")
    exit(1)

def upload_model():
    """Upload model card and metadata to HuggingFace."""
    api = HfApi(token=HF_TOKEN)
    
    # Create model repo
    try:
        create_repo(repo_id=REPO_ID, token=HF_TOKEN, exist_ok=True)
        print(f"Created/verified repo: {REPO_ID}")
    except Exception as e:
        print(f"Repo creation note: {e}")
    
    # Upload README (model card)
    upload_file(
        path_or_fileobj="huggingface/README.md",
        path_in_repo="README.md",
        repo_id=REPO_ID,
        token=HF_TOKEN,
    )
    print("Uploaded model card")
    
    # Upload benchmark
    upload_file(
        path_or_fileobj="training/benchmarks/marketing_routing_benchmark.json",
        path_in_repo="benchmark/marketing_routing_benchmark.json",
        repo_id=REPO_ID,
        token=HF_TOKEN,
    )
    print("Uploaded benchmark")
    
    # Upload training config
    config = {
        "base_model": "meta-llama/Llama-3.1-8B",
        "lora_rank": 32,
        "sft_steps": 100,
        "sft_batch_size": 128,
        "sft_learning_rate": 2.86e-4,
        "rl_iterations": 12,
        "rl_groups_per_batch": 64,
        "rl_group_size": 32,
        "rl_learning_rate": 2e-5,
        "tinker_checkpoint": "tinker://4f4bae1f-5a95-5f53-a55a-a14f2872825c:train:0/sampler_weights/rl_iter_012",
        "reward_weights": {
            "f1": 0.6,
            "temporal": 0.2,
            "parity": 0.1,
            "efficiency": 0.1
        }
    }
    
    with open("huggingface/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    upload_file(
        path_or_fileobj="huggingface/config.json",
        path_in_repo="config.json",
        repo_id=REPO_ID,
        token=HF_TOKEN,
    )
    print("Uploaded config")
    
    print(f"\nModel uploaded to: https://huggingface.co/{REPO_ID}")


def upload_dataset():
    """Upload training dataset to HuggingFace Datasets."""
    api = HfApi(token=HF_TOKEN)
    
    # Create dataset repo
    try:
        create_repo(repo_id=DATASET_REPO_ID, token=HF_TOKEN, repo_type="dataset", exist_ok=True)
        print(f"Created/verified dataset repo: {DATASET_REPO_ID}")
    except Exception as e:
        print(f"Dataset repo creation note: {e}")
    
    # Create dataset README
    dataset_readme = """---
license: apache-2.0
language:
- en
tags:
- memory-routing
- marketing
- classification
- synthetic
size_categories:
- 1K<n<10K
---

# Memory Routing Marketing Dataset

2,001 synthetic marketing conversations for training memory routing classifiers.

## Dataset Description

This dataset contains marketing conversations labeled with memory categories. Each conversation includes:
- Multi-turn dialogue between a user and AI assistant
- Category labels (13 possible categories)
- Persistence horizon (long/medium/short)
- Memory scope (company/user/none)

## Categories

### Company Categories
- `company.brand_core` - Voice, values, positioning
- `company.strategic_signatures` - Decision frameworks
- `company.knowledge_artifacts` - Docs, style guides
- `company.business_priorities` - Quarterly goals
- `company.tools_config` - Integrations, APIs
- `company.performance_context` - Campaign metrics

### User Categories
- `user.communication_style` - Tone, format preferences
- `user.strategic_approach` - Personal priorities
- `user.role_context` - Title, scope
- `user.workflow_patterns` - Review cadence
- `user.session_history` - Immediate context
- `user.interaction_preferences` - Coaching style

### Special
- `none` - Transactional or irrelevant content

## Generation

Generated using Cohere Command-R-Plus (104B) as teacher model with diverse prompts covering:
- Multiple industries (tech, retail, healthcare, finance, etc.)
- Various user roles (CMO, VP Marketing, Growth Lead, etc.)
- Different conversation styles and complexities

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("muratcankoylan/memory-routing-marketing")
```

## Citation

```bibtex
@misc{memory-routing-agent-2024,
  title={Memory Routing Agent: Prompt Distillation for Marketing AI},
  author={Muratcan Koylan},
  year={2024},
  howpublished={\\url{https://github.com/muratcankoylan/memory-routing-agent}},
}
```
"""
    
    with open("huggingface/dataset_readme.md", "w") as f:
        f.write(dataset_readme)
    
    upload_file(
        path_or_fileobj="huggingface/dataset_readme.md",
        path_in_repo="README.md",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print("Uploaded dataset README")
    
    # Upload training data
    upload_file(
        path_or_fileobj="synthetic_data/merged_training_dataset_2001.jsonl",
        path_in_repo="data/train.jsonl",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print("Uploaded training data")
    
    print(f"\nDataset uploaded to: https://huggingface.co/datasets/{DATASET_REPO_ID}")


if __name__ == "__main__":
    print("=" * 60)
    print("Uploading Memory Routing Agent to HuggingFace")
    print("=" * 60)
    
    print("\n1. Uploading model...")
    upload_model()
    
    print("\n2. Uploading dataset...")
    upload_dataset()
    
    print("\n" + "=" * 60)
    print("UPLOAD COMPLETE")
    print("=" * 60)
    print(f"Model: https://huggingface.co/{REPO_ID}")
    print(f"Dataset: https://huggingface.co/datasets/{DATASET_REPO_ID}")

