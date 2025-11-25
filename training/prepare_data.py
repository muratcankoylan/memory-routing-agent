"""
Prepare training data by merging datasets and preprocessing for Tinker.

This script:
1. Merges the original dataset with the new diverse dataset
2. Validates and cleans the data
3. Converts to the format expected by train_v2.py
4. Splits into train/test sets
5. Analyzes category distribution
"""

import json
import os
from collections import Counter
from typing import List, Dict, Any
import random

# Paths
ORIGINAL_DATASET = "synthetic_data/training_dataset_1000.jsonl"
DIVERSE_DATASET = "synthetic_data/diverse_dataset_20251124_192207.jsonl"
OUTPUT_DIR = "training/processed_data"
TRAIN_OUTPUT = os.path.join(OUTPUT_DIR, "train_data.json")
TEST_OUTPUT = os.path.join(OUTPUT_DIR, "test_data.json")

# System prompt for memory routing
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

VALID_CATEGORIES = {
    "company.brand_core", "company.strategic_signatures", "company.knowledge_artifacts",
    "company.business_priorities", "company.tools_config", "company.performance_context",
    "user.communication_style", "user.strategic_approach", "user.role_context",
    "user.workflow_patterns", "user.session_history", "user.interaction_preferences",
    "none"
}


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    return data


def clean_categories(categories: List[str]) -> List[str]:
    """Clean and validate categories."""
    cleaned = []
    for cat in categories:
        cat_lower = cat.strip().lower()
        if cat_lower in VALID_CATEGORIES:
            cleaned.append(cat_lower)
    
    # Remove "none" if other categories exist
    if len(cleaned) > 1 and "none" in cleaned:
        cleaned = [c for c in cleaned if c != "none"]
    
    # Deduplicate while preserving order
    seen = set()
    result = []
    for c in cleaned:
        if c not in seen:
            seen.add(c)
            result.append(c)
    
    return result if result else ["none"]


def convert_to_training_format(item: Dict) -> Dict:
    """
    Convert a synthetic data item to the training format.
    
    Output format:
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "...conversation..."},
            {"role": "assistant", "content": "category1, category2"}
        ],
        "categories": ["category1", "category2"],
        "scenario_id": "...",
        "metadata": {...}
    }
    """
    # Get conversation
    conversation = item.get("conversation", [])
    if not conversation:
        return None
    
    # Build conversation text
    conv_text = ""
    for turn in conversation:
        if isinstance(turn, dict):
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            conv_text += f"{role.upper()}: {content}\n"
        elif isinstance(turn, str):
            conv_text += f"{turn}\n"
    
    if not conv_text.strip():
        return None
    
    # Get categories
    categories = item.get("labels", {}).get("categories", [])
    if not categories:
        categories = [item.get("metadata", {}).get("primary_category", "none")]
    
    categories = clean_categories(categories)
    
    # Build messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze this conversation and determine which memory categories apply:\n\n{conv_text.strip()}"},
        {"role": "assistant", "content": ", ".join(categories)}
    ]
    
    return {
        "messages": messages,
        "categories": categories,
        "scenario_id": item.get("scenario_id", ""),
        "metadata": item.get("metadata", {})
    }


def analyze_distribution(data: List[Dict]) -> Dict[str, int]:
    """Analyze category distribution."""
    counter = Counter()
    for item in data:
        for cat in item.get("categories", []):
            counter[cat] += 1
    return dict(counter)


def main():
    print("=" * 70)
    print("PREPARING TRAINING DATA")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load datasets
    print(f"\nLoading original dataset: {ORIGINAL_DATASET}")
    original_data = load_jsonl(ORIGINAL_DATASET)
    print(f"  Loaded {len(original_data)} items")
    
    print(f"\nLoading diverse dataset: {DIVERSE_DATASET}")
    diverse_data = load_jsonl(DIVERSE_DATASET)
    print(f"  Loaded {len(diverse_data)} items")
    
    # Convert to training format
    print("\nConverting to training format...")
    
    all_data = []
    skipped = 0
    
    for item in original_data:
        converted = convert_to_training_format(item)
        if converted:
            converted["source"] = "original"
            all_data.append(converted)
        else:
            skipped += 1
    
    for item in diverse_data:
        converted = convert_to_training_format(item)
        if converted:
            converted["source"] = "diverse"
            all_data.append(converted)
        else:
            skipped += 1
    
    print(f"  Converted: {len(all_data)}")
    print(f"  Skipped: {skipped}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_data)
    
    # Split train/test (90/10)
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_data)}")
    print(f"  Test: {len(test_data)}")
    
    # Analyze distribution
    print("\n" + "-" * 50)
    print("CATEGORY DISTRIBUTION (Train)")
    print("-" * 50)
    
    train_dist = analyze_distribution(train_data)
    total = sum(train_dist.values())
    
    for cat in sorted(train_dist.keys()):
        count = train_dist[cat]
        pct = count / total * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"{cat:<35} {count:>4} ({pct:>5.1f}%) {bar[:30]}")
    
    print(f"\nTotal labels: {total}")
    print(f"Unique categories: {len(train_dist)}")
    
    # Check balance
    min_count = min(train_dist.values())
    max_count = max(train_dist.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"\nImbalance ratio: {imbalance_ratio:.1f}x (max/min)")
    
    if imbalance_ratio < 3:
        print("  Status: GOOD - Dataset is reasonably balanced")
    elif imbalance_ratio < 5:
        print("  Status: OK - Some imbalance but acceptable")
    else:
        print("  Status: WARNING - Dataset is imbalanced")
    
    # Save
    print(f"\nSaving to {OUTPUT_DIR}/...")
    
    with open(TRAIN_OUTPUT, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"  Saved train_data.json ({len(train_data)} items)")
    
    with open(TEST_OUTPUT, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"  Saved test_data.json ({len(test_data)} items)")
    
    # Summary
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)
    print(f"Train: {TRAIN_OUTPUT}")
    print(f"Test: {TEST_OUTPUT}")
    print(f"\nReady for training with train_v2.py")


if __name__ == "__main__":
    main()

