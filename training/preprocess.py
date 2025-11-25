"""
Data Preprocessing for Memory Routing Training

This script converts synthetic JSONL conversations to Tinker-compatible
types.Datum objects for supervised fine-tuning.

Per Tinker docs (rendering.mdx):
- Use renderer.build_supervised_example() to get tokens and weights
- Weights indicate which tokens to train on (1.0 for completion, 0.0 for prompt)
- Target tokens are shifted by 1 (predicting next token)

Per PRD Section 6.6:
- Validate datum length <= 4096
- Ensure non-zero weights
- Verify token IDs are within vocab range
"""

import json
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Note: These imports require tinker and tinker-cookbook to be installed
# pip install git+https://github.com/thinking-machines-lab/tinker.git
# pip install git+https://github.com/thinking-machines-lab/tinker-cookbook.git

MODEL_NAME = "meta-llama/Llama-3.1-8B"
RENDERER_NAME = "llama3"
MAX_SEQUENCE_LENGTH = 4096

# Memory taxonomy for validation
VALID_CATEGORIES = {
    "company.brand_core",
    "company.strategic_signatures", 
    "company.knowledge_artifacts",
    "company.business_priorities",
    "company.tools_config",
    "company.performance_context",
    "user.communication_style",
    "user.strategic_approach",
    "user.role_context",
    "user.workflow_patterns",
    "user.session_history",
    "user.interaction_preferences",
    "none"
}

@dataclass
class PreprocessingStats:
    total_examples: int = 0
    valid_examples: int = 0
    skipped_too_long: int = 0
    skipped_zero_weights: int = 0
    skipped_invalid_tokens: int = 0
    skipped_invalid_categories: int = 0


def build_routing_prompt(conversation: List[Dict[str, str]], categories: List[str]) -> List[Dict[str, str]]:
    """
    Build the full conversation for training, including:
    1. System prompt with taxonomy
    2. User message with conversation
    3. Assistant response with categories
    
    Per PRD Section 6 - Student Prompt format.
    """
    # System prompt with taxonomy
    system_content = """You route marketing conversations into structured memory categories.

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

    # Format the conversation for the user message
    conversation_text = ""
    for turn in conversation:
        # Handle malformed turns (string instead of dict)
        if isinstance(turn, str):
            conversation_text += f"UNKNOWN: {turn}\n"
            continue
        if not isinstance(turn, dict):
            continue
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        conversation_text += f"{role.upper()}: {content}\n"
    
    user_content = f"Conversation:\n{conversation_text.strip()}\n\nWhat memory categories apply?"
    
    # Assistant response is the comma-separated categories
    assistant_content = ", ".join(categories)
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]


def load_synthetic_data(filepath: str) -> List[Dict[str, Any]]:
    """Load synthetic data from JSONL file."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append(item)
    return data


def validate_categories(categories: List[str]) -> bool:
    """Validate that all categories are in the taxonomy."""
    return all(cat in VALID_CATEGORIES for cat in categories)


def preprocess_example_mock(example: Dict[str, Any], stats: PreprocessingStats) -> Dict[str, Any] | None:
    """
    Mock preprocessing that validates structure without Tinker.
    Returns a dict representation of what would become a Datum.
    
    Use this for testing without Tinker installed.
    """
    conversation = example.get("conversation", [])
    labels = example.get("labels", {})
    categories = labels.get("categories", [])
    
    # Validate categories
    if not validate_categories(categories):
        stats.skipped_invalid_categories += 1
        return None
    
    # Build the full training conversation
    training_messages = build_routing_prompt(conversation, categories)
    
    # Mock token estimation (rough: 4 chars per token)
    total_chars = sum(len(m["content"]) for m in training_messages)
    estimated_tokens = total_chars // 4
    
    if estimated_tokens > MAX_SEQUENCE_LENGTH:
        stats.skipped_too_long += 1
        return None
    
    stats.valid_examples += 1
    
    return {
        "messages": training_messages,
        "categories": categories,
        "estimated_tokens": estimated_tokens,
        "scenario_id": example.get("scenario_id", "unknown")
    }


def preprocess_with_tinker(example: Dict[str, Any], renderer, tokenizer, vocab_size: int, stats: PreprocessingStats):
    """
    Full preprocessing with Tinker renderer.
    
    Per Tinker docs (rendering.mdx):
    - build_supervised_example returns (tokens, weights)
    - weights=1.0 for completion tokens, weights=0.0 for prompt tokens
    
    Per Tinker docs (training-sampling.mdx):
    - input_tokens = tokens[:-1]
    - target_tokens = tokens[1:]  # Shifted for next-token prediction
    - weights = weights[1:]
    """
    from tinker import types
    
    conversation = example.get("conversation", [])
    labels = example.get("labels", {})
    categories = labels.get("categories", [])
    
    # Validate categories
    if not validate_categories(categories):
        stats.skipped_invalid_categories += 1
        return None
    
    # Build the full training conversation
    training_messages = build_routing_prompt(conversation, categories)
    
    # Use renderer to tokenize and get weights
    # Per Tinker rendering.mdx: build_supervised_example returns tokens and weights
    tokens, weights = renderer.build_supervised_example(training_messages)
    
    # Check sequence length
    if len(tokens) > MAX_SEQUENCE_LENGTH:
        stats.skipped_too_long += 1
        return None
    
    # Prepare for next-token prediction
    # Per Tinker training-sampling.mdx example
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    loss_weights = weights[1:]
    
    # Validate non-zero weights
    if sum(loss_weights) == 0:
        stats.skipped_zero_weights += 1
        return None
    
    # Validate token IDs
    if not all(0 <= t < vocab_size for t in target_tokens):
        stats.skipped_invalid_tokens += 1
        return None
    
    # Create Datum object
    # Per Tinker types (Datum class)
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(input_tokens),
        loss_fn_inputs=dict(
            target_tokens=target_tokens,
            weights=loss_weights
        )
    )
    
    stats.valid_examples += 1
    return datum


def preprocess_dataset(
    input_path: str,
    output_dir: str,
    use_tinker: bool = False,
    train_split: float = 0.8
) -> Tuple[PreprocessingStats, str, str]:
    """
    Preprocess the full dataset.
    
    Args:
        input_path: Path to training_dataset_1000.jsonl
        output_dir: Directory to save processed data
        use_tinker: Whether to use actual Tinker (requires installation)
        train_split: Fraction for training (rest is test)
    
    Returns:
        stats, train_path, test_path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {input_path}...")
    raw_data = load_synthetic_data(input_path)
    print(f"Loaded {len(raw_data)} examples")
    
    stats = PreprocessingStats(total_examples=len(raw_data))
    
    if use_tinker:
        # Import Tinker components
        from tinker_cookbook import renderers, tokenizer_utils
        
        print(f"Initializing tokenizer for {MODEL_NAME}...")
        tokenizer = tokenizer_utils.get_tokenizer(MODEL_NAME)
        renderer = renderers.get_renderer(name=RENDERER_NAME, tokenizer=tokenizer)
        vocab_size = len(tokenizer)
        print(f"Vocab size: {vocab_size}")
        
        processed_data = []
        for i, example in enumerate(raw_data):
            if i % 100 == 0:
                print(f"Processing {i}/{len(raw_data)}...")
            datum = preprocess_with_tinker(example, renderer, tokenizer, vocab_size, stats)
            if datum is not None:
                processed_data.append(datum)
    else:
        # Mock preprocessing for testing
        print("Running mock preprocessing (no Tinker)...")
        processed_data = []
        for i, example in enumerate(raw_data):
            if i % 100 == 0:
                print(f"Processing {i}/{len(raw_data)}...")
            result = preprocess_example_mock(example, stats)
            if result is not None:
                processed_data.append(result)
    
    # Split into train/test
    split_idx = int(len(processed_data) * train_split)
    train_data = processed_data[:split_idx]
    test_data = processed_data[split_idx:]
    
    # Save processed data
    train_path = os.path.join(output_dir, "train_data.json")
    test_path = os.path.join(output_dir, "test_data.json")
    
    with open(train_path, "w") as f:
        json.dump([d if isinstance(d, dict) else d.model_dump() for d in train_data], f)
    
    with open(test_path, "w") as f:
        json.dump([d if isinstance(d, dict) else d.model_dump() for d in test_data], f)
    
    print(f"\n=== Preprocessing Complete ===")
    print(f"Total examples: {stats.total_examples}")
    print(f"Valid examples: {stats.valid_examples}")
    print(f"Skipped (too long): {stats.skipped_too_long}")
    print(f"Skipped (zero weights): {stats.skipped_zero_weights}")
    print(f"Skipped (invalid tokens): {stats.skipped_invalid_tokens}")
    print(f"Skipped (invalid categories): {stats.skipped_invalid_categories}")
    print(f"\nTrain set: {len(train_data)} examples")
    print(f"Test set: {len(test_data)} examples")
    print(f"\nSaved to:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")
    
    return stats, train_path, test_path


if __name__ == "__main__":
    import sys
    
    input_path = sys.argv[1] if len(sys.argv) > 1 else "synthetic_data/training_dataset_1000.jsonl"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "training/processed_data"
    use_tinker = "--tinker" in sys.argv
    
    preprocess_dataset(input_path, output_dir, use_tinker=use_tinker)

