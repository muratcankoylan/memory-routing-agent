"""
Memory Routing RL Environment

This implements the MemoryRoutingEnv for Stage 2 (RL Optimization) per PRD Section 8.

Per Tinker docs (rl/rl-envs.mdx):
- Env operates on tokens, not strings
- Implement initial_observation() and step()
- EnvGroupBuilder creates groups of environments
- RLDataset provides batches of EnvGroupBuilders

Per PRD Section 4 (Reward Computation):
- R_F1: Token-level F1 between predicted and gold categories
- R_temp: Persistence alignment (+1.0 exact, +0.5 adjacent, 0.0 otherwise)
- R_parity: Company/user scope alignment
- R_eff: Storage efficiency (penalize >3 categories)
- R_total = 0.6*R_F1 + 0.2*R_temp + 0.1*R_parity + 0.1*R_eff

Per PRD Section 4 (Environment Design):
- Single-step bandit: initial_observation returns conversation, step terminates
- EnvGroupBuilder clones each conversation across group_size rollouts
"""

import json
from typing import List, Dict, Any, Tuple, Set, Optional, Sequence
from dataclasses import dataclass

# Memory taxonomy
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

# Persistence mapping
CATEGORY_PERSISTENCE = {
    "company.brand_core": "long",
    "company.strategic_signatures": "long",
    "company.knowledge_artifacts": "long",
    "company.business_priorities": "short",
    "company.tools_config": "medium",
    "company.performance_context": "rolling",
    "user.communication_style": "long",
    "user.strategic_approach": "long",
    "user.role_context": "medium",
    "user.workflow_patterns": "medium",
    "user.session_history": "short",
    "user.interaction_preferences": "evolving",
    "none": "short"
}

# Scope mapping
CATEGORY_SCOPE = {
    "company.brand_core": "company",
    "company.strategic_signatures": "company",
    "company.knowledge_artifacts": "company",
    "company.business_priorities": "company",
    "company.tools_config": "company",
    "company.performance_context": "company",
    "user.communication_style": "user",
    "user.strategic_approach": "user",
    "user.role_context": "user",
    "user.workflow_patterns": "user",
    "user.session_history": "user",
    "user.interaction_preferences": "user",
    "none": "none"
}


@dataclass
class RewardComponents:
    """Breakdown of reward computation."""
    r_f1: float = 0.0
    r_temp: float = 0.0
    r_parity: float = 0.0
    r_eff: float = 0.0
    r_total: float = 0.0
    format_valid: bool = True


def parse_categories(text: str) -> Tuple[Set[str], bool]:
    """
    Parse comma-separated categories from model output.
    
    Returns:
        (set of valid categories, parse_success)
    """
    if not text or not text.strip():
        return set(), False
    
    # Split on comma, strip whitespace, lowercase
    raw_cats = [c.strip().lower() for c in text.split(",")]
    
    # Filter to valid categories
    valid_cats = {c for c in raw_cats if c in VALID_CATEGORIES}
    
    if not valid_cats:
        return set(), False
    
    # Check for invalid "none" mixing
    # Per PRD: "none" must be exclusive
    if "none" in valid_cats and len(valid_cats) > 1:
        valid_cats.discard("none")
    
    return valid_cats, True


def compute_f1(predicted: Set[str], gold: Set[str]) -> float:
    """
    Compute F1 score between predicted and gold category sets.
    
    Per PRD: Use macro-averaging if multi-label.
    """
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    
    true_positives = len(predicted & gold)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(gold) if gold else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def compute_temporal_reward(predicted: Set[str], gold: Set[str]) -> float:
    """
    Compute temporal alignment reward.
    
    Per PRD:
    - +1.0 if predicted persistence matches gold
    - +0.5 if adjacent (long<->medium or medium<->short)
    - 0.0 otherwise
    - Use majority vote if multi-label
    """
    if not predicted or not gold:
        return 0.0
    
    # Get persistence for each category
    pred_persistence = [CATEGORY_PERSISTENCE.get(c, "medium") for c in predicted]
    gold_persistence = [CATEGORY_PERSISTENCE.get(c, "medium") for c in gold]
    
    # Majority vote
    def majority(items):
        from collections import Counter
        if not items:
            return "medium"
        counts = Counter(items)
        return counts.most_common(1)[0][0]
    
    pred_pers = majority(pred_persistence)
    gold_pers = majority(gold_persistence)
    
    # Exact match
    if pred_pers == gold_pers:
        return 1.0
    
    # Adjacent match
    adjacency = {
        ("long", "medium"): True,
        ("medium", "long"): True,
        ("medium", "short"): True,
        ("short", "medium"): True,
        ("medium", "rolling"): True,
        ("rolling", "medium"): True,
        ("short", "rolling"): True,
        ("rolling", "short"): True,
    }
    
    if (pred_pers, gold_pers) in adjacency:
        return 0.5
    
    return 0.0


def compute_parity_reward(predicted: Set[str], gold: Set[str]) -> float:
    """
    Compute company/user scope alignment reward.
    
    Per PRD:
    - +1.0 if predicted scope matches gold scope exactly
    - 0.0 otherwise
    """
    def get_scope(categories: Set[str]) -> str:
        scopes = {CATEGORY_SCOPE.get(c, "none") for c in categories}
        if "company" in scopes and "user" in scopes:
            return "mixed"
        elif "company" in scopes:
            return "company"
        elif "user" in scopes:
            return "user"
        else:
            return "none"
    
    pred_scope = get_scope(predicted)
    gold_scope = get_scope(gold)
    
    return 1.0 if pred_scope == gold_scope else 0.0


def compute_efficiency_reward(predicted: Set[str]) -> float:
    """
    Compute storage efficiency reward.
    
    Per PRD:
    - 1.0 if ≤3 categories
    - 0.7 if 4 categories
    - 0.4 if 5 categories
    - 0.0 if ≥6 categories
    """
    n = len(predicted)
    if n <= 3:
        return 1.0
    elif n == 4:
        return 0.7
    elif n == 5:
        return 0.4
    else:
        return 0.0


def compute_reward(predicted_text: str, gold_categories: List[str]) -> RewardComponents:
    """
    Compute full reward for a prediction.
    
    Per PRD Section 4:
    R_total = 0.6 * R_F1 + 0.2 * R_temp + 0.1 * R_parity + 0.1 * R_eff
    
    Returns RewardComponents with breakdown.
    """
    result = RewardComponents()
    
    # Parse prediction
    predicted, parse_success = parse_categories(predicted_text)
    gold = set(gold_categories)
    
    # Format validation failure
    if not parse_success:
        result.format_valid = False
        result.r_total = -1.0
        return result
    
    # Compute components
    result.r_f1 = compute_f1(predicted, gold)
    result.r_temp = compute_temporal_reward(predicted, gold)
    result.r_parity = compute_parity_reward(predicted, gold)
    result.r_eff = compute_efficiency_reward(predicted)
    
    # Weighted sum
    result.r_total = (
        0.6 * result.r_f1 +
        0.2 * result.r_temp +
        0.1 * result.r_parity +
        0.1 * result.r_eff
    )
    
    return result


# Tinker Environment Classes
# Per Tinker docs (rl/rl-envs.mdx)

class MemoryRoutingEnv:
    """
    Single-step bandit environment for memory routing.
    
    Per Tinker Env interface:
    - initial_observation() -> (Observation, StopCondition)
    - step(action) -> StepResult
    
    Per PRD: Single-step episodes - step() terminates immediately with reward.
    """
    
    def __init__(
        self,
        conversation: List[Dict[str, str]],
        gold_categories: List[str],
        prompt_tokens: List[int],
        stop_tokens: List[int],
        scenario_id: str = ""
    ):
        self.conversation = conversation
        self.gold_categories = gold_categories
        self.prompt_tokens = prompt_tokens
        self.stop_tokens = stop_tokens
        self.scenario_id = scenario_id
        self._done = False
    
    async def initial_observation(self):
        """
        Return the initial observation (prompt tokens) and stop condition.
        
        Per Tinker: Returns (Observation, StopCondition)
        - Observation is the model input (tokens)
        - StopCondition tells the sampler when to stop
        """
        from tinker import types
        from tinker_cookbook.rl.types import StopCondition
        
        observation = types.ModelInput.from_ints(self.prompt_tokens)
        stop_condition = StopCondition(stop_tokens=self.stop_tokens)
        
        return observation, stop_condition
    
    async def step(self, action):
        """
        Process the model's action (generated tokens) and return reward.
        
        Per Tinker: Returns StepResult with reward and done=True
        Per PRD: Single-step bandit, so always terminates.
        """
        from tinker_cookbook.rl.types import StepResult
        
        # Decode action tokens to text
        # Note: In actual implementation, we'd use tokenizer.decode()
        # For now, assume action is already decoded text or we have tokenizer
        if isinstance(action, list):
            # Would decode here: action_text = tokenizer.decode(action)
            action_text = str(action)  # Placeholder
        else:
            action_text = str(action)
        
        # Compute reward
        reward_components = compute_reward(action_text, self.gold_categories)
        
        self._done = True
        
        return StepResult(
            reward=reward_components.r_total,
            done=True,
            info={
                "r_f1": reward_components.r_f1,
                "r_temp": reward_components.r_temp,
                "r_parity": reward_components.r_parity,
                "r_eff": reward_components.r_eff,
                "format_valid": reward_components.format_valid,
                "scenario_id": self.scenario_id
            }
        )


class MemoryRoutingEnvGroupBuilder:
    """
    Builds a group of identical environments for variance reduction.
    
    Per Tinker docs (rl/rl-envs.mdx):
    - EnvGroupBuilder creates group_size copies of the same environment
    - This enables comparing multiple samples for the same input
    """
    
    def __init__(
        self,
        conversation: List[Dict[str, str]],
        gold_categories: List[str],
        prompt_tokens: List[int],
        stop_tokens: List[int],
        group_size: int = 8,
        scenario_id: str = ""
    ):
        self.conversation = conversation
        self.gold_categories = gold_categories
        self.prompt_tokens = prompt_tokens
        self.stop_tokens = stop_tokens
        self.group_size = group_size
        self.scenario_id = scenario_id
    
    async def make_envs(self) -> Sequence["MemoryRoutingEnv"]:
        """Create group_size copies of the environment."""
        return [
            MemoryRoutingEnv(
                conversation=self.conversation,
                gold_categories=self.gold_categories,
                prompt_tokens=self.prompt_tokens,
                stop_tokens=self.stop_tokens,
                scenario_id=self.scenario_id
            )
            for _ in range(self.group_size)
        ]
    
    def logging_tags(self) -> Dict[str, Any]:
        """Return tags for logging."""
        return {
            "scenario_id": self.scenario_id,
            "num_gold_categories": len(self.gold_categories),
            "has_none": "none" in self.gold_categories
        }


class MemoryRoutingDataset:
    """
    Dataset of EnvGroupBuilders for RL training.
    
    Per Tinker docs (rl/rl-envs.mdx):
    - RLDataset.get_batch(index) returns list of EnvGroupBuilders
    """
    
    def __init__(
        self,
        examples: List[Dict[str, Any]],
        batch_size: int,
        group_size: int,
        renderer,
        tokenizer
    ):
        self.examples = examples
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.stop_tokens = renderer.get_stop_sequences()
    
    def __len__(self) -> int:
        return len(self.examples) // self.batch_size
    
    def get_batch(self, index: int) -> List[MemoryRoutingEnvGroupBuilder]:
        """Get a batch of EnvGroupBuilders."""
        start_idx = (index * self.batch_size) % len(self.examples)
        end_idx = start_idx + self.batch_size
        
        if end_idx <= len(self.examples):
            batch_examples = self.examples[start_idx:end_idx]
        else:
            batch_examples = self.examples[start_idx:]
            batch_examples.extend(self.examples[:end_idx - len(self.examples)])
        
        builders = []
        for example in batch_examples:
            # Build prompt for this example
            messages = example.get("messages", [])
            if not messages:
                # Need to construct from conversation
                conversation = example.get("conversation", [])
                categories = example.get("labels", {}).get("categories", [])
                # Build without the assistant response (for generation)
                from training.preprocess import build_routing_prompt
                full_messages = build_routing_prompt(conversation, categories)
                # Remove assistant response for generation prompt
                messages = full_messages[:-1]
            
            # Tokenize prompt
            prompt = self.renderer.build_generation_prompt(messages)
            prompt_tokens = prompt.to_ints()
            
            # Get gold categories
            gold_categories = example.get("categories", [])
            if not gold_categories:
                gold_categories = example.get("labels", {}).get("categories", [])
            
            builders.append(MemoryRoutingEnvGroupBuilder(
                conversation=example.get("conversation", []),
                gold_categories=gold_categories,
                prompt_tokens=prompt_tokens,
                stop_tokens=self.stop_tokens,
                group_size=self.group_size,
                scenario_id=example.get("scenario_id", "")
            ))
        
        return builders


class MemoryRoutingDatasetBuilder:
    """
    Factory for creating train/test RL datasets.
    
    Per Tinker pattern from math_env.py example.
    """
    
    def __init__(
        self,
        train_data_path: str,
        test_data_path: str,
        batch_size: int = 64,
        group_size: int = 8,
        model_name: str = "meta-llama/Llama-3.1-8B",
        renderer_name: str = "llama3"
    ):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.group_size = group_size
        self.model_name = model_name
        self.renderer_name = renderer_name
    
    def __call__(self) -> Tuple[MemoryRoutingDataset, MemoryRoutingDataset]:
        """Create train and test datasets."""
        from tinker_cookbook import renderers, tokenizer_utils
        
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer = renderers.get_renderer(name=self.renderer_name, tokenizer=tokenizer)
        
        # Load data
        with open(self.train_data_path, "r") as f:
            train_examples = json.load(f)
        
        with open(self.test_data_path, "r") as f:
            test_examples = json.load(f)
        
        train_dataset = MemoryRoutingDataset(
            examples=train_examples,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            tokenizer=tokenizer
        )
        
        test_dataset = MemoryRoutingDataset(
            examples=test_examples,
            batch_size=min(self.batch_size, len(test_examples)),
            group_size=self.group_size,
            renderer=renderer,
            tokenizer=tokenizer
        )
        
        return train_dataset, test_dataset


# Test the reward computation
if __name__ == "__main__":
    # Test cases
    test_cases = [
        # (predicted_text, gold_categories, expected_valid)
        ("company.brand_core, user.strategic_approach", ["company.brand_core", "user.strategic_approach"], True),
        ("none", ["none"], True),
        ("company.brand_core, none", ["company.brand_core"], True),  # none should be removed
        ("invalid_category", ["company.brand_core"], False),
        ("", ["company.brand_core"], False),
        ("company.brand_core", ["company.brand_core", "user.role_context"], True),  # Partial match
    ]
    
    print("Testing reward computation:")
    print("=" * 60)
    
    for pred, gold, expected_valid in test_cases:
        result = compute_reward(pred, gold)
        print(f"\nPredicted: '{pred}'")
        print(f"Gold: {gold}")
        print(f"Format valid: {result.format_valid} (expected: {expected_valid})")
        print(f"R_F1: {result.r_f1:.3f}")
        print(f"R_temp: {result.r_temp:.3f}")
        print(f"R_parity: {result.r_parity:.3f}")
        print(f"R_eff: {result.r_eff:.3f}")
        print(f"R_total: {result.r_total:.3f}")

