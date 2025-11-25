# Product Requirements Document: Intelligent Memory Routing System (Tinker Implementation)

## 1. Executive Summary
Deliver a production memory-routing agent for marketing AI systems using Tinker as the exclusive training platform. The agent ingests conversation snippets, decides if the content merits storage, assigns the correct ontology slot, and respects persistence horizons. We follow a two-stage pipeline: supervised prompt distillation for initialization and reinforcement learning to optimize downstream retrieval utility. All code must rely on Tinker’s asynchronous APIs and built-in loss functions (cross-entropy and importance sampling) to stay within supported patterns.

## 2. Goals and Non-Goals
- **Goals**
  - Maintain selective, temporally-aware storage across the 12-category taxonomy plus `none`.
  - Achieve reliable multi-label routing with tight output formatting compatible with Tinker renderers.
  - Support RL reward shaping for retrieval F1, temporal correctness, company/user separation, and storage efficiency.
  - Produce checkpoints consumable by downstream services via Tinker sampling clients.
- **Non-Goals**
  - Building a retrieval engine or serving layer.
  - Extending Tinker beyond LoRA or supported loss functions.

## 3. Success Metrics
- Exact-match accuracy ≥80% on held-out labeled data.
- Macro F1 ≥90% across categories; `none` precision ≥90%, recall ≥85%.
- Average predicted categories per utterance ≤2.0.
- Temporal alignment accuracy ≥90% (long/medium/short mapping).
- KL divergence: target <0.005, warn 0.005–0.01, critical >0.01.

## 4. System Architecture Overview
1. **Synthetic Conversation Library** – Scenario templates drive GPT-5 generation to cover ontology breadth and noise patterns.
2. **Teacher Labeling** – GPT-5, prompted with taxonomy/persistence guidance, produces gold labels (multi-label + `none`).
3. **Prompt Distillation (SFT)** – Llama-3.1-8B LoRA (rank 32) is trained asynchronously via Tinker `forward_backward_async(..., loss_fn="cross_entropy")`.
4. **RL Optimization** – Same model undergoes importance sampling policy gradient loss with a custom `MemoryRoutingEnv`.
5. **Evaluation Harness** – Tinker evaluator builders and offline scripts verify accuracy, pruning behavior, and reward stability.

### Model Selection Rationale
- Using `meta-llama/Llama-3.1-8B` (🐙 Base, 🧱 Dense, 🦆 Small) as the foundation model for this classification task. While Tinker recommends MoE models for cost efficiency and instruction-tuned models for task-specific work, we choose the base model for three reasons:
  1. **Routing Neutrality**: Instruction-tuned models may have ingrained biases toward helpfulness/verbosity that conflict with selective storage decisions. The base model learns routing behavior purely from our synthetic data.
  2. **Prompt Distillation Alignment**: Our two-stage pipeline (teacher labels → SFT → RL) is a classic prompt distillation setup where starting from a base model ensures we're not fighting pre-existing instruction-following patterns.
  3. **Evaluation Baseline**: Establishes a clean baseline for comparing LoRA vs full fine-tuning effects without confounding variables from prior post-training.
- LoRA rank 32 mirrors Tinker defaults for classification-style tasks. Higher ranks can be evaluated later if capacity becomes a bottleneck.
- **Post-MVP**: Once baseline performance is established, evaluate `meta-llama/Llama-3.1-8B-Instruct` (to measure instruction-tuning impact) and `Qwen/Qwen3-30B-A3B` (MoE cost efficiency) as alternative starting points.

### Environment Design Notes
- Each `MemoryRoutingEnv` is a single-step bandit: `initial_observation()` returns a tokenized conversation + stop conditions, `step()` receives the model's generated classification tokens and terminates immediately with reward.
- EnvGroupBuilder clones each conversation across `group_size` rollouts for variance reduction; dataset builder provides `batch_size` EnvGroupBuilders per iteration.
- No multi-turn transitions, which matches Tinker's Env definitions and keeps reward computation simple.

### Reward Computation Details
The `step()` method in `MemoryRoutingEnv` performs the following sequence:
1. **Parse Model Output**: Extract predicted categories from generated tokens using renderer stop sequences. Expected format: `category1, category2, category3` (comma-separated, from valid taxonomy).
2. **Format Validation**: If parsing fails or any category is invalid, assign `R_format = -1.0` and return immediately (zero for all other reward components).
3. **Component Calculation**:
   - `R_F1`: Token-level F1 between predicted and gold category sets. Use macro-averaging if multi-label.
   - `R_temp`: Persistence alignment. +1.0 if predicted persistence matches gold (long/medium/short), +0.5 if adjacent (long↔medium or medium↔short), 0.0 otherwise. Use majority vote if multi-label predictions span multiple persistence horizons.
   - `R_parity`: Company/user scope alignment. +1.0 if predicted scope (company/user/mixed/none) matches gold scope exactly, 0.0 otherwise.
   - `R_eff`: Storage efficiency. `1.0` if ≤3 categories predicted, `0.7` if 4 categories, `0.4` if 5 categories, `0.0` if ≥6 categories.
4. **Composite Reward**: `R_total = 0.6 * R_F1 + 0.2 * R_temp + 0.1 * R_parity + 0.1 * R_eff` (unless format validation failed, then `R_total = -1.0`).

**Edge Cases**:
- Model outputs empty string or only stop tokens → format validation failure.
- Model outputs `none` + other categories → invalid, format failure (none must be exclusive).
- Model outputs duplicate categories → deduplicate before computing metrics.
- Model exceeds max_tokens without hitting stop sequence → truncate and attempt parse, format failure if no valid categories extracted.

## 5. Memory Ontology
| Category | Description | Persistence |
| --- | --- | --- |
| `company.brand_core` | Voice, values, positioning, identity anchors. | Long (>1y) |
| `company.strategic_signatures` | Decision frameworks, strategic heuristics. | Long (>1y) |
| `company.knowledge_artifacts` | Docs, style guides, playbooks. | Long (>1y) |
| `company.business_priorities` | Quarterly/seasonal goals, active campaigns. | Short (<3m) |
| `company.tools_config` | Integrations, API keys, workflow settings. | Medium (~6m) |
| `company.performance_context` | Campaign metrics, retrospectives, learnings. | Rolling (~6m) |
| `user.communication_style` | Tone, verbosity, format expectations. | Long (>1y) |
| `user.strategic_approach` | Personal priorities, success definitions. | Long (>1y) |
| `user.role_context` | Title, scope, decision authority. | Medium (~1y) |
| `user.workflow_patterns` | Review cadence, collaboration norms. | Medium (~1y) |
| `user.session_history` | Immediate context, recent asks. | Short (<2w) |
| `user.interaction_preferences` | Coaching style, feedback expectations. | Evolving |
| `none` | Irrelevant, vague, or transactional content. | Critical for noise reduction |

## 6. Data & Prompt Strategy

### Scenario Generation
- Script: customize `tinker_cookbook/recipes/prompt_distillation/create_data.py`.
- Inputs: category focus, distractor category, emotional tone, required signal; 4–10 turns per dialogue.
- Outputs: JSONL with scenario metadata, teacher confidence, persistence hints.

### Teacher Prompt
```
System: You route marketing conversations into persistent memory. Consider each utterance and decide if it conveys a durable fact. Prefer `none` unless confident.
Ontology: <category table with definitions + persistence>
Rules:
  1. Distinguish company.* from user.* details.
  2. Match persistence horizon (long/medium/short) to signal lifetime.
  3. Predict ≤3 categories unless strictly necessary.
Output:
categories: cat1, cat2 (use `none` for no storage)
```
- Temperature 0.2, max tokens 256, stop newline.

### Student Prompt
```
System: You route marketing conversations into structured memory categories.
User: Conversation:
{dialogue}

Available categories:
- company.brand_core ...
- ...
- none

Respond with comma-separated categories.
```

### Renderer Configuration
```python
from tinker_cookbook import renderers, tokenizer_utils

tokenizer = tokenizer_utils.get_tokenizer("meta-llama/Llama-3.1-8B")
renderer = renderers.get_renderer(name="llama3", tokenizer=tokenizer)
stop_sequences = renderer.get_stop_sequences()

sampling_params = types.SamplingParams(
    max_tokens=150,
    temperature=0.0,
    stop=stop_sequences,
)
```

### Parsing & Validation
- Normalize whitespace/case, strip bullets, deduplicate, enforce taxonomy membership.
- Validation helper:
```python
def validate_datum(datum: types.Datum, vocab_size: int) -> bool:
    if datum.model_input.length > 512:
        return False
    weights = datum.loss_fn_inputs["weights"].tolist()
    if sum(weights) == 0:
        return False
    target_tokens = datum.loss_fn_inputs["target_tokens"].tolist()
    if not all(0 <= t < vocab_size for t in target_tokens):
        return False
    return True
```

## 6.5 Synthetic Data Quality Assurance
- **Coverage:** ≥20 examples per category × persistence; ≥20% multi-label, ≥10% `none`-only.
- **Noise:** ≥30% dialogues include distractors to stress selectivity.
- **Signal Density:** >60% of turns include relevant info; length 6.5 ± 1.5 turns.
- **Human Audit:** Spot-check 100 samples per refresh; require ≥95% teacher agreement.
- **Continuous Improvement:** Log production misses, refresh quarterly, retrain teacher prompt if accuracy drops >10%.

## 6.6 Data Preprocessing Pipeline

Before SFT training, synthetic JSONL conversations must be converted to Tinker-compatible `types.Datum` objects:

**Step 1: Load Synthetic Data**
```python
import json
with open("train.jsonl", "r") as f:
    conversations = [json.loads(line) for line in f]
```

**Step 2: Convert to Datum Objects**
```python
from tinker import types
from tinker_cookbook import renderers, tokenizer_utils

tokenizer = tokenizer_utils.get_tokenizer("meta-llama/Llama-3.1-8B")
renderer = renderers.get_renderer(name="llama3", tokenizer=tokenizer)

def conversation_to_datum(conversation_json: dict) -> types.Datum:
    """Convert synthetic conversation to training datum."""
    tokens, weights = renderer.build_supervised_example(
        conversation_json["conversation"]
    )
    model_input = types.ModelInput.from_ints(tokens[:-1])
    datum = types.Datum(
        model_input=model_input,
        loss_fn_inputs=dict(
            target_tokens=tokens[1:],
            weights=weights[1:],
        ),
    )
    return datum

train_data = [conversation_to_datum(conv) for conv in conversations]
```

**Step 3: Validate Datum Objects**
```python
vocab_size = len(tokenizer)
valid_data = []
for datum in train_data:
    if datum.model_input.length > 4096:
        print(f"Warning: Skipping example with length {datum.model_input.length}")
        continue
    weights = datum.loss_fn_inputs["weights"].tolist()
    if sum(weights) == 0:
        print("Warning: Skipping example with zero loss weights")
        continue
    target_tokens = datum.loss_fn_inputs["target_tokens"].tolist()
    if not all(0 <= t < vocab_size for t in target_tokens):
        print(f"Warning: Invalid token IDs found")
        continue
    valid_data.append(datum)

print(f"Preprocessed {len(valid_data)}/{len(train_data)} examples")
```

**Step 4: Split and Save**
```python
train_size = int(0.8 * len(valid_data))
train_dataset = valid_data[:train_size]
test_dataset = valid_data[train_size:]
```

## 7. Stage 1 – Prompt Distillation (Supervised Learning)

### Dataset & Batch Size
- 1–2k labeled conversations (80/20 split after preprocessing per Section 6.6).
- Batch size 128 (per Tinker SL guidance) balances stability/throughput; if changed, scale LR ∝ √batch_size.
- Expected preprocessing yield: ~90-95% of raw JSONL (some examples filtered for length/validity).

### Hyperparameter Selection
```python
from tinker_cookbook.hyperparam_utils import get_lr

model_name = "meta-llama/Llama-3.1-8B"
learning_rate = get_lr(model_name)  # Returns LoRA-adjusted LR: ~2.86e-4
```
- Tinker's `get_lr()` utility already returns the LoRA-optimized learning rate for the specified model, accounting for model size and architecture. No manual scaling needed.
- Use Adam β1=0.9, β2=0.95, ε=1e-8 (Tinker SL defaults).
- **Training Duration**: Start with 300 steps minimum (≈20-25 epochs for 1.5k samples at batch_size=128). Tinker SL guidance recommends "at least 100 steps but usually best results with 1000 or more" - for LoRA classification tasks, 300-500 steps typically ensures convergence.
- **Early Stopping**: Validate every 20 steps on test set. Stop if test loss doesn't improve for 5 consecutive evaluations (100 steps patience).
- **Convergence Check**: Plot train/test loss curves. If test loss hasn't plateaued by step 300, extend to 500 steps before RL initialization.

### Async Training Loop
```python
import tinker
from tinker import types
from tinker_cookbook.hyperparam_utils import get_lr

service_client = tinker.ServiceClient()
training_client = await service_client.create_lora_training_client_async(
    base_model="meta-llama/Llama-3.1-8B",
    rank=32,
)

learning_rate = get_lr("meta-llama/Llama-3.1-8B")

for step in range(num_steps):
    # Submit forward-backward pass
    fwd_bwd_future = await training_client.forward_backward_async(
        batch_data,
        loss_fn="cross_entropy",
    )
    
    # Submit optimizer step (can overlap with forward-backward)
    adam_params = types.AdamParams(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )
    optim_future = await training_client.optim_step_async(adam_params)

    # Wait for both operations to complete
    fwd_bwd_result = await fwd_bwd_future.result_async()
    optim_result = await optim_future.result_async()
    
    # Log metrics from both operations
    log_metrics(step, fwd_bwd_result, optim_result)
```

### Checkpointing & Sampling
```python
# Save checkpoint for sampling (every 20 steps)
checkpoint_future = await training_client.save_weights_for_sampler_async(
    name=f"sft_{step:04d}"
)
checkpoint_result = await checkpoint_future.result_async()
sampling_path = checkpoint_result.path

# Create sampling client with the checkpoint
sampling_client = service_client.create_sampling_client(
    model_path=sampling_path
)

# Use with renderer stop sequences for evaluation
stop_sequences = renderer.get_stop_sequences()
sampling_params = types.SamplingParams(
    max_tokens=150,
    temperature=0.0,
    stop=stop_sequences,
)
```
- Save weights every 20 steps for periodic evaluation.
- Critical: Must call `.result_async()` on the checkpoint future to get the path before creating sampling client.

## 8. Stage 2 – Reinforcement Learning

### Environment & Reward
- `MemoryRoutingEnv` implements single-step episodes; EnvGroupBuilder replicates conversations across `group_size=8`.
- Reward: `0.6 * R_F1 + 0.2 * R_temp + 0.1 * R_parity + 0.1 * R_eff`.
  - `R_F1`: F1 overlap with teacher labels.
  - `R_temp`: +1 (correct persistence), +0.5 (adjacent), 0 otherwise.
  - `R_parity`: +1 when company/user presence matches ground truth.
  - `R_eff`: 1.0 (≤3 cats), 0.7 (4), 0.4 (5), 0 (≥6) with hard penalty for parser failures.

### Policy & Sampling Workflow
```python
# Save current policy weights for sampling
checkpoint_future = await training_client.save_weights_for_sampler_async(
    name=f"rl_step_{step:04d}"
)
checkpoint_result = await checkpoint_future.result_async()
sampling_path = checkpoint_result.path

# Create sampling client with current policy
sampling_client = service_client.create_sampling_client(
    model_path=sampling_path,
)

# Wrap in policy completer for RL rollouts
policy = TinkerTokenCompleter(
    sampling_client=sampling_client,
    max_tokens=150,
    temperature=0.0,
    stop=renderer.get_stop_sequences(),
)
```

### Async Training Loop
```python
for iteration in range(num_iterations):
    # 1. Gather rollouts concurrently
    trajectory_groups = await asyncio.gather(
        *[do_group_rollout(env_builder, policy) for env_builder in env_builders]
    )
    
    # 2. Process trajectories
    filtered_groups = remove_constant_reward_groups(trajectory_groups)
    advantages = compute_advantages(filtered_groups)
    train_data, metadata = assemble_training_data(filtered_groups, advantages)
    
    # 3. Submit forward-backward pass
    fwd_bwd_future = await training_client.forward_backward_async(
        train_data,
        loss_fn="importance_sampling"
    )
    
    # 4. Submit optimizer step
    adam_params = types.AdamParams(
        learning_rate=2e-5,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8
    )
    optim_future = await training_client.optim_step_async(adam_params)
    
    # 5. Wait for both operations to complete
    fwd_bwd_result = await fwd_bwd_future.result_async()
    optim_result = await optim_future.result_async()
    
    # 6. Log metrics including KL divergence
    log_metrics(iteration, fwd_bwd_result, optim_result, metadata)
```
- Run ≈25 iterations (256 rollouts each). Adjust based on convergence and KL monitoring.

### KL Monitoring
| Status | KL Range | Action |
| --- | --- | --- |
| Target | <0.005 | Optimal on-policy stability |
| Warning | 0.005–0.01 | Log warning, monitor closely; still stable per Tinker guidance |
| Critical | >0.01 | Halt run immediately, inspect sampler vs learner drift |

**Implementation Notes**:
- Always log `kl_sample_train_v1` and `kl_sample_train_v2` (two KL estimators per Tinker RL docs).
- Per Tinker: "training is stable with KL divergence below 0.01" - values above this threshold indicate numerical instability or off-policy issues.
- Even with full on-policy training, KL won't be exactly zero due to [non-determinism](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) in batched inference.
- Keep sampling temperature at 0.0 for deterministic rollouts.
- Enable gradient clipping (max_norm=1.0) if KL repeatedly exceeds 0.005.
- If KL exceeds 0.01, halt training, inspect checkpoint drift, and verify sampling client is using correct weights.

### Future Throughput Optimizations
- After MVP, consider `StreamMinibatchConfig` to overlap sampling/training per Tinker RL docs (20–30% throughput gain).

## 9. Evaluation & Monitoring

### Inline
- SFT: track train/test loss, exact-match, macro/micro F1, avg categories.
- RL: log reward components, KL metrics, entropy, avg categories, stop reasons.

### Evaluators
```python
from tinker_cookbook.evaluators import SamplingClientEvaluator
from tinker import types

class MemoryRoutingEvaluator(SamplingClientEvaluator):
    """Evaluates memory routing classification on held-out test set."""
    
    def __init__(self, test_set, renderer, tokenizer):
        self.test_set = test_set  # List of preprocessed conversations with gold labels
        self.renderer = renderer
        self.tokenizer = tokenizer

    async def __call__(self, sampling_client):
        """Run holdout evaluation on the test set."""
        predictions = []
        gold_labels = []
        
        # Sample predictions for each test conversation
        for example in self.test_set:
            # Build generation prompt from conversation
            prompt = self.renderer.build_generation_prompt(
                example["conversation"]
            )
            
            # Generate classification
            sampling_params = types.SamplingParams(
                max_tokens=150,
                temperature=0.0,
                stop=self.renderer.get_stop_sequences(),
            )
            result = await sampling_client.sample_async(
                prompt=prompt,
                num_samples=1,
                sampling_params=sampling_params
            )
            
            # Parse model output into categories
            pred_tokens = result.sequences[0].tokens
            pred_text = self.tokenizer.decode(pred_tokens)
            pred_categories = self._parse_categories(pred_text)
            
            predictions.append(pred_categories)
            gold_labels.append(set(example["labels"]["categories"]))
        
        # Compute metrics
        return {
            "exact_match": self._compute_exact_match(predictions, gold_labels),
            "macro_f1": self._compute_macro_f1(predictions, gold_labels),
            "none_precision": self._compute_none_precision(predictions, gold_labels),
            "temporal_accuracy": self._compute_temporal_accuracy(predictions, gold_labels),
        }
    
    def _parse_categories(self, text: str) -> set:
        """Parse comma-separated categories from model output."""
        # Implementation: split on comma, strip whitespace, validate against taxonomy
        # Return set of valid categories or {"none"} if parsing fails
        pass
    
    def _compute_exact_match(self, preds, golds) -> float:
        """Fraction of examples where predicted set exactly matches gold set."""
        pass
    
    def _compute_macro_f1(self, preds, golds) -> float:
        """Macro-averaged F1 across all categories."""
        pass
    
    def _compute_none_precision(self, preds, golds) -> float:
        """Precision of 'none' category predictions."""
        pass
    
    def _compute_temporal_accuracy(self, preds, golds) -> float:
        """Accuracy of persistence horizon alignment (requires loading full examples)."""
        pass
```
- Register evaluator builders with `eval_every=20` for SFT (every checkpoint) and RL loops.
- Consider Inspect AI tasks after MVP for standardized benchmarking.

### Offline & Compliance
- Offline script computes exact-match, macro/micro F1, `none` precision/recall, temporal accuracy, confusion matrix.
- Regression suite: 100 held-out dialogues rerun after each checkpoint.
- Format validator ensures comma-separated taxonomy outputs and ≤3 categories typical.

## 10. Implementation Plan
1. **Scenario Refresh & QA** – Generate new datasets, run teacher labeling, enforce Section 6.5 checks.
2. **Preprocessing & Validation** – Convert to `Datum`, run parser + validator.
3. **SFT Training** – 120–160 async steps with early stopping, checkpoint weights.
4. **RL Environment Build** – Implement env/reward/evaluators, add unit tests.
5. **RL Training** – 25 iteration importance sampling run with KL monitoring.
6. **Evaluation & Sign-off** – Execute evaluator builders + offline scripts, capture qualitative samples, document results.
7. **Future Optimization** – Investigate streaming minibatch and Inspect AI integration after MVP.

## 11. Risks & Mitigations
- **Format Drift:** reward penalty + strict parser; renderer stop sequences enforce termination.
- **`none` Collapse:** reward weights emphasize recall, track per-category confusion, rebalance data.
- **Off-Policy Instability:** monitor KL each step, warn at 0.01, halt at 0.05, keep temperature=0.0, clip gradients.
- **Temporal Mislabeling:** targeted scenario generation plus dedicated reward component; run temporal audits weekly.
- **Synthetic Bias:** quarterly data refresh with human audits; ingest production edge cases.

## 12. Deployment Considerations
- **Inference:** Export final LoRA checkpoint via `save_weights_for_sampler(name="prod_v1")`; serve via Tinker SamplingClient or export to preferred inference stack.
- **Performance Targets:** <200 ms p95 latency per routing decision; ≥100 decisions/sec on A100 (LoRA overhead ≈8 GB).
- **Monitoring:** Weekly dashboards for category distribution, `none` precision (>85%), avg categories (<2.5), temporal accuracy, reward drift.
- **Versioning:** Semantic versioning (major.minor.patch); record lineage (base → SFT → RL); keep last 3 versions for rollback.

---
**Owner:** Technical Architecture Lead

