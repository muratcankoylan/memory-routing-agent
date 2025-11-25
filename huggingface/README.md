---
license: apache-2.0
language:
- en
tags:
- memory-routing
- marketing
- classification
- llama
- lora
- tinker
base_model: meta-llama/Llama-3.1-8B
datasets:
- muratcankoylan/memory-routing-marketing
metrics:
- f1
- accuracy
pipeline_tag: text-classification
---

# Memory Routing Agent (Llama-8B + LoRA)

A specialized 8B parameter model that **outperforms 104B models** on marketing conversation classification.

## Key Results

| Model | Size | Avg F1 | Exact Match |
|-------|------|--------|-------------|
| **This Model** | 8B | **0.68** | **60%** |
| Cohere Command-R-Plus | 104B | 0.61 | 26% |

**11.1% higher F1** than the 104B teacher model that generated its training data.

## Model Description

The Memory Routing Agent classifies marketing conversations into 13 memory categories:

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

## Training

- **Base Model**: meta-llama/Llama-3.1-8B
- **Method**: LoRA (rank 32) + SFT + RL
- **Platform**: Tinker (Thinking Machines)
- **Dataset**: 2,001 synthetic marketing conversations
- **Teacher**: Cohere Command-R-Plus (104B)

### Training Pipeline

1. **SFT Phase**: 100 steps, batch size 128, cross-entropy loss
2. **RL Phase**: 12 iterations, importance sampling policy gradient
3. **Reward**: 0.6×F1 + 0.2×temporal + 0.1×parity + 0.1×efficiency

## Usage

```python
# Note: This model was trained on Tinker platform
# The checkpoint is: tinker://4f4bae1f-5a95-5f53-a55a-a14f2872825c:train:0/sampler_weights/rl_iter_012

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

service_client = tinker.ServiceClient()
checkpoint = "tinker://4f4bae1f-5a95-5f53-a55a-a14f2872825c:train:0/sampler_weights/rl_iter_012"
sampling_client = service_client.create_sampling_client(model_path=checkpoint)

tokenizer = get_tokenizer("meta-llama/Llama-3.1-8B")
renderer = renderers.get_renderer(name="llama3", tokenizer=tokenizer)

conversation = """
USER: Our brand voice is professional but approachable.
ASSISTANT: So authoritative content with a conversational tone?
USER: Exactly. We never use jargon without explaining it first.
"""

messages = [
    {"role": "system", "content": "You route marketing conversations into structured memory categories..."},
    {"role": "user", "content": f"Analyze this conversation:\n\n{conversation}"}
]

prompt = renderer.build_generation_prompt(messages)
params = types.SamplingParams(max_tokens=100, temperature=0.1, stop=renderer.get_stop_sequences())
result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1).result()

response, _ = renderer.parse_response(result.sequences[0].tokens)
print(f"Categories: {response['content']}")
# Output: company.brand_core
```

## Benchmark

50 challenging marketing scenarios across 7 domains:

| Difficulty | Our Model | Cohere (104B) |
|------------|-----------|---------------|
| Easy | 0.86 F1 | 0.48 F1 |
| Medium | 0.65 F1 | 0.64 F1 |
| Hard | 0.50 F1 | 0.72 F1 |

## Limitations

- Under-predicts multi-label scenarios
- Sometimes confuses company vs user categories
- Marketing-specific; not tested on other domains

## Citation

```bibtex
@misc{memory-routing-agent-2024,
  title={Memory Routing Agent: Prompt Distillation for Marketing AI},
  author={Muratcan Koylan},
  year={2024},
  howpublished={\url{https://github.com/muratcankoylan/memory-routing-agent}},
}
```

## Links

- **GitHub**: [muratcankoylan/memory-routing-agent](https://github.com/muratcankoylan/memory-routing-agent)
- **Training Platform**: [Tinker by Thinking Machines](https://thinkingmachines.ai/)

