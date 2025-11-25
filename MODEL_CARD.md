# Model Card: Memory Routing Agent (Llama-8B + LoRA)

## Model Details

- **Model Name**: memory-routing-llama-8b-lora
- **Base Model**: meta-llama/Llama-3.1-8B
- **Architecture**: LoRA (Low-Rank Adaptation), rank 32
- **Training Platform**: Tinker (Thinking Machines)
- **Training Method**: SFT (Supervised Fine-Tuning) + RL (Reinforcement Learning)
- **Parameters**: ~8B base + ~100M LoRA adapters
- **License**: Apache 2.0

## Intended Use

This model classifies marketing conversations into memory categories for AI assistant systems. It determines which pieces of information from a conversation should be stored in long-term memory and how they should be categorized.

### Primary Use Cases
- Marketing AI assistants that need to remember user preferences
- CRM systems that extract structured data from conversations
- Knowledge management systems for marketing teams

### Out-of-Scope Uses
- General-purpose chatbots
- Non-marketing domains (healthcare, legal, finance)
- Real-time conversation generation

## Training Data

### Synthetic Dataset
- **Size**: 2,001 conversations
- **Generation**: Cohere Command-R-Plus (104B) as teacher model
- **Format**: Multi-turn marketing conversations with category labels

### Category Taxonomy (13 categories)
| Category | Description | Persistence |
|----------|-------------|-------------|
| company.brand_core | Voice, values, positioning | Long (>1y) |
| company.strategic_signatures | Decision frameworks | Long (>1y) |
| company.knowledge_artifacts | Docs, style guides | Long (>1y) |
| company.business_priorities | Quarterly goals | Short (<3m) |
| company.tools_config | Integrations, APIs | Medium (~6m) |
| company.performance_context | Campaign metrics | Rolling (~6m) |
| user.communication_style | Tone, format preferences | Long (>1y) |
| user.strategic_approach | Personal priorities | Long (>1y) |
| user.role_context | Title, scope | Medium (~1y) |
| user.workflow_patterns | Review cadence | Medium (~1y) |
| user.session_history | Immediate context | Short (<2w) |
| user.interaction_preferences | Coaching style | Evolving |
| none | Irrelevant content | N/A |

## Training Procedure

### Phase 1: Supervised Fine-Tuning (SFT)
- **Steps**: 100
- **Batch Size**: 128
- **Learning Rate**: 2.86e-4 (Tinker default for Llama-8B)
- **Optimizer**: Adam (β1=0.9, β2=0.95)
- **Loss Function**: Cross-entropy

### Phase 2: Reinforcement Learning (RL)
- **Iterations**: 12
- **Groups per Batch**: 64
- **Group Size**: 32
- **Learning Rate**: 2e-5
- **Loss Function**: Importance sampling policy gradient
- **Reward Function**: 
  - R_F1 (60%): F1 score vs gold labels
  - R_temp (20%): Temporal alignment
  - R_parity (10%): Company/user scope
  - R_eff (10%): Storage efficiency

## Evaluation Results

### Marketing Routing Benchmark (50 scenarios)

| Model | Any Match | Exact Match | Avg F1 |
|-------|-----------|-------------|--------|
| **Ours (8B + LoRA)** | 72% | **60%** | **0.68** |
| Cohere Command-R-Plus (104B) | 82% | 26% | 0.61 |

### Key Findings
- **11.1% higher F1** than the 104B teacher model
- **2.3x better exact match** accuracy
- **13x smaller** than the teacher model
- Excels at single-category classification (86% exact on easy cases)
- Struggles with multi-label scenarios (10% exact on hard cases)

### Performance by Difficulty
| Difficulty | Our Model (F1) | Cohere (F1) | Delta |
|------------|----------------|-------------|-------|
| Easy | 0.86 | 0.48 | +79% |
| Medium | 0.65 | 0.64 | +2% |
| Hard | 0.50 | 0.72 | -31% |

## Limitations

1. **Multi-label Detection**: Under-predicts when multiple categories apply
2. **Company vs User Confusion**: Sometimes confuses `company.strategic_signatures` with `user.strategic_approach`
3. **Hard Cases**: Performance drops on complex overlapping categories
4. **Domain Specificity**: Trained only on marketing scenarios

## Ethical Considerations

- Model trained on synthetic data; may not capture all real-world edge cases
- Should be used with human oversight for critical decisions
- Privacy: Does not store or transmit conversation data

## Citation

```bibtex
@misc{memory-routing-agent-2025,
  title={Memory Routing Agent: Prompt Distillation for Marketing AI},
  author={Muratcan Koylan},
  year={2025},
  howpublished={\url{https://github.com/muratcankoylan/memory-routing-agent}},
}
```

## Model Files

- `training/checkpoints/rl_iter_012/` - Final RL checkpoint
- `training/benchmarks/marketing_routing_benchmark.json` - Benchmark dataset
- `synthetic_data/merged_training_dataset_2001.jsonl` - Training data

## Contact

For questions or issues, please open a GitHub issue.

