# I Used Claude to Train My First AI Model - Here's What I Learned

Last week, I trained my first machine learning model. Not by taking a course or following a tutorial, but by pair-programming with Claude Opus 4.5. The result: an 8B parameter model that outperforms its 104B teacher by 11%.

Here's the full story.

---

## The Problem

I'm building a marketing AI assistant that needs to remember the right things from conversations. Not everything is worth storing:

- "Our brand voice is professional but approachable" → **Store this forever**
- "What time is the meeting?" → **Don't store**

This sounds simple, but it's actually a nuanced 13-category classification problem. You need to distinguish between company-level and user-level information, understand persistence horizons (some things expire in weeks, others last years), and critically, know when to say "none."

## The Approach: Prompt Distillation

I decided to try **prompt distillation** - using a large model to generate training data, then training a smaller model to do the same task better.

The pipeline:
1. Use Cohere Command-R-Plus (104B) to generate 2,001 labeled conversations
2. Fine-tune Llama-3.1-8B with LoRA
3. Apply reinforcement learning to optimize for exact matching
4. Benchmark against the teacher

## Working with Claude

I didn't know how to implement any of this. I'd never used PyTorch, never trained a model, never written a reward function. But I had Claude.

What made this work:

### 1. Claude as a Technical Lead

I didn't just ask Claude to write code. I asked it to think like Andrej Karpathy training this model. To challenge my assumptions. To point out when something was wrong.

When my RL training showed 0% accuracy for 5 iterations, Claude didn't just debug - it identified that I was saving SFT weights incorrectly, preventing RL from continuing the training. That's the kind of insight you'd get from an experienced ML engineer.

### 2. Iterative Debugging

Training ML models is messy. My first synthetic data was too homogeneous (everything was eco-related). My reward function had negative KL divergence (mathematically impossible). My batch sizes were wrong.

Each time, Claude helped diagnose the issue, propose a fix, and verify it worked. We went through probably 20 iterations of the training pipeline before it worked correctly.

### 3. Reading Documentation Together

The Tinker platform (by Thinking Machines) has excellent documentation, but it's dense. Claude helped me understand:
- When to use `save_state()` vs `save_weights_for_sampler()`
- How to compute advantages for importance sampling
- Why KL divergence monitoring matters for training stability

Having an AI that could read the docs, understand my codebase, and synthesize both was invaluable.

## The Tinker Platform

I used [Tinker](https://thinkingmachines.ai/) for training. A few things that stood out:

**What worked well:**
- Async API design - you can overlap forward/backward passes with optimizer steps
- Built-in loss functions for both SFT (`cross_entropy`) and RL (`importance_sampling`)
- Checkpoint management - easy to save and resume training
- The documentation is genuinely excellent. Every API is explained with examples.

**What I learned:**
- LoRA rank 32 is the sweet spot for classification tasks
- You need at least 100 SFT steps before RL makes sense
- KL divergence should stay below 0.01 for stable training
- Always shuffle your training data between epochs

## The Results

After SFT + RL training:

| Model | Size | F1 Score | Exact Match |
|-------|------|----------|-------------|
| My Model | 8B | **0.68** | **60%** |
| Cohere (Teacher) | 104B | 0.61 | 26% |

**The 8B student outperformed the 104B teacher by 11% on F1 and 2.3x on exact match.**

Why? Three reasons:

1. **Focused training**: My model only does one thing. The teacher is a general-purpose model.
2. **RL optimization**: The reward function explicitly rewards exact category matching, not just plausible outputs.
3. **Clean data**: Synthetic data with consistent labeling. No noise from human annotator disagreements.

## Lessons Learned

### On AI-Assisted Development

1. **Treat AI as a senior engineer, not a code generator.** Ask it to review your approach, not just implement it.

2. **Iterate fast.** We probably ran 50+ experiments. Most failed. That's fine.

3. **Read the errors carefully.** Claude is great at debugging, but you need to give it the full context.

### On Model Training

1. **Data quality > data quantity.** My first 1,000 examples were too similar. The second 1,000 with higher temperature and more diverse prompts made a bigger difference than doubling the dataset.

2. **SFT gets you 80% of the way.** RL is for the last 20%. Don't skip SFT.

3. **Log everything.** I saved metrics every step. When something went wrong, I could trace exactly where.

4. **Test on novel inputs.** The model can overfit to your test set. Create truly new scenarios to verify generalization.

### On Prompt Distillation

1. **The student can beat the teacher.** This still surprises me. But specialization beats generalization for narrow tasks.

2. **RL is powerful but finicky.** Getting the reward function right took multiple iterations. Start simple.

3. **Multi-label is hard.** My model still struggles when 3+ categories apply. This is a known limitation.

## What's Next

The model is open-source:
- **HuggingFace**: [MuratcanKoylan/Marketing-Memory-Routing-8B](https://huggingface.co/MuratcanKoylan/Marketing-Memory-Routing-8B)
- **GitHub**: [muratcankoylan/memory-routing-agent](https://github.com/muratcankoylan/memory-routing-agent)

I'm planning to:
1. Train on a larger dataset (10k+ examples)
2. Try different base models (Qwen, Mistral)
3. Add per-category evaluation metrics
4. Create a proper benchmark suite

## Final Thoughts

A year ago, training a custom ML model felt like something only ML engineers could do. Now, with the right AI assistant and a good training platform, it's accessible to anyone willing to iterate.

The model isn't perfect. It struggles with edge cases. But it works, it's 13x smaller than the alternative, and I built it in a week.

If you're thinking about training your first model, just start. Pick a narrow problem. Generate some data. Train something. See what breaks. Fix it. Repeat.

---

*Thanks to Thinking Machines for Tinker, Cohere for the teacher model, Meta for Llama, and Anthropic for Claude.*

**Links:**
- Model: https://huggingface.co/MuratcanKoylan/Marketing-Memory-Routing-8B
- Code: https://github.com/muratcankoylan/memory-routing-agent
- Tinker: https://thinkingmachines.ai/

