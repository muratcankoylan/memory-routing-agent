# Synthetic Data Generation Guide: Memory Routing System

## Overview
This guide provides detailed specifications for generating synthetic training data for the intelligent memory routing system. The data will be used to train both the supervised (teacher labels) and RL (reward signals) stages.

---

## Output Format Specification

### JSONL Schema
Each line in the output file should be a valid JSON object with the following structure:

```json
{
  "scenario_id": "string",
  "conversation": [
    {
      "role": "user|assistant",
      "content": "string"
    }
  ],
  "labels": {
    "categories": ["string"],
    "persistence_horizon": "long|medium|short",
    "memory_scope": "company|user|mixed|none",
    "rationale": "string"
  },
  "metadata": {
    "scenario_type": "string",
    "primary_category": "string",
    "distractor_present": boolean,
    "turn_count": integer,
    "signals_present": ["string"]
  }
}
```

### Field Definitions

**scenario_id**: Unique identifier (format: `{category}_{type}_{counter}`, e.g., `brand_core_standard_001`)

**conversation**: Array of message objects representing the dialogue
- Must be 4-10 turns total
- Should alternate between user and assistant (can start with either)
- Content should be realistic marketing/strategy dialogue

**labels.categories**: Array of category strings from taxonomy
- Valid values: `company.brand_core`, `company.strategic_signatures`, `company.knowledge_artifacts`, `company.business_priorities`, `company.tools_config`, `company.performance_context`, `user.communication_style`, `user.strategic_approach`, `user.role_context`, `user.workflow_patterns`, `user.session_history`, `user.interaction_preferences`, `none`
- Can be multi-label (typically 1-3 categories)
- Use `["none"]` for transactional/vague content

**labels.persistence_horizon**: Expected lifetime of information
- `long`: >1 year (e.g., brand values, communication style)
- `medium`: 6-12 months (e.g., role context, tools config)
- `short`: <3 months (e.g., business priorities, session history)

**labels.memory_scope**: Who this information pertains to
- `company`: Company-level information (brand, processes, etc.)
- `user`: Individual user preferences/context
- `mixed`: Contains both company and user information
- `none`: No memorable information

**labels.rationale**: Brief explanation (1-2 sentences) of why these categories were chosen

**metadata**: Additional context for training/evaluation
- `scenario_type`: Descriptive label (e.g., "brand_discovery", "campaign_review", "preference_setting")
- `primary_category`: The main category this example focuses on
- `distractor_present`: Whether irrelevant information was intentionally included
- `turn_count`: Number of conversation turns
- `signals_present`: List of specific signals (e.g., ["brand_voice_example", "tone_preference", "transactional_question"])

---

## Data Generation Prompts

### Stage 1: Scenario Generation Prompt

Use this prompt to generate diverse scenario specifications first, then use those to create conversations.

```
You are designing training scenarios for an AI memory system in marketing context. Generate a scenario specification with the following requirements:

TARGET SPECIFICATIONS:
- Primary Category: {category}
- Distractor Category: {distractor_category if applicable}
- Persistence Level: {long/medium/short}
- Emotional Tone: {neutral/excited/frustrated/collaborative}
- Turn Count: {4-10}
- Special Requirements: {e.g., "include specific brand voice example", "multi-label with user preference"}

OUTPUT FORMAT:
Return a JSON object with:
{
  "scenario_description": "Brief narrative setup (2-3 sentences)",
  "user_profile": "User role and context",
  "key_signals_to_include": ["List of 2-4 specific memory-worthy signals"],
  "distractor_signals": ["Optional list of noise/irrelevant info"],
  "suggested_turn_breakdown": "How the conversation should flow"
}

EXAMPLE OUTPUT:
{
  "scenario_description": "Marketing director discussing their personal communication preferences while reviewing campaign performance. They reveal tone expectations and decision-making style.",
  "user_profile": "Senior Marketing Director at B2B SaaS company, prefers data-driven discussions, values conciseness",
  "key_signals_to_include": [
    "Explicit statement about preferring bullet points over paragraphs",
    "Request for 'bottom-line-up-front' approach",
    "Mention of quarterly review cadence"
  ],
  "distractor_signals": [
    "Transactional question about meeting time",
    "Small talk about weather"
  ],
  "suggested_turn_breakdown": "Start with campaign review (business_priorities), transition to feedback on communication style (user.communication_style), end with scheduling (none)"
}

Generate a scenario for: {TARGET SPECIFICATIONS}
```

### Stage 2: Conversation Generation Prompt

Use this prompt with GPT-5 (or Claude) to generate the actual conversation based on a scenario spec.

```
You are generating realistic marketing conversations between a user and an AI marketing assistant. Generate natural dialogue that contains specific information worth storing in long-term memory.

CONTEXT:
You will create a conversation that exemplifies certain memory categories while maintaining realism and natural flow.

SCENARIO SPECIFICATION:
{Insert scenario_spec from Stage 1}

MEMORY TAXONOMY (for reference):
COMPANY MEMORY:
- company.brand_core: Voice, values, positioning, identity anchors (Persistence: Long >1y)
- company.strategic_signatures: Decision frameworks, strategic heuristics (Persistence: Long >1y)
- company.knowledge_artifacts: Docs, style guides, playbooks (Persistence: Long >1y)
- company.business_priorities: Quarterly/seasonal goals, active campaigns (Persistence: Short <3m)
- company.tools_config: Integrations, API keys, workflow settings (Persistence: Medium ~6m)
- company.performance_context: Campaign metrics, retrospectives, learnings (Persistence: Rolling ~6m)

USER MEMORY:
- user.communication_style: Tone, verbosity, format expectations (Persistence: Long >1y)
- user.strategic_approach: Personal priorities, success definitions (Persistence: Long >1y)
- user.role_context: Title, scope, decision authority (Persistence: Medium ~1y)
- user.workflow_patterns: Review cadence, collaboration norms (Persistence: Medium ~1y)
- user.session_history: Immediate context, recent asks (Persistence: Short <2w)
- user.interaction_preferences: Coaching style, feedback expectations (Persistence: Evolving)

SPECIAL:
- none: Irrelevant, vague, or transactional content

GENERATION RULES:
1. Make conversations feel natural - include some filler, transitions, acknowledgments
2. Embed memory-worthy information organically (don't make it too obvious)
3. Include 1-2 utterances that should map to "none" for realism
4. If multi-label scenario, ensure signals for both categories are present
5. Length: {turn_count} turns (alternating user/assistant)
6. Include specific, concrete details (not generic statements)
7. For company.* categories: use "we", "our company", "our brand"
8. For user.* categories: use "I prefer", "my approach", "I typically"

OUTPUT FORMAT:
Return a JSON object with:
{
  "scenario_id": "{primary_category}_{scenario_type}_{random_3_digit_number}",
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ],
  "labels": {
    "categories": ["array of applicable categories"],
    "persistence_horizon": "long|medium|short",
    "memory_scope": "company|user|mixed|none",
    "rationale": "1-2 sentence explanation of category choices"
  },
  "metadata": {
    "scenario_type": "descriptive_label",
    "primary_category": "main_category",
    "distractor_present": true|false,
    "turn_count": integer,
    "signals_present": ["list of specific signals included"]
  }
}

CRITICAL: Respond with ONLY the JSON object. No markdown formatting, no explanation, no preamble.

Generate the conversation now.
```

---

## Example Synthetic Data Instances

### Example 1: Single-Label, company.brand_core

```json
{
  "scenario_id": "brand_core_voice_definition_047",
  "conversation": [
    {
      "role": "user",
      "content": "I'm working on our new product launch messaging and want to make sure we're staying true to our brand voice. Can you remind me what our core voice attributes are?"
    },
    {
      "role": "assistant",
      "content": "I'd be happy to help! Let me check our brand guidelines. What specific aspect of the launch are you working on?"
    },
    {
      "role": "user",
      "content": "The hero section copy. We've always positioned ourselves as the approachable expert - professional but not corporate, if that makes sense. Our voice should be confident without being arrogant, and we use contractions to sound human."
    },
    {
      "role": "assistant",
      "content": "That's perfect context. So you're looking for that balance of expertise and warmth. Would you like me to review a draft with those voice principles in mind?"
    },
    {
      "role": "user",
      "content": "Yes, that would be great. Also, I want to make sure we avoid jargon - that's been a key part of our brand since day one. We explain complex marketing concepts in plain English."
    },
    {
      "role": "assistant",
      "content": "Absolutely, accessibility through clear language is core to your brand identity. Share the draft whenever you're ready and I'll evaluate it against those voice principles."
    }
  ],
  "labels": {
    "categories": ["company.brand_core"],
    "persistence_horizon": "long",
    "memory_scope": "company",
    "rationale": "User explicitly defines multiple core brand voice attributes (approachable expert, contractions, jargon-free) that represent enduring identity anchors."
  },
  "metadata": {
    "scenario_type": "brand_voice_definition",
    "primary_category": "company.brand_core",
    "distractor_present": false,
    "turn_count": 6,
    "signals_present": [
      "voice_attribute_approachable_expert",
      "voice_attribute_contractions",
      "voice_principle_no_jargon",
      "positioning_statement"
    ]
  }
}
```

### Example 2: Multi-Label, company.business_priorities + user.communication_style

```json
{
  "scenario_id": "mixed_priority_communication_183",
  "conversation": [
    {
      "role": "user",
      "content": "Quick update on Q4 - our north star metric is increasing trial-to-paid conversion by 15%. Everything else is secondary."
    },
    {
      "role": "assistant",
      "content": "Got it, trial-to-paid conversion is the priority. What initiatives are you planning to support that goal?"
    },
    {
      "role": "user",
      "content": "Three main campaigns: onboarding email sequence optimization, in-app value demonstration, and sales-assist for high-intent trials. When you give me updates on these, I need the data upfront - show me the numbers first, then the narrative. I don't have time for long explanations before I see performance."
    },
    {
      "role": "assistant",
      "content": "Understood - metrics first, context second. Should I track these three campaigns as your active Q4 priorities?"
    },
    {
      "role": "user",
      "content": "Yes, and one more thing - if any campaign is underperforming, just tell me straight. I value directness over softening bad news."
    }
  ],
  "labels": {
    "categories": ["company.business_priorities", "user.communication_style"],
    "persistence_horizon": "mixed",
    "memory_scope": "mixed",
    "rationale": "Contains Q4 business goal and active campaigns (short-term, company-level) plus explicit communication preferences about data-first reporting and directness (long-term, user-level)."
  },
  "metadata": {
    "scenario_type": "priority_setting_with_communication_preference",
    "primary_category": "company.business_priorities",
    "distractor_present": false,
    "turn_count": 5,
    "signals_present": [
      "quarterly_goal_conversion",
      "active_campaigns_list",
      "communication_pref_data_first",
      "communication_pref_directness"
    ]
  }
}
```

### Example 3: None Category (Transactional)

```json
{
  "scenario_id": "none_transactional_312",
  "conversation": [
    {
      "role": "user",
      "content": "What time is it in New York right now?"
    },
    {
      "role": "assistant",
      "content": "It's currently 2:34 PM EST in New York."
    },
    {
      "role": "user",
      "content": "Thanks. Can you find that report I asked for earlier?"
    },
    {
      "role": "assistant",
      "content": "I don't see a previous request for a report in our conversation. Could you clarify which report you're looking for?"
    },
    {
      "role": "user",
      "content": "Never mind, I think I was talking to someone else about it."
    }
  ],
  "labels": {
    "categories": ["none"],
    "persistence_horizon": "short",
    "memory_scope": "none",
    "rationale": "Purely transactional queries with no durable information worth storing. Time query and confused reference have no strategic or personal value."
  },
  "metadata": {
    "scenario_type": "transactional_noise",
    "primary_category": "none",
    "distractor_present": false,
    "turn_count": 5,
    "signals_present": ["time_query", "confused_context"]
  }
}
```

### Example 4: Multi-Label with Distractor, user.strategic_approach + user.workflow_patterns + none

```json
{
  "scenario_id": "user_strategic_workflow_mixed_089",
  "conversation": [
    {
      "role": "user",
      "content": "Before we dive into campaign planning, I need to set some expectations about how I work. I'm a big believer in testing small before scaling - I'd rather run five $1k experiments than one $5k campaign until we find what works."
    },
    {
      "role": "assistant",
      "content": "That makes sense - iterative testing to de-risk before investment. How do you typically structure these experiments?"
    },
    {
      "role": "user",
      "content": "I run them for two weeks minimum, review results every Friday afternoon, and I need at least 3x ROAS to consider something scalable. Also, completely unrelated, but is it supposed to rain tomorrow?"
    },
    {
      "role": "assistant",
      "content": "Let me check the weather... Yes, there's a 70% chance of rain tomorrow afternoon. Going back to your testing framework - do you involve your team in the Friday reviews?"
    },
    {
      "role": "user",
      "content": "Always. I do weekly reviews with the full marketing team, and I expect everyone to come prepared with their numbers. Collaboration is key to my approach - I don't make scaling decisions alone."
    },
    {
      "role": "assistant",
      "content": "Got it - test-and-learn methodology with collaborative weekly reviews every Friday. I'll keep that workflow in mind when making recommendations."
    }
  ],
  "labels": {
    "categories": ["user.strategic_approach", "user.workflow_patterns"],
    "persistence_horizon": "long",
    "memory_scope": "user",
    "rationale": "Defines personal strategic philosophy (test small before scaling, 3x ROAS threshold) and recurring workflow pattern (Friday reviews with team). Weather question is noise."
  },
  "metadata": {
    "scenario_type": "strategic_workflow_definition_with_distractor",
    "primary_category": "user.strategic_approach",
    "distractor_present": true,
    "turn_count": 6,
    "signals_present": [
      "strategic_approach_test_before_scale",
      "success_metric_3x_roas",
      "workflow_friday_reviews",
      "workflow_collaborative_decisions",
      "distractor_weather"
    ]
  }
}
```

### Example 5: Edge Case - Long-term company.knowledge_artifacts

```json
{
  "scenario_id": "knowledge_artifacts_style_guide_221",
  "conversation": [
    {
      "role": "user",
      "content": "I just updated our email marketing style guide - it's in the shared drive under Marketing/Guidelines/Email_Style_Guide_v3.pdf. This supersedes the old version from 2023."
    },
    {
      "role": "assistant",
      "content": "Thanks for letting me know. What are the major changes in version 3?"
    },
    {
      "role": "user",
      "content": "Main updates: we now use sentence case for subject lines instead of title case, maximum 50 characters. Email body should be 150 words max, and we've added a new section on accessibility requirements - minimum 14pt font, 1.5 line spacing, alt text mandatory for all images."
    },
    {
      "role": "assistant",
      "content": "Those are significant changes, especially the accessibility requirements. Should I reference this guide when reviewing email campaigns?"
    },
    {
      "role": "user",
      "content": "Yes, absolutely. This is now the canonical source for all email creative. If anyone asks about email standards, point them to this document."
    }
  ],
  "labels": {
    "categories": ["company.knowledge_artifacts"],
    "persistence_horizon": "long",
    "memory_scope": "company",
    "rationale": "Introduction of updated canonical style guide with specific location and new standards. This is a durable knowledge artifact that will be referenced repeatedly."
  },
  "metadata": {
    "scenario_type": "knowledge_artifact_update",
    "primary_category": "company.knowledge_artifacts",
    "distractor_present": false,
    "turn_count": 5,
    "signals_present": [
      "document_location",
      "canonical_source_declaration",
      "specific_guidelines_subject_line",
      "specific_guidelines_accessibility"
    ]
  }
}
```

---

## Generation Strategy & Best Practices

## Preparing Data for Tinker Training

Use the official Tinker renderer utilities to transform the JSONL data into `types.Datum` objects before SFT/RL runs. This ensures the tokenizer, stop sequences, and weight masks match what the trainer expects ([`renderers.build_supervised_example`](tinker_docs.md#file-renderingmdx) and [`types.Datum`](tinker_docs.md#part-2-type-definitions)).

```python
import json
import tinker
from tinker import types
from tinker_cookbook import renderers, tokenizer_utils

tokenizer = tokenizer_utils.get_tokenizer("meta-llama/Llama-3.1-8B")
renderer = renderers.get_renderer(name="llama3", tokenizer=tokenizer)

def conversation_to_datum(conversation_json: dict) -> types.Datum:
    tokens, weights = renderer.build_supervised_example(conversation_json["conversation"])
    model_input = types.ModelInput.from_ints(tokens[:-1])
    datum = types.Datum(
        model_input=model_input,
        loss_fn_inputs=dict(
            target_tokens=tokens[1:],
            weights=weights[1:],
        ),
    )
    if datum.model_input.length > 4096:
        raise ValueError("Conversation exceeds model context window")
    return datum
```

**Checklist**
- [ ] Conversations tokenized with the same renderer used during training
- [ ] Resulting `ModelInput` length < base model context window (4k for Llama-3.1-8B)
- [ ] Non-zero loss weights present (otherwise drop example)
- [ ] Saved as pickled or JSONL `Datum` payloads ready for `forward_backward_async`

### Coverage Matrix

Generate scenarios to ensure balanced coverage:

| Category | Target % | Min Examples | With Distractor | Multi-Label |
|----------|----------|--------------|-----------------|-------------|
| company.brand_core | 10% | 100 | 30 | 20 |
| company.strategic_signatures | 8% | 80 | 25 | 15 |
| company.knowledge_artifacts | 8% | 80 | 25 | 15 |
| company.business_priorities | 10% | 100 | 40 | 30 |
| company.tools_config | 7% | 70 | 20 | 10 |
| company.performance_context | 9% | 90 | 30 | 20 |
| user.communication_style | 10% | 100 | 30 | 25 |
| user.strategic_approach | 9% | 90 | 25 | 20 |
| user.role_context | 7% | 70 | 20 | 15 |
| user.workflow_patterns | 8% | 80 | 25 | 20 |
| user.session_history | 6% | 60 | 15 | 10 |
| user.interaction_preferences | 8% | 80 | 25 | 20 |
| none | 10% | 100 | 50 | 5 |

**Total Target:** 1,100-1,200 examples minimum for SFT

### Quality Validation Checklist

For each generated example, validate:

**Structural:**
- [ ] Valid JSON format
- [ ] All required fields present
- [ ] 4-10 turns in conversation
- [ ] Alternating roles (mostly)
- [ ] Categories are from valid taxonomy

**Content:**
- [ ] Natural language flow (not robotic)
- [ ] Specific details present (not generic)
- [ ] Clear signal for each labeled category
- [ ] Distractor is truly off-topic (if present)
- [ ] Persistence horizon matches category definition

**Label Quality:**
- [ ] Rationale explains category choice
- [ ] Multi-label examples have signals for all categories
- [ ] "none" examples have no memorable information
- [ ] Memory scope matches categories (company.* → company)

### Batch Generation Process

1. **Define Coverage Plan** 
   - Decide on total dataset size (1,500-2,000 recommended)
   - Allocate examples per category per coverage matrix
   - Generate scenario specifications for each category

2. **Generate Conversations** 
   - Process scenarios in batches of 50-100
   - Use GPT-5 or Claude Opus for generation
   - Temperature: 0.7 for diversity
   - Validate each batch before proceeding

3. **Quality Review** 
   - Sample 100 random examples for human review
   - Check for common failure modes:
     * Generic statements ("our brand is innovative")
     * Unclear signals (ambiguous category)
     * Unrealistic dialogue (too formal/robotic)
     * Missing distractors where planned
   - Iterate prompts if quality issues found

4. **Teacher Labeling** 
   - Run ALL examples through teacher labeling prompt
   - Compare teacher labels to synthetic labels
   - Agreement threshold: >95%
   - If disagreement, review and regenerate

5. **Final Dataset Assembly**
   - Split train/test (80/20)
   - Stratify by category to ensure test coverage
   - Save as `train.jsonl` and `test.jsonl`
   - Document metadata (generation date, model used, prompts)

---

## Teacher Labeling Prompt

Use this prompt to generate gold labels for any conversation (including real production data later):

```
You are a memory routing classifier for a marketing AI system. Your job is to analyze conversations and determine what information should be stored in long-term memory and in which categories.

MEMORY TAXONOMY:

COMPANY MEMORY (about the organization):
1. company.brand_core - Voice, values, positioning, identity anchors [Long-term: >1y]
2. company.strategic_signatures - Decision frameworks, strategic heuristics [Long-term: >1y]
3. company.knowledge_artifacts - Documents, style guides, playbooks [Long-term: >1y]
4. company.business_priorities - Quarterly/seasonal goals, active campaigns [Short-term: <3m]
5. company.tools_config - Integrations, API keys, workflow settings [Medium-term: ~6m]
6. company.performance_context - Campaign metrics, retrospectives, learnings [Rolling: ~6m]

USER MEMORY (about the individual):
7. user.communication_style - Tone, verbosity, format expectations [Long-term: >1y]
8. user.strategic_approach - Personal priorities, success definitions [Long-term: >1y]
9. user.role_context - Title, scope, decision authority [Medium-term: ~1y]
10. user.workflow_patterns - Review cadence, collaboration norms [Medium-term: ~1y]
11. user.session_history - Immediate context, recent asks [Short-term: <2w]
12. user.interaction_preferences - Coaching style, feedback expectations [Evolving]

SPECIAL:
13. none - Irrelevant, vague, or transactional content (use when nothing is worth remembering)

ROUTING RULES:
1. Distinguish company.* (organization-level) from user.* (individual-level)
2. Match persistence horizon to information lifetime
3. Predict ≤3 categories unless strictly necessary
4. Prefer "none" unless there's CONCRETE, DURABLE information
5. For company.* categories: look for "we", "our", organizational facts
6. For user.* categories: look for "I", "my", personal preferences
7. Vague statements like "we should be innovative" → none (too generic)
8. Specific statements like "our brand voice uses contractions" → company.brand_core

CONVERSATION TO ANALYZE:
{conversation}

OUTPUT FORMAT (JSON):
{
  "categories": ["category1", "category2"],
  "persistence_horizon": "long|medium|short",
  "memory_scope": "company|user|mixed|none",
  "rationale": "Brief explanation of why these categories were selected",
  "confidence": "high|medium|low",
  "extractable_facts": [
    "List 1-3 specific facts that would be stored in memory"
  ]
}

CRITICAL RULES:
- Be conservative with labeling - when in doubt, use "none"
- Only label what's EXPLICITLY stated, not implied
- Multi-label only when multiple distinct types of information are present
- If conversation is small talk or transactional → ["none"]

Analyze the conversation and provide your classification:
```

---

## Validation & Metrics

### Data Quality Metrics to Track

**Coverage Metrics:**
- Category distribution (should match target ±5%)
- Persistence distribution (long: 35%, medium: 30%, short: 25%, mixed: 10%)
- Memory scope distribution (company: 45%, user: 45%, mixed: 5%, none: 5%)
- Multi-label frequency (target: 20-25% of non-none examples)

**Quality Metrics:**
- Teacher agreement rate (target: >95%)
- Average turn length (target: 20-150 tokens)
- Conversation length distribution (target: mean 6.5 ± 1.5 turns)
- "none" precision via human review (target: >90%)

**Signal Metrics:**
- Average signals per conversation (target: 2-3)
- Signal diversity (unique signal types / total signals: target >0.7)
- Distractor effectiveness (human annotators can identify: target >85%)

### Automated Validation Script

```python
import json
from collections import Counter
from typing import Dict, List

def validate_synthetic_data(filepath: str) -> Dict[str, any]:
    """Validate synthetic data quality"""
    
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Category distribution
    all_categories = []
    for item in data:
        all_categories.extend(item['labels']['categories'])
    category_dist = Counter(all_categories)
    
    # Multi-label frequency
    multi_label_count = sum(1 for item in data if len(item['labels']['categories']) > 1)
    multi_label_freq = multi_label_count / len(data)
    
    # Turn count distribution
    turn_counts = [item['metadata']['turn_count'] for item in data]
    avg_turns = sum(turn_counts) / len(turn_counts)
    
    # Persistence distribution
    persistence_dist = Counter(item['labels']['persistence_horizon'] for item in data)
    
    # Memory scope distribution
    scope_dist = Counter(item['labels']['memory_scope'] for item in data)
    
    return {
        'total_examples': len(data),
        'category_distribution': dict(category_dist),
        'multi_label_frequency': multi_label_freq,
        'avg_turns_per_conversation': avg_turns,
        'persistence_distribution': dict(persistence_dist),
        'scope_distribution': dict(scope_dist),
        'warnings': _generate_warnings(category_dist, multi_label_freq, avg_turns)
    }

def _generate_warnings(cat_dist, ml_freq, avg_turns):
    warnings = []
    
    # Check for imbalanced categories
    total = sum(cat_dist.values())
    for cat, count in cat_dist.items():
        if count / total < 0.05:
            warnings.append(f"Category '{cat}' underrepresented: {count/total:.1%}")
    
    # Check multi-label frequency
    if ml_freq < 0.15:
        warnings.append(f"Low multi-label frequency: {ml_freq:.1%} (target: 20-25%)")
    
    # Check turn length
    if avg_turns < 5 or avg_turns > 8:
        warnings.append(f"Average turns out of range: {avg_turns:.1f} (target: 6.5±1.5)")
    
    return warnings

# Usage
metrics = validate_synthetic_data('train.jsonl')
print(json.dumps(metrics, indent=2))
```

---

## Prompt Engineering Tips

### For Better Diversity:
1. Set the `temperature` parameter (1)
2. Add "Generate a DIFFERENT conversation that..." to avoid repetition
3. Provide counter-examples: "Don't make it like this: [generic example]"
4. Use different starting phrases: "Create", "Generate", "Produce", "Design"

### For Better Quality:
1. Include GOOD and BAD examples in prompt
2. Specify "Be specific, not generic" multiple times
3. Add negative instructions: "Avoid phrases like 'let's think about', 'going forward'"
4. Request concrete numbers, names, specific tools/platforms

### For Better Label Accuracy:
1. Show the model 3-5 labeled examples before asking for labels
2. Use chain-of-thought: "First, identify the durable facts. Then, assign categories."
3. Add calibration: "Be conservative - most conversations should have 1-2 categories, not 4+"
4. Include edge cases in few-shot examples (generic vs specific, multi-label)

---

## Common Pitfalls & How to Avoid Them

### Pitfall 1: Generic, Vague Content
**Bad Example:** "We value innovation and customer focus."
**Good Example:** "We use contractions in all customer-facing copy to sound conversational, and we never say 'utilize' when 'use' works fine."

**Fix:** Add to prompt: "Include SPECIFIC details like exact phrases, numbers, tool names, or concrete examples."

### Pitfall 2: Over-labeling
**Bad:** Every conversation gets 3-4 categories
**Good:** Most conversations get 1-2 categories, some get 0 (none)

**Fix:** Emphasize in teacher prompt: "Be conservative. Most conversations are noise or context-specific."

### Pitfall 3: Unrealistic Dialogue
**Bad:** "Hello, I would like to discuss our brand positioning strategy and establish core value propositions."
**Good:** "Hey, quick question about our brand voice - are we still doing the contractions thing in emails?"

**Fix:** Add natural language examples and specify: "Make it sound like a real conversation, including filler words, casual language, and natural transitions."

### Pitfall 4: Missing Edge Cases
**Bad:** Every example is clean and obvious
**Good:** 20% have ambiguity, distractors, or edge cases

**Fix:** Explicitly generate "hard negative" scenarios: near-misses, multi-label, heavy distractors

### Pitfall 5: Persistence Mismatch
**Bad:** "Our Q1 campaign goal" labeled as long-term
**Good:** "Our Q1 campaign goal" labeled as short-term (company.business_priorities)

**Fix:** Include persistence definitions in EVERY prompt and validate programmatically

---

## Summary Checklist

Before finalizing your synthetic dataset:

- [ ] 1,500-2,000 total examples generated
- [ ] All 13 categories represented (100+ examples each)
- [ ] 20-25% multi-label examples
- [ ] 10-15% "none" examples
- [ ] 30% with intentional distractors
- [ ] Teacher labeling agreement >95%
- [ ] Average 6.5 ± 1.5 turns per conversation
- [ ] Persistence distribution: ~35% long, ~30% medium, ~25% short
- [ ] Human review of 100 random samples (quality check)
- [ ] Train/test split (80/20, stratified by category)
- [ ] Documentation of generation process and prompts saved