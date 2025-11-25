# Synthetic Data Generation Pipeline

This directory contains the tools for generating and validating synthetic training data using Cohere's `command-a-reasoning-08-2025` model.

## Setup

1.  **Install Dependencies**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install cohere python-dotenv tinker tinker-cookbook
    ```

2.  **Environment Variables**:
    Ensure your `.env` file contains your Cohere API key:
    ```
    COHERE_API_KEY=your_api_key_here
    ```

## Usage

### 1. Generate Data
Use the `SyntheticDataPipeline` class to generate data batches.

```python
from synthetic_data.pipeline import SyntheticDataPipeline

pipeline = SyntheticDataPipeline()
# Generate 10 examples for a specific category
results = pipeline.run_batch(count=10, category="company.brand_core")
```

You can also run the sample generator script:
```bash
python3 synthetic_data/generate_sample.py
```

### 2. Validate Data
Run the validation script on any generated JSON or JSONL file to check compliance with the schema and distribution targets.

```bash
python3 synthetic_data/validate.py synthetic_data/sample_batch.json
```

The validator checks:
*   JSON structure and required fields
*   Category distribution
*   Multi-label frequency
*   Conversation length
*   Persistence and scope consistency

## Pipeline Components

*   `pipeline.py`: Core logic for 2-stage generation (Scenario -> Conversation) using Cohere.
*   `validate.py`: Quality assurance script implementing checks from `docs/synthetic_data.md`.
*   `test_pipeline.py`: Unit tests for the pipeline structure.
*   `generate_sample.py`: Helper script to produce a quick sample batch.

