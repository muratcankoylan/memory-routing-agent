import json
from synthetic_data.pipeline import SyntheticDataPipeline

def generate_sample():
    pipeline = SyntheticDataPipeline()
    print("Generating sample batch...")
    results = pipeline.run_batch(count=2, category="company.brand_core")
    
    with open("synthetic_data/sample_batch.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} examples to synthetic_data/sample_batch.json")

if __name__ == "__main__":
    generate_sample()

