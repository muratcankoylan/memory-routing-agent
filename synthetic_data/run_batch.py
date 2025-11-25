import json
import random
import time
import sys
from typing import List, Dict, Any
from synthetic_data.pipeline import SyntheticDataPipeline
from synthetic_data.validate import validate_synthetic_data

CATEGORY_DISTRIBUTION = {
    "company.brand_core": 0.10,
    "company.strategic_signatures": 0.08,
    "company.knowledge_artifacts": 0.08,
    "company.business_priorities": 0.10,
    "company.tools_config": 0.07,
    "company.performance_context": 0.09,
    "user.communication_style": 0.10,
    "user.strategic_approach": 0.09,
    "user.role_context": 0.07,
    "user.workflow_patterns": 0.08,
    "user.session_history": 0.06,
    "user.interaction_preferences": 0.08,
    "none": 0.10
}

def run_pipeline_batches(total_items: int = 100, batch_size: int = 10):
    pipeline = SyntheticDataPipeline()
    categories = list(CATEGORY_DISTRIBUTION.keys())
    weights = list(CATEGORY_DISTRIBUTION.values())
    
    all_data = []
    num_batches = max(1, total_items // batch_size)
    
    print(f"Starting generation of {total_items} items in {num_batches} batches (Size: {batch_size})...")

    for batch_num in range(1, num_batches + 1):
        print(f"\n=== Processing Batch {batch_num}/{num_batches} ===")
        batch_data = []
        
        while len(batch_data) < batch_size:
            category = random.choices(categories, weights=weights, k=1)[0]
            current_count = len(batch_data) + 1
            print(f"  Generating item {current_count}/{batch_size} (Category: {category})...")
            
            # Determine if we should add a distractor (30% chance)
            distractor = None
            if random.random() < 0.30 and category != "none":
                 possible_distractors = [c for c in categories if c != category and c != "none"]
                 if possible_distractors:
                     distractor = random.choice(possible_distractors)

            persistence = _get_persistence_for_category(category)
            turns = random.randint(4, 10)
            
            scenario = pipeline.generate_scenario_spec(
                category=category,
                distractor=distractor,
                persistence=persistence,
                turns=turns
            )
            
            if not scenario:
                print(f"    Failed to generate scenario for {category}. Retrying...")
                time.sleep(20)
                continue
                
            conversation = pipeline.generate_conversation(scenario, turn_count=turns)
            
            if conversation:
                batch_data.append(conversation)
                print(f"    Generated: {conversation.get('scenario_id', 'Unknown ID')}")
            else:
                 print(f"    Failed to generate conversation for {category}. Retrying...")
                 time.sleep(20)
                 continue
            
            print("    Sleeping for 15s to avoid rate limits...")
            time.sleep(15)
        
        # Save batch
        batch_filename = f"synthetic_data/batch_{batch_num:02d}.json"
        with open(batch_filename, "w") as f:
            json.dump(batch_data, f, indent=2)
        print(f"  Saved batch to {batch_filename}")
        
        # Validate batch
        print("  Validating batch...")
        metrics = validate_synthetic_data(batch_filename)
        print(json.dumps(metrics, indent=2))
        
        all_data.extend(batch_data)
        
    # Save all data
    with open("synthetic_data/all_generated_data_100.json", "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nCompleted. Total items generated: {len(all_data)}")
    print("Full dataset saved to synthetic_data/all_generated_data_100.json")

def _get_persistence_for_category(category: str) -> str:
    if "brand_core" in category or "strategic_signatures" in category or "knowledge_artifacts" in category or "communication_style" in category or "strategic_approach" in category:
        return "long"
    elif "tools_config" in category or "role_context" in category or "workflow_patterns" in category:
        return "medium"
    elif "business_priorities" in category or "session_history" in category:
        return "short"
    elif "performance_context" in category:
        return "rolling"
    elif "interaction_preferences" in category:
        return "evolving"
    elif "none" in category:
        return "short"
    return "medium" 

if __name__ == "__main__":
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run_pipeline_batches(total, batch)
