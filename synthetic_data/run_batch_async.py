import json
import random
import time
import sys
import asyncio
import os
from typing import List, Dict, Any
from synthetic_data.pipeline import SyntheticDataPipeline
from synthetic_data.validate import validate_synthetic_data
from synthetic_data.clean_data import clean_datum

CATEGORY_DISTRIBUTION = {
    "none": 0.15,
    "user.interaction_preferences": 0.12,
    "user.session_history": 0.10,
    "company.brand_core": 0.10,
    
    "company.strategic_signatures": 0.07,
    "company.knowledge_artifacts": 0.07,
    "user.communication_style": 0.07,
    "user.strategic_approach": 0.07,
    "user.workflow_patterns": 0.07,
    
    "company.tools_config": 0.05,
    "company.performance_context": 0.05,
    "company.business_priorities": 0.04,
    "user.role_context": 0.04
}

async def generate_single_item(pipeline: SyntheticDataPipeline, category: str, item_num: int) -> Dict[str, Any]:
    """Generate a single conversation item asynchronously."""
    print(f"  Starting item {item_num} (Target: {category})...")
    
    # Determine distractor
    categories = list(CATEGORY_DISTRIBUTION.keys())
    distractor = None
    if random.random() < 0.30 and category != "none":
        possible_distractors = [c for c in categories if c != category and c != "none"]
        if possible_distractors:
            distractor = random.choice(possible_distractors)
    
    persistence = _get_persistence_for_category(category)
    turns = random.randint(4, 10)
    
    # Generate scenario (synchronous call wrapped in executor)
    loop = asyncio.get_event_loop()
    scenario = await loop.run_in_executor(
        None,
        pipeline.generate_scenario_spec,
        category,
        distractor,
        persistence,
        "neutral",
        turns,
        ""
    )
    
    if not scenario:
        print(f"  Failed item {item_num}: scenario generation failed")
        return None
    
    # Generate conversation
    conversation = await loop.run_in_executor(
        None,
        pipeline.generate_conversation,
        scenario,
        turns,
        category
    )
    
    if conversation:
        # Clean the item immediately
        cleaned_conversation = clean_datum(conversation)
        print(f"  Completed item {item_num}: {cleaned_conversation.get('scenario_id', 'Unknown')}")
        return cleaned_conversation
    else:
        print(f"  Failed item {item_num}: conversation generation failed")
        return None

async def generate_batch_concurrent(pipeline: SyntheticDataPipeline, batch_size: int, batch_num: int) -> List[Dict[str, Any]]:
    """Generate a full batch of items concurrently, retrying until batch is full."""
    print(f"\n=== Processing Batch {batch_num} (Concurrent) ===")
    
    categories = list(CATEGORY_DISTRIBUTION.keys())
    weights = list(CATEGORY_DISTRIBUTION.values())
    
    batch_data = []
    items_needed = batch_size
    
    while items_needed > 0:
        # Select categories for this chunk of work
        batch_categories = random.choices(categories, weights=weights, k=items_needed)
        
        # Create tasks for needed items
        tasks = [
            generate_single_item(pipeline, category, len(batch_data) + i + 1)
            for i, category in enumerate(batch_categories)
        ]
        
        print(f"  Launch {items_needed} concurrent tasks...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successes
        success_count = 0
        for result in results:
            if isinstance(result, Exception):
                print(f"    Task exception: {result}")
            elif result is not None:
                batch_data.append(result)
                success_count += 1
        
        items_needed = batch_size - len(batch_data)
        if items_needed > 0:
            print(f"  Batch incomplete ({len(batch_data)}/{batch_size}). Retrying {items_needed} items in 5s...")
            await asyncio.sleep(5)
    
    print(f"Batch {batch_num} complete: {len(batch_data)}/{batch_size} items generated")
    return batch_data

async def run_pipeline_batches_async(total_items: int = 100, batch_size: int = 10):
    """Run the full pipeline with concurrent batch processing."""
    pipeline = SyntheticDataPipeline(max_retries=5)
    
    all_data = []
    num_batches = max(1, total_items // batch_size)
    
    print(f"Starting CONCURRENT generation of {total_items} items in {num_batches} batches...")
    print(f"Batch size: {batch_size} items (generated in parallel)")
    
    for batch_num in range(1, num_batches + 1):
        # Check if batch already exists
        batch_filename = f"synthetic_data/batch_{batch_num:02d}.jsonl"
        if os.path.exists(batch_filename):
            print(f"Batch {batch_num} already exists ({batch_filename}). Skipping generation...")
            # Load existing data to include in final output
            try:
                with open(batch_filename, 'r') as f:
                    for line in f:
                        if line.strip():
                            all_data.append(json.loads(line))
                print(f"Loaded {len(all_data)} items so far.")
                continue
            except Exception as e:
                print(f"Error reading existing batch {batch_num}: {e}. Regenerating...")
        
        # Generate entire batch concurrently
        batch_data = await generate_batch_concurrent(pipeline, batch_size, batch_num)
        
        # Save batch as JSONL
        with open(batch_filename, "w") as f:
            for item in batch_data:
                f.write(json.dumps(item) + "\n")
        print(f"Saved batch to {batch_filename}")
        
        # Validate batch
        print("Validating batch...")
        metrics = validate_synthetic_data(batch_filename)
        print(json.dumps(metrics, indent=2))
        
        all_data.extend(batch_data)
        
        # Wait 5 seconds before next batch
        if batch_num < num_batches:
            print("Waiting 5 seconds before next batch...")
            await asyncio.sleep(5)
    
    # Save all data
    output_file = f"synthetic_data/all_generated_data_{total_items}.jsonl"
    with open(output_file, "w") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"\n{'='*60}")
    print(f"COMPLETED: {len(all_data)} items generated")
    print(f"Full dataset saved to {output_file}")
    print(f"{'='*60}")

def _get_persistence_for_category(category: str) -> str:
    """Map category to its expected persistence level."""
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
    
    asyncio.run(run_pipeline_batches_async(total, batch))
