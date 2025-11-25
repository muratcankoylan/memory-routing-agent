import json
import sys
from typing import List, Dict, Any

def clean_datum(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean a single data item:
    1. Remove 'none' if other categories are present.
    2. Deduplicate categories.
    3. Ensure consistent formatting.
    """
    if "labels" not in item or "categories" not in item["labels"]:
        return item

    cats = item["labels"]["categories"]
    # Deduplicate
    cats = list(set(cats))
    
    # Remove 'none' if other categories exist
    if len(cats) > 1 and "none" in cats:
        cats.remove("none")
    
    # Update the item
    item["labels"]["categories"] = cats
    return item

def clean_file(input_path: str, output_path: str):
    print(f"Cleaning {input_path} -> {output_path}")
    cleaned_count = 0
    data = []
    
    # Read input
    with open(input_path, 'r') as f:
        content = f.read().strip()
        if not content:
            print("Empty file")
            return

        # Handle JSONL or list of JSON
        if content.startswith('[') and content.endswith(']'):
            raw_data = json.loads(content)
        else:
            raw_data = [json.loads(line) for line in content.split('\n') if line.strip()]
            
    # Process
    for item in raw_data:
        original_cats = item.get("labels", {}).get("categories", [])
        cleaned_item = clean_datum(item)
        new_cats = cleaned_item["labels"]["categories"]
        
        if set(original_cats) != set(new_cats):
            cleaned_count += 1
            
        data.append(cleaned_item)
        
    # Write output (always as JSONL for training)
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
            
    print(f"Processed {len(data)} items. Cleaned {cleaned_count} items (removed 'none' or duplicates).")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_data.py input_file [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.json', '_cleaned.jsonl').replace('.jsonl', '_cleaned.jsonl')
    
    clean_file(input_file, output_file)

