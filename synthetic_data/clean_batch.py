import json
import sys

def clean_batch(filepath):
    print(f"Cleaning {filepath}...")
    cleaned_data = []
    fixed_count = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            cats = item['labels']['categories']
            
            if 'none' in cats and len(cats) > 1:
                print(f"Fixing mixed 'none' in {item['scenario_id']}: {cats}")
                cats.remove('none')
                item['labels']['categories'] = cats
                item['metadata']['cleaned_none_mix'] = True
                fixed_count += 1
            
            cleaned_data.append(item)
    
    output_path = filepath.replace('.jsonl', '_cleaned.jsonl')
    with open(output_path, 'w') as f:
        for item in cleaned_data:
            f.write(json.dumps(item) + '\n')
            
    print(f"Cleaned {len(cleaned_data)} items. Fixed {fixed_count} issues.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 clean_batch.py <jsonl_file>")
        sys.exit(1)
    clean_batch(sys.argv[1])

