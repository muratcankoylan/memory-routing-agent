import json
import argparse
from collections import Counter
from typing import Dict, List, Any

def validate_synthetic_data(filepath: str) -> Dict[str, Any]:
    """Validate synthetic data quality based on the PRD guidelines."""
    
    try:
        with open(filepath, 'r') as f:
            # Handle both single JSON array and JSONL formats
            content = f.read().strip()
            if content.startswith('[') and content.endswith(']'):
                data = json.loads(content)
            else:
                data = [json.loads(line) for line in content.split('\n') if line.strip()]
    except json.JSONDecodeError as e:
        return {'error': f"Invalid JSON format: {e}"}
    except Exception as e:
        return {'error': f"Error reading file: {e}"}
    
    if not data:
        return {'error': "Empty dataset"}

    # Category distribution
    all_categories = []
    for item in data:
        if 'labels' in item and 'categories' in item['labels']:
            all_categories.extend(item['labels']['categories'])
    category_dist = Counter(all_categories)
    
    # Multi-label frequency
    multi_label_count = sum(1 for item in data 
                          if 'labels' in item and 'categories' in item['labels'] 
                          and len(item['labels']['categories']) > 1)
    multi_label_freq = multi_label_count / len(data) if len(data) > 0 else 0
    
    # Turn count distribution
    turn_counts = [item['metadata'].get('turn_count', 0) for item in data if 'metadata' in item]
    avg_turns = sum(turn_counts) / len(turn_counts) if turn_counts else 0
    
    # Persistence distribution
    persistence_dist = Counter(item['labels'].get('persistence_horizon', 'unknown') for item in data if 'labels' in item)
    
    # Memory scope distribution
    scope_dist = Counter(item['labels'].get('memory_scope', 'unknown') for item in data if 'labels' in item)
    
    return {
        'total_examples': len(data),
        'category_distribution': dict(category_dist),
        'multi_label_frequency': multi_label_freq,
        'avg_turns_per_conversation': avg_turns,
        'persistence_distribution': dict(persistence_dist),
        'scope_distribution': dict(scope_dist),
        'warnings': _generate_warnings(category_dist, multi_label_freq, avg_turns, len(data))
    }

def _generate_warnings(cat_dist, ml_freq, avg_turns, total_count):
    warnings = []
    
    # Check for imbalanced categories (only if dataset is large enough)
    if total_count > 20:
        total_cats = sum(cat_dist.values())
        for cat, count in cat_dist.items():
            if count / total_cats < 0.05:
                warnings.append(f"Category '{cat}' underrepresented: {count/total_cats:.1%}")
    
    # Check multi-label frequency
    if ml_freq < 0.15:
        warnings.append(f"Low multi-label frequency: {ml_freq:.1%} (target: 20-25%)")
    
    # Check turn length
    if avg_turns < 4 or avg_turns > 10:
        warnings.append(f"Average turns out of range: {avg_turns:.1f} (target: 6.5±1.5)")
    
    return warnings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate synthetic data quality")
    parser.add_argument("filepath", help="Path to JSON/JSONL file")
    args = parser.parse_args()
    
    metrics = validate_synthetic_data(args.filepath)
    print(json.dumps(metrics, indent=2))

