import json
from collections import defaultdict

# Change this to your actual task file path
TASK_FILE = "data/tasks/v8_osc_total_200.jsonl"

def analyze_tasks(filepath):
    topics = defaultdict(lambda: {'count': 0, 'approaches': set(), 'sample_code': None})
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            task = json.loads(line)
            topic = task.get('topic', 'unknown')
            approaches = task.get('approaches', [])
            starter_code = task.get('starter_code', '')
            
            topics[topic]['count'] += 1
            topics[topic]['approaches'].update(approaches)
            if topics[topic]['sample_code'] is None:
                topics[topic]['sample_code'] = starter_code[:500]
    
    print("=" * 80)
    print("TOPICS AND APPROACHES IN YOUR TASK SET")
    print("=" * 80)
    
    for topic in sorted(topics.keys()):
        info = topics[topic]
        print(f"\n### {topic} ({info['count']} tasks)")
        print(f"Approaches: {sorted(info['approaches'])}")
        print(f"Sample code snippet:")
        print("-" * 40)
        print(info['sample_code'][:300])
        print("-" * 40)
    
    # Also output just the raw data for easy copy-paste
    print("\n" + "=" * 80)
    print("RAW TOPIC -> APPROACHES MAPPING (copy this)")
    print("=" * 80)
    for topic in sorted(topics.keys()):
        approaches = sorted(topics[topic]['approaches'])
        print(f"'{topic}': {approaches},")

if __name__ == "__main__":
    analyze_tasks(TASK_FILE)