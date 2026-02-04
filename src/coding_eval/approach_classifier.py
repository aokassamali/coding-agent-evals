# approach_classifier.py
"""
Thin wrapper around shared approach detection for runner metadata.
"""

from typing import Optional

from approach_detection import detect_approach


def classify_approach(code: str, task_id: str, topic: Optional[str] = None) -> Optional[str]:
    """
    Detect which architectural approach the code uses based on task/topic and code patterns.
    Returns approach name or None if unclassifiable.
    """
    return detect_approach(code, task_id=task_id, topic=topic)


if __name__ == "__main__":
    test_cases = [
        ("osc_queue_001", "from collections import deque\nself.items = deque()", "collections_deque"),
        ("osc_queue_001", "self.items = []\nself.head = 0", "list_with_indices"),
        ("osc_cache_001", "from collections import OrderedDict", "ordereddict"),
        ("osc_search_001", "mid = (low + high) // 2", "binary_iterative"),
    ]

    for task_id, code, expected in test_cases:
        result = classify_approach(code, task_id)
        status = "OK" if result == expected else "BAD"
        print(f"{status} {task_id}: expected={expected}, got={result}")
