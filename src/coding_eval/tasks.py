from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class Task:
    task_id: str
    prompt: str
    signature: str
    tests: str
    starter_code: str = ""
    category: Optional[str] = None       # bugfix | stability_noop | hang_timeout | from_scratch
    topic: Optional[str] = None          # coarse tag like object_model, concurrency, etc.
    starter_check: Optional[str] = None  # pass | fail | timeout (informational)

def load_tasks_jsonl(path: str) -> List[Task]:
    tasks: List[Task] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            tasks.append(
                Task(
                    task_id=obj["task_id"],
                    prompt=obj["prompt"],
                    signature=obj["signature"],
                    tests=obj["tests"],
                    starter_code=obj.get("starter_code", "") or "",
                    category=obj.get("category"),
                    topic=obj.get("topic"),
                    starter_check=obj.get("starter_check"),
                )
            )
    return tasks
