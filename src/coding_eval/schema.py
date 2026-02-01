# What this module does:
# Define small, explicit data structures for run-level and step-level logs,
# so we can save traces consistently and compute metrics later.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import json
import time
import uuid


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class StepLog:
    run_id: str
    step_id: int
    step_type: str  # e.g., "code", "test", "reflect"
    prompt: Optional[str]
    response: Optional[str]
    code: Optional[str]
    stdout: Optional[str]
    stderr: Optional[str]
    exit_code: Optional[int]
    started_ms: int
    ended_ms: int
    meta: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class RunLog:
    run_id: str
    task_id: str
    variant_id: str
    model_id: str
    success: int
    steps: int
    started_ms: int
    ended_ms: int
    failure_mode: Optional[str]
    category: Optional[str] = None
    topic: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

