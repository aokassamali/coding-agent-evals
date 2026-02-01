# This script checks whether different run folders produced literally identical behavior.
# It hashes per-task outcomes using step logs (tests + code) and compares across run dirs.

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import tyro


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def task_signature(steps: List[Dict[str, Any]]) -> str:
    """
    Hash the sequence of:
      - test exit codes
      - code strings for code_fix/code_propose/code_edit
    This is robust enough to detect "same outputs" even if run_ids differ.
    """
    parts: List[str] = []
    for s in sorted(steps, key=lambda r: int(r.get("step_id", 0))):
        st = s.get("step_type")
        if st == "test":
            parts.append(f"T:{s.get('exit_code')}")
        elif st in ("code_propose", "code_edit", "code_fix", "code_fix_rejected"):
            code = (s.get("code") or "").strip()
            parts.append(f"C:{st}:{code}")
    blob = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class Args:
    run_dirs: List[str]


def main(args: Args) -> None:
    per_dir: Dict[str, Dict[str, str]] = {}  # dir -> task_id -> sig

    for d in args.run_dirs:
        run_dir = Path(d)
        steps = read_jsonl(run_dir / "fact_step.jsonl")
        by_run: Dict[str, List[Dict[str, Any]]] = {}
        for s in steps:
            rid = s.get("run_id")
            if rid:
                by_run.setdefault(str(rid), []).append(s)

        runs = read_jsonl(run_dir / "fact_run.jsonl")
        task_sigs: Dict[str, str] = {}
        for r in runs:
            rid = str(r["run_id"])
            task_id = str(r["task_id"])
            sig = task_signature(by_run.get(rid, []))
            task_sigs[task_id] = sig

        per_dir[run_dir.name] = task_sigs

    names = list(per_dir.keys())
    if len(names) < 2:
        print("Need at least 2 run dirs.")
        return

    base = names[0]
    base_map = per_dir[base]
    for other in names[1:]:
        other_map = per_dir[other]
        common = sorted(set(base_map.keys()) & set(other_map.keys()))
        if not common:
            print(f"{base} vs {other}: no common task_ids")
            continue
        same = sum(1 for t in common if base_map[t] == other_map[t])
        print(f"{base} vs {other}: identical task signatures = {same}/{len(common)} ({100.0*same/len(common):.1f}%)")

    print("\nTip: if you see ~100% identical, you’re not getting meaningful model variation (or runs got reused).")


if __name__ == "__main__":
    main(tyro.cli(Args))
