from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tyro


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _compute_data_fingerprint(runs: List[Dict[str, Any]]) -> str:
    import hashlib
    h = hashlib.sha1()
    rows = sorted(
        (r.get("run_id"), r.get("task_id"), r.get("model_id"), r.get("variant_id"), r.get("success"))
        for r in runs
    )
    for row in rows:
        h.update("|".join(map(str, row)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:12]


def _count_decisions(steps: List[Dict[str, Any]]) -> int:
    """Count accepted decisions like oscillation_metrics does (code_fix + subsequent test)."""
    steps = sorted(steps, key=lambda x: int(x.get("step_id", 0)))
    decisions = 0
    pending = False
    for step in steps:
        st = step.get("step_type")
        if st == "code_fix":
            pending = True
        elif st == "test" and pending:
            decisions += 1
            pending = False
    return decisions


def _load_tasks(task_path: Path) -> List[str]:
    if not task_path.exists():
        return []
    task_ids = []
    with task_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            task_id = obj.get("task_id")
            if task_id:
                task_ids.append(str(task_id))
    return task_ids


@dataclass
class Args:
    run_dirs: List[str]
    tasks_path: str = ""
    expected_n: int = 0
    output_json: str = ""
    strict: bool = False
    diff_tasks: bool = False
    diff_limit: int = 20


def main(args: Args) -> None:
    task_ids_expected = _load_tasks(Path(args.tasks_path)) if args.tasks_path else []
    expected_set = set(task_ids_expected)

    summaries = []
    fingerprints: Dict[str, str] = {}
    task_success_by_run: Dict[str, Dict[str, int]] = {}

    for d in args.run_dirs:
        run_dir = Path(d)
        runs_path = run_dir / "fact_run.jsonl"
        steps_path = run_dir / "fact_step.jsonl"

        runs = _read_jsonl(runs_path)
        steps = _read_jsonl(steps_path)

        steps_by_run: Dict[str, List[Dict[str, Any]]] = {}
        for s in steps:
            rid = s.get("run_id")
            if rid:
                steps_by_run.setdefault(str(rid), []).append(s)

        run_ids = [r.get("run_id") for r in runs]
        task_ids = [r.get("task_id") for r in runs]
        model_ids = sorted({r.get("model_id") for r in runs if r.get("model_id")})
        variant_ids = sorted({r.get("variant_id") for r in runs if r.get("variant_id")})

        missing_steps = sum(1 for r in runs if str(r.get("run_id")) not in steps_by_run)
        failure_modes = Counter((r.get("failure_mode") or "none") for r in runs)

        decisions = []
        no_decision = 0
        for r in runs:
            rid = str(r.get("run_id"))
            n_dec = _count_decisions(steps_by_run.get(rid, []))
            decisions.append(n_dec)
            if n_dec == 0:
                no_decision += 1

        success_rate = (
            sum(1 for r in runs if r.get("success")) / len(runs)
            if runs
            else 0.0
        )

        # Per-task success map for cross-model diff
        task_success: Dict[str, int] = {}
        for r in runs:
            tid = r.get("task_id")
            if not tid:
                continue
            task_success[str(tid)] = 1 if r.get("success") else 0
        task_success_by_run[run_dir.name] = task_success

        missing_tasks = []
        extra_tasks = []
        if expected_set:
            got_set = set(t for t in task_ids if t)
            missing_tasks = sorted(list(expected_set - got_set))
            extra_tasks = sorted(list(got_set - expected_set))

        fp = _compute_data_fingerprint(runs)
        if fp in fingerprints:
            msg = f"Duplicate fingerprint: {fp} for {run_dir.name} and {fingerprints[fp]}"
            if args.strict:
                raise AssertionError(msg)
            print(f"[!] Warning: {msg}")
        else:
            fingerprints[fp] = run_dir.name

        if args.expected_n and len(runs) != args.expected_n:
            msg = f"{run_dir.name}: runs={len(runs)} (expected {args.expected_n})"
            if args.strict:
                raise AssertionError(msg)
            print(f"[!] Warning: {msg}")

        summaries.append({
            "run_dir": str(run_dir),
            "runs": len(runs),
            "steps": len(steps),
            "unique_tasks": len(set(t for t in task_ids if t)),
            "missing_steps": missing_steps,
            "success_rate": success_rate,
            "no_decision_runs": no_decision,
            "decision_mean": statistics.mean(decisions) if decisions else 0.0,
            "model_ids": model_ids,
            "variant_ids": variant_ids,
            "failure_modes_top": failure_modes.most_common(5),
            "missing_tasks": missing_tasks[:10],
            "missing_tasks_count": len(missing_tasks),
            "extra_tasks_count": len(extra_tasks),
            "data_fingerprint": fp,
        })

    print("\n=== RUN AUDIT SUMMARY ===")
    for s in summaries:
        print(
            f"{Path(s['run_dir']).name}: runs={s['runs']} tasks={s['unique_tasks']} "
            f"success={s['success_rate']:.1%} no_decision={s['no_decision_runs']} "
            f"fingerprint={s['data_fingerprint']}"
        )
        if s["missing_tasks_count"] > 0:
            print(f"  missing_tasks={s['missing_tasks_count']} (showing {min(10, s['missing_tasks_count'])})")
        if s["extra_tasks_count"] > 0:
            print(f"  extra_tasks={s['extra_tasks_count']}")
        if s["failure_modes_top"]:
            print(f"  failure_modes_top={s['failure_modes_top']}")

    if args.output_json:
        out_path = Path(args.output_json)
        payload: Dict[str, Any] = {"runs": summaries}
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote audit report to: {out_path}")

    # Cross-model diff
    if args.diff_tasks and len(task_success_by_run) >= 2:
        run_names = sorted(task_success_by_run.keys())
        common_tasks = set.intersection(
            *[set(task_success_by_run[n].keys()) for n in run_names]
        )
        if not common_tasks:
            print("\n[!] No common tasks across run dirs for diff.")
            return

        print("\n=== CROSS-MODEL TASK DIFF ===")
        print(f"Models: {', '.join(run_names)}")
        print(f"Common tasks: {len(common_tasks)}")

        # Per-model pass rate on common tasks
        for name in run_names:
            vals = [task_success_by_run[name][t] for t in common_tasks]
            rate = sum(vals) / len(vals) if vals else 0.0
            print(f"  {name}: pass_rate={rate:.1%}")

        # Disagreement analysis
        patterns: Dict[Tuple[int, ...], List[str]] = {}
        for t in sorted(common_tasks):
            pattern = tuple(task_success_by_run[name][t] for name in run_names)
            patterns.setdefault(pattern, []).append(t)

        all_pass = patterns.get(tuple([1] * len(run_names)), [])
        all_fail = patterns.get(tuple([0] * len(run_names)), [])
        mixed = {k: v for k, v in patterns.items() if k not in [(1,) * len(run_names), (0,) * len(run_names)]}

        print(f"\nAll-pass tasks: {len(all_pass)}")
        print(f"All-fail tasks: {len(all_fail)}")
        print(f"Mixed outcomes: {sum(len(v) for v in mixed.values())}")

        # Show a few mixed tasks
        shown = 0
        for pattern, tasks in sorted(mixed.items(), key=lambda x: -len(x[1])):
            if shown >= args.diff_limit:
                break
            print(f"  pattern={pattern} count={len(tasks)} e.g. {tasks[0]}")
            shown += 1


if __name__ == "__main__":
    main(tyro.cli(Args))
