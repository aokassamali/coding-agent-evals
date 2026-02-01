from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import tyro


# -----------------------------
# Parsing helpers
# -----------------------------

_FAILED_RE = re.compile(r"(?P<failed>\d+)\s+failed\b", re.IGNORECASE)
_PASSED_RE = re.compile(r"(?P<passed>\d+)\s+passed\b", re.IGNORECASE)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _percentile(xs: List[float], p: float) -> Optional[float]:
    # p in [0, 100]
    if not xs:
        return None
    xs_sorted = sorted(xs)
    if len(xs_sorted) == 1:
        return xs_sorted[0]
    k = (len(xs_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs_sorted) - 1)
    if f == c:
        return xs_sorted[f]
    d0 = xs_sorted[f] * (c - k)
    d1 = xs_sorted[c] * (k - f)
    return d0 + d1


def _parse_failed_count(pytest_stdout: str, pytest_stderr: str) -> Optional[int]:
    """
    Try to estimate "distance to green" from pytest output.

    Common endings:
      "1 failed in 0.08s"
      "7 failed, 12 passed in ..."
      "1 failed, 1 passed, 1 skipped in ..."
    """
    s = (pytest_stdout or "") + "\n" + (pytest_stderr or "")
    m = _FAILED_RE.search(s)
    if m:
        return int(m.group("failed"))
    # If we can't find "failed", but we can find "passed", assume 0 failed.
    mp = _PASSED_RE.search(s)
    if mp:
        return 0
    return None


# -----------------------------
# Metrics per run_id
# -----------------------------

@dataclass
class RunDerived:
    run_id: str
    model_id: str
    variant_id: str
    task_id: str
    category: Optional[str]
    topic: Optional[str]

    # Derived from step logs
    has_starter: bool

    # test-attempt index: counts ALL test steps (including starter test if present)
    first_pass_test_attempt: Optional[int]     # 1..N, None if never passes

    # fix-attempt index: counts FIX attempts only (starter test is "0 fixes")
    first_pass_fix_attempt: Optional[int]      # 0..Nfix, None if never passes

    failed_tests_attempt1: Optional[int]       # failed tests on first test step
    n_test_steps: int                          # number of test steps recorded
    rewrite_rejected: bool                     # any code_fix_rejected or failure_mode rewrite_too_large

    # Behavior discriminator: model repeats unchanged code while still failing
    thrash: bool

    # Latency for LLM steps (ms)
    llm_latencies_ms: List[int]



def _derive_from_steps(
    run_row: Dict[str, Any],
    step_rows: List[Dict[str, Any]],
) -> RunDerived:
    run_id = run_row["run_id"]
    model_id = str(run_row.get("model_id", ""))
    variant_id = str(run_row.get("variant_id", ""))
    task_id = str(run_row.get("task_id", ""))
    category = run_row.get("category")
    topic = run_row.get("topic")

    # sort steps by step_id
    step_rows = sorted(step_rows, key=lambda r: int(r.get("step_id", 0)))

    has_starter = any(s.get("step_type") == "code_starter" for s in step_rows)

    test_steps = [s for s in step_rows if s.get("step_type") == "test"]
    fix_steps = [s for s in step_rows if s.get("step_type") == "code_fix"]

    # --- first pass (test attempt index) ---
    first_pass_test_attempt: Optional[int] = None
    failed_tests_attempt1: Optional[int] = None

    attempt_idx = 0
    for ts in test_steps:
        attempt_idx += 1
        exit_code = ts.get("exit_code")

        if attempt_idx == 1:
            failed_tests_attempt1 = _parse_failed_count(ts.get("stdout", ""), ts.get("stderr", ""))

        if exit_code == 0 and first_pass_test_attempt is None:
            first_pass_test_attempt = attempt_idx

    # --- first pass (fix attempt index) ---
    # Map a passing test attempt -> number of fixes that must have happened before it.
    # If starter exists: test attempt #1 is starter check => 0 fixes.
    # So: fix_attempt = test_attempt - 1
    # If no starter: test attempt #1 can pass without any fix => 0 fixes.
    # So: fix_attempt = test_attempt - 1 as well (still works).
    first_pass_fix_attempt: Optional[int] = None
    if first_pass_test_attempt is not None:
        first_pass_fix_attempt = max(0, first_pass_test_attempt - 1)

    # --- rewrite rejected? ---
    rewrite_rejected = False
    if (run_row.get("failure_mode") or "") == "rewrite_too_large":
        rewrite_rejected = True
    for s in step_rows:
        if s.get("step_type") == "code_fix_rejected":
            rewrite_rejected = True
            break

    # --- thrash detection ---
    # thrash=True if we observe:
    #   code_fix.meta.no_change == True
    #   AND the *next* test still fails (exit_code != 0)
    thrash = False
    # build quick index by step_id
    by_step_id = {int(s.get("step_id", 0)): s for s in step_rows if "step_id" in s}
    for s in fix_steps:
        meta = s.get("meta") or {}
        no_change = bool(meta.get("no_change", False))
        if not no_change:
            continue
        sid = int(s.get("step_id", 0))
        next_test = by_step_id.get(sid + 1)
        if next_test and next_test.get("step_type") == "test":
            exit_code = next_test.get("exit_code")
            if exit_code != 0:
                thrash = True
                break

    # --- Latencies for LLM actions only ---
    llm_latencies_ms: List[int] = []
    for s in step_rows:
        st = s.get("step_type")
        if st in ("code_propose", "code_fix", "code_edit"):
            started = s.get("started_ms")
            ended = s.get("ended_ms")
            if isinstance(started, int) and isinstance(ended, int) and ended >= started:
                llm_latencies_ms.append(ended - started)

    return RunDerived(
        run_id=run_id,
        model_id=model_id,
        variant_id=variant_id,
        task_id=task_id,
        category=category,
        topic=topic,
        has_starter=has_starter,
        first_pass_test_attempt=first_pass_test_attempt,
        first_pass_fix_attempt=first_pass_fix_attempt,
        failed_tests_attempt1=failed_tests_attempt1,
        n_test_steps=len(test_steps),
        rewrite_rejected=rewrite_rejected,
        thrash=thrash,
        llm_latencies_ms=llm_latencies_ms,
    )



# -----------------------------
# Aggregation
# -----------------------------

@dataclass
class Args:
    run_dirs: List[str]
    k_values: Tuple[int, ...] = (1, 2, 3, 4)


def _pass_at_k(rs: List[RunDerived], k: int, *, which: str = "test") -> Optional[float]:
    """
    which: "test" uses first_pass_test_attempt
           "fix"  uses first_pass_fix_attempt
    """
    if not rs:
        return None
    num = 0
    for r in rs:
        if which == "fix":
            a = r.first_pass_fix_attempt
        else:
            a = r.first_pass_test_attempt
        if a is not None and a <= k:
            num += 1
    return num / len(rs)


def _median_attempts_to_pass(rs: List[RunDerived], *, which: str = "test") -> Optional[float]:
    xs = []
    for r in rs:
        a = r.first_pass_fix_attempt if which == "fix" else r.first_pass_test_attempt
        if a is not None:
            xs.append(a)
    if not xs:
        return None
    return float(statistics.median(xs))


def _median_failed_tests_attempt1(rs: List[RunDerived]) -> Optional[float]:
    xs = [r.failed_tests_attempt1 for r in rs if r.failed_tests_attempt1 is not None]
    if not xs:
        return None
    return float(statistics.median(xs))


def _rate(preds: Iterable[bool]) -> Optional[float]:
    preds = list(preds)
    if not preds:
        return None
    return sum(1 for x in preds if x) / len(preds)


def _summarize_run_dir(run_dir: Path, k_values: Tuple[int, ...]) -> List[Dict[str, Any]]:
    runs_path = run_dir / "fact_run.jsonl"
    steps_path = run_dir / "fact_step.jsonl"

    runs = _read_jsonl(runs_path)
    steps = _read_jsonl(steps_path)

    # Map run_id -> steps
    steps_by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in steps:
        rid = s.get("run_id")
        if rid:
            steps_by_run[str(rid)].append(s)

    # Derive per-run metrics
    derived: List[RunDerived] = []
    for r in runs:
        rid = r.get("run_id")
        if not rid:
            continue
        rid = str(rid)
        d = _derive_from_steps(r, steps_by_run.get(rid, []))
        derived.append(d)

    # Group by (model, variant) — in your sweep this is usually 1 model + 1 variant per folder,
    # but this supports multiple variants in a single run dir too.
    by_mv: Dict[Tuple[str, str], List[RunDerived]] = defaultdict(list)
    for d in derived:
        by_mv[(d.model_id, d.variant_id)].append(d)

    out_rows: List[Dict[str, Any]] = []

    for (model_id, variant_id), rs in sorted(by_mv.items()):
        # categories
        bugfix = [r for r in rs if r.category == "bugfix"]
        stab = [r for r in rs if r.category == "stability_noop"]
        hang = [r for r in rs if r.category == "hang_timeout"]

        # rewrite rate (behavior)
        rewrite_rate = _rate(r.rewrite_rejected for r in rs)

        # latency stats (LLM step latency, not full wall clock)
        lat_all = []
        for r in rs:
            lat_all.extend(r.llm_latencies_ms)
        lat_p50 = _percentile([float(x) for x in lat_all], 50) if lat_all else None
        lat_p95 = _percentile([float(x) for x in lat_all], 95) if lat_all else None

        # bugfix pass@k (TEST-attempt based; keep if you want)
        bugfix_pass_test = {k: _pass_at_k(bugfix, k, which="test") for k in k_values}

        # bugfix pass@k (FIX-attempt based; this is what you want for starter-code bugfix)
        bugfix_pass_fix = {k: _pass_at_k(bugfix, k, which="fix") for k in k_values}

        # extra bugfix discriminators (fix-attempt space)
        bugfix_med_attempts = _median_attempts_to_pass(bugfix, which="fix")

        # First-fix success rate: passes after exactly 1 fix (i.e. first_pass_fix_attempt == 1)
        bugfix_first_fix_success = _rate(
            (r.first_pass_fix_attempt == 1) for r in bugfix
        )

        # Thrash rate: no_change while failing
        bugfix_thrash_rate = _rate(r.thrash for r in bugfix)

        bugfix_med_failed1 = _median_failed_tests_attempt1(bugfix)

        # stability regression = 1 - pass rate (but stability tasks "should pass")
        stab_pass_at1 = _pass_at_k(stab, 1)  # stability should pass immediately if starter is correct
        stab_reg = None if stab_pass_at1 is None else (1.0 - stab_pass_at1)

        # hang recovery = pass@k (use largest k in k_values as your overall recovery)
        k_max = max(k_values) if k_values else 1
        hang_rec = _pass_at_k(hang, k_max)

        row = {
            "run_tag": run_dir.name,
            "model": model_id[:28] if model_id else "",
            "var": variant_id,
            "bugfix_n": len(bugfix),
            "stab_n": len(stab),
            "hang_n": len(hang),
            "rewrite": rewrite_rate,
            "lat_p50_ms": int(lat_p50) if lat_p50 is not None else None,
            "lat_p95_ms": int(lat_p95) if lat_p95 is not None else None,
            "bugfix_med_attempts": bugfix_med_attempts,
            "bugfix_med_failed1": bugfix_med_failed1,
            "stab_reg": stab_reg,
            "hang_rec": hang_rec,
            "bugfix_first_fix": bugfix_first_fix_success,
            "bugfix_thrash": bugfix_thrash_rate,

        }
        for k in k_values:
            row[f"bugfix_fix_p@{k}"] = bugfix_pass_fix[k]
            # optional: keep old test-based too
            row[f"bugfix_test_p@{k}"] = bugfix_pass_test[k]
        out_rows.append(row)

    return out_rows


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "  n/a "
    return f"{100.0*x:5.1f}%"


def _fmt_num(x: Optional[float], width: int = 7, nd: int = 1) -> str:
    if x is None:
        return " " * (width - 3) + "n/a"
    return f"{x:{width}.{nd}f}"


def main(args: Args) -> None:
    all_rows: List[Dict[str, Any]] = []
    for d in args.run_dirs:
        run_dir = Path(d)
        if not run_dir.exists():
            print(f"[warn] missing run dir: {run_dir}")
            continue
        all_rows.extend(_summarize_run_dir(run_dir, args.k_values))

    if not all_rows:
        print("No runs found.")
        return

    # Sanity check: make sure expected keys exist
    required = [
        "run_tag", "model", "var",
        "bugfix_n", "stab_n", "hang_n",
        "stab_reg", "hang_rec", "rewrite",
        "lat_p50_ms", "lat_p95_ms",
        "bugfix_med_attempts", "bugfix_med_failed1",
        "bugfix_first_fix", "bugfix_thrash",
    ]
    for k in args.k_values:
        required.append(f"bugfix_fix_p@{k}")

    for r in all_rows:
        missing = [k for k in required if k not in r]
        if missing:
            raise KeyError(f"Missing keys in row for run_tag={r.get('run_tag')}: {missing}")

        # rate sanity checks
        for rk in ["stab_reg", "hang_rec", "rewrite", "bugfix_first_fix", "bugfix_thrash"] + [f"bugfix_fix_p@{k}" for k in args.k_values]:
            v = r.get(rk)
            if v is None:
                continue
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"Bad rate {rk}={v} in run_tag={r.get('run_tag')}")

    # Header
    k_cols = "  ".join([f"fix_p@{k}" for k in args.k_values])
    print("\n=== Ablation Summary (per run folder) ===")
    print(
        "run_tag".ljust(44),
        "model".ljust(28),
        "var".ljust(8),
        "bugfix".rjust(6),
        k_cols.rjust(7 * len(args.k_values)),
        "medAtt".rjust(8),
        "medFail1".rjust(9),
        "firstFix".rjust(9),
        "thrash".rjust(8),
        "stab_reg".rjust(9),
        "hang_rec".rjust(9),
        "rewrite".rjust(8),
        "lat_p50".rjust(9),
        "lat_p95".rjust(9),
        sep="  ",
    )
    print("-" * 160)

    for r in all_rows:
        bugfix_pcts = "  ".join(_fmt_pct(r[f"bugfix_fix_p@{k}"]) for k in args.k_values)
        print(
            str(r["run_tag"]).ljust(44),
            str(r["model"]).ljust(28),
            str(r["var"]).ljust(8),
            f"{int(r['bugfix_n']):6d}",
            bugfix_pcts,
            _fmt_num(r["bugfix_med_attempts"], width=8, nd=2),
            _fmt_num(r["bugfix_med_failed1"], width=9, nd=1),
            _fmt_pct(r["bugfix_first_fix"]).rjust(9),
            _fmt_pct(r["bugfix_thrash"]).rjust(8),
            _fmt_pct(r["stab_reg"]).rjust(9),
            _fmt_pct(r["hang_rec"]).rjust(9),
            _fmt_pct(r["rewrite"]).rjust(8),
            (str(r["lat_p50_ms"]) if r["lat_p50_ms"] is not None else "n/a").rjust(9),
            (str(r["lat_p95_ms"]) if r["lat_p95_ms"] is not None else "n/a").rjust(9),
            sep="  ",
        )



if __name__ == "__main__":
    main(tyro.cli(Args))
