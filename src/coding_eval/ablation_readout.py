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


def _min_int(xs: List[Optional[int]]) -> Optional[int]:
    ys = [x for x in xs if isinstance(x, int)]
    return min(ys) if ys else None


def _first_int(xs: List[Optional[int]]) -> Optional[int]:
    for x in xs:
        if isinstance(x, int):
            return x
    return None


def _oscillation(xs: List[Optional[int]]) -> bool:
    """
    Thrash/oscillation on failure counts: there exists a decrease at some point,
    and later an increase.
      [2, 1, 2] => True
      [3, 2, 1] => False
      [2, 2, 2] => False
    """
    ys = [x for x in xs if isinstance(x, int)]
    if len(ys) < 3:
        return False
    saw_decrease = False
    for i in range(1, len(ys)):
        if ys[i] < ys[i - 1]:
            saw_decrease = True
        elif saw_decrease and ys[i] > ys[i - 1]:
            return True
    return False


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
    Estimate "distance to green" from pytest output.

    Common endings:
      "1 failed in 0.08s"
      "7 failed, 12 passed in ..."
      "1 failed, 1 passed, 1 skipped in ..."
    """
    s = (pytest_stdout or "") + "\n" + (pytest_stderr or "")
    m = _FAILED_RE.search(s)
    if m:
        return int(m.group("failed"))
    mp = _PASSED_RE.search(s)
    if mp:
        return 0
    return None


def _fail_curve(step_rows: List[Dict[str, Any]]) -> List[Optional[int]]:
    """
    Returns a list of failed-test counts for each pytest test step in order.
    Each element is:
      - int >= 0 if parsed
      - None if we can't parse
    """
    curve: List[Optional[int]] = []
    for s in step_rows:
        if s.get("step_type") != "test":
            continue
        curve.append(_parse_failed_count(s.get("stdout", ""), s.get("stderr", "")))
    return curve

_TIER_RE = re.compile(r"^(tier\d+)_", re.IGNORECASE)

def _task_tier(task_id: str) -> str:
    m = _TIER_RE.match(task_id or "")
    return m.group(1).lower() if m else "tier?"

def _fmt_rate(x: Optional[float], width: int = 7) -> str:
    if x is None:
        return " " * (width - 3) + "n/a"
    return f"{100.0*x:{width-1}.1f}%"


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

    # Overall success (1/0). Prefer fact_run.jsonl's success if present; else infer from tests.
    success: int

    # Derived from step logs
    has_starter: bool

    # test-attempt index: counts ALL test steps (including starter test if present)
    first_pass_test_attempt: Optional[int]     # 1..N, None if never passes

    # fix-attempt index: counts FIX attempts only (starter test is "0 fixes")
    first_pass_fix_attempt: Optional[int]      # 0..Nfix, None if never passes

    failed_tests_attempt1: Optional[int]       # failed tests on first test step (parsed)
    n_test_steps: int                          # number of test steps recorded
    rewrite_rejected: bool                     # any code_fix_rejected or failure_mode rewrite_too_large

    # Behavior discriminator: model repeats unchanged code while still failing
    thrash: bool

    # Latency for LLM steps (ms)
    llm_latencies_ms: List[int]

    # Progress metrics
    failed_counts_by_test: List[Optional[int]]  # per test step: #failed (0 if passing)
    best_failed: Optional[int]                  # min failed across attempts (includes 0 if passes)
    improved: Optional[bool]                    # best_failed < failed_tests_attempt1 (if both known)
    improve_any: bool                           # True iff best_failed < first failed count
    improve_frac: Optional[float]               # (failed1 - best)/failed1 (if failed1>0)
    thrash_failcurve: bool                      # oscillation on failed-count curve
    tier: str



def _derive_from_steps(
    run_row: Dict[str, Any],
    step_rows: List[Dict[str, Any]],
) -> RunDerived:
    run_id = str(run_row.get("run_id", ""))
    model_id = str(run_row.get("model_id", ""))
    variant_id = str(run_row.get("variant_id", ""))
    task_id = str(run_row.get("task_id", ""))
    category = run_row.get("category")
    topic = run_row.get("topic")
    tier = _task_tier(task_id)
    # sort steps by step_id
    step_rows = sorted(step_rows, key=lambda r: int(r.get("step_id", 0)))

    has_starter = any(s.get("step_type") == "code_starter" for s in step_rows)

    test_steps = [s for s in step_rows if s.get("step_type") == "test"]
    fix_steps = [s for s in step_rows if s.get("step_type") == "code_fix"]

    # ---- fail curve + progress metrics ----
    fc = _fail_curve(step_rows)
    failed_tests_attempt1: Optional[int] = fc[0] if fc else None

    fc_numeric = [x for x in fc if isinstance(x, int)]
    best_failed: Optional[int] = min(fc_numeric) if fc_numeric else None

    improved: Optional[bool] = None
    if isinstance(failed_tests_attempt1, int) and isinstance(best_failed, int):
        improved = (best_failed < failed_tests_attempt1)

    improve_any = bool(improved) if isinstance(improved, bool) else False

    improve_frac: Optional[float] = None
    if isinstance(failed_tests_attempt1, int) and isinstance(best_failed, int) and failed_tests_attempt1 > 0:
        improve_frac = (failed_tests_attempt1 - best_failed) / float(failed_tests_attempt1)

    thrash_failcurve = _oscillation(fc)

    # --- first pass (test attempt index) ---
    first_pass_test_attempt: Optional[int] = None
    attempt_idx = 0
    for ts in test_steps:
        attempt_idx += 1
        exit_code = ts.get("exit_code")
        if exit_code == 0 and first_pass_test_attempt is None:
            first_pass_test_attempt = attempt_idx

    # --- first pass (fix attempt index) ---
    first_pass_fix_attempt: Optional[int] = None
    if first_pass_test_attempt is not None:
        first_pass_fix_attempt = max(0, first_pass_test_attempt - 1)

    # --- overall success ---
    # Prefer run_row["success"] if present; else infer from whether any test passed.
    success_raw = run_row.get("success", None)
    if isinstance(success_raw, int):
        success = 1 if success_raw != 0 else 0
    elif isinstance(success_raw, bool):
        success = 1 if success_raw else 0
    else:
        success = 1 if first_pass_test_attempt is not None else 0

    # --- rewrite rejected? ---
    rewrite_rejected = False
    if (run_row.get("failure_mode") or "") == "rewrite_too_large":
        rewrite_rejected = True
    for s in step_rows:
        if s.get("step_type") == "code_fix_rejected":
            rewrite_rejected = True
            break

    # --- thrash (meta.no_change-based) ---
    thrash = False
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

    # --- LLM step latencies ---
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
        success=success,
        has_starter=has_starter,
        first_pass_test_attempt=first_pass_test_attempt,
        first_pass_fix_attempt=first_pass_fix_attempt,
        failed_tests_attempt1=failed_tests_attempt1,
        n_test_steps=len(test_steps),
        rewrite_rejected=rewrite_rejected,
        thrash=thrash,
        llm_latencies_ms=llm_latencies_ms,
        failed_counts_by_test=fc,
        best_failed=best_failed,
        improved=improved,
        improve_any=improve_any,
        improve_frac=improve_frac,
        thrash_failcurve=thrash_failcurve,
        tier=tier,
    )


# -----------------------------
# Aggregation helpers
# -----------------------------

@dataclass
class Args:
    run_dirs: List[str]
    k_values: Tuple[int, ...] = (1, 2, 3, 4)


def _rate(preds: Iterable[bool]) -> Optional[float]:
    preds = list(preds)
    if not preds:
        return None
    return sum(1 for x in preds if x) / len(preds)


def _pass_at_k(rs: List[RunDerived], k: int, *, which: str = "fix") -> Optional[float]:
    """
    which: "test" uses first_pass_test_attempt
           "fix"  uses first_pass_fix_attempt
    """
    if not rs:
        return None
    num = 0
    for r in rs:
        a = r.first_pass_fix_attempt if which == "fix" else r.first_pass_test_attempt
        if a is not None and a <= k:
            num += 1
    return num / len(rs)


def _median_attempts_to_pass(rs: List[RunDerived], *, which: str = "fix") -> Optional[float]:
    xs: List[int] = []
    for r in rs:
        a = r.first_pass_fix_attempt if which == "fix" else r.first_pass_test_attempt
        if isinstance(a, int):
            xs.append(a)
    if not xs:
        return None
    return float(statistics.median(xs))


def _median_failed_tests_attempt1(rs: List[RunDerived]) -> Optional[float]:
    xs = [r.failed_tests_attempt1 for r in rs if isinstance(r.failed_tests_attempt1, int)]
    if not xs:
        return None
    return float(statistics.median(xs))


def _median_best_failed(rs: List[RunDerived]) -> Optional[float]:
    xs = [r.best_failed for r in rs if isinstance(r.best_failed, int)]
    if not xs:
        return None
    return float(statistics.median(xs))


def _improve_rate(rs: List[RunDerived]) -> Optional[float]:
    xs = [r.improved for r in rs if isinstance(r.improved, bool)]
    if not xs:
        return None
    return sum(1 for x in xs if x) / len(xs)


def _median_improve_frac(rs: List[RunDerived]) -> Optional[float]:
    xs = [r.improve_frac for r in rs if isinstance(r.improve_frac, float)]
    if not xs:
        return None
    return float(statistics.median(xs))


def _filter_failed_runs(rs: List[RunDerived]) -> List[RunDerived]:
    return [r for r in rs if r.success == 0]


def _median_best_failed_failed_only(rs: List[RunDerived]) -> Optional[float]:
    failed = _filter_failed_runs(rs)
    xs = [r.best_failed for r in failed if isinstance(r.best_failed, int)]
    if not xs:
        return None
    return float(statistics.median(xs))


def _improve_rate_failed_only(rs: List[RunDerived]) -> Optional[float]:
    failed = _filter_failed_runs(rs)
    xs = [r.improved for r in failed if isinstance(r.improved, bool)]
    if not xs:
        return None
    return sum(1 for x in xs if x) / len(xs)


def _median_improve_frac_failed_only(rs: List[RunDerived]) -> Optional[float]:
    failed = _filter_failed_runs(rs)
    xs = [r.improve_frac for r in failed if isinstance(r.improve_frac, float)]
    if not xs:
        return None
    return float(statistics.median(xs))


# -----------------------------
# Summarize
# -----------------------------

def _summarize_run_dir(run_dir: Path, k_values: Tuple[int, ...]) -> List[Dict[str, Any]]:
    runs_path = run_dir / "fact_run.jsonl"
    steps_path = run_dir / "fact_step.jsonl"

    runs = _read_jsonl(runs_path)
    steps = _read_jsonl(steps_path)

    steps_by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in steps:
        rid = s.get("run_id")
        if rid:
            steps_by_run[str(rid)].append(s)

    derived: List[RunDerived] = []
    for r in runs:
        rid = r.get("run_id")
        if not rid:
            continue
        d = _derive_from_steps(r, steps_by_run.get(str(rid), []))
        derived.append(d)

    by_mv: Dict[Tuple[str, str], List[RunDerived]] = defaultdict(list)
    for d in derived:
        by_mv[(d.model_id, d.variant_id)].append(d)

    out_rows: List[Dict[str, Any]] = []

    for (model_id, variant_id), rs in sorted(by_mv.items()):
        bugfix = [r for r in rs if r.category == "bugfix"]
        stab = [r for r in rs if r.category == "stability_noop"]
        hang = [r for r in rs if r.category == "hang_timeout"]

        rewrite_rate = _rate(r.rewrite_rejected for r in rs)

        lat_all: List[int] = []
        for r in rs:
            lat_all.extend(r.llm_latencies_ms)
        lat_p50 = _percentile([float(x) for x in lat_all], 50) if lat_all else None
        lat_p95 = _percentile([float(x) for x in lat_all], 95) if lat_all else None

        # --- Tier stats (Option A) ---
        tiers: Dict[str, List[RunDerived]] = defaultdict(list)
        for r in bugfix:
            tiers[_task_tier(r.task_id)].append(r)

        tier_rows: List[Dict[str, Any]] = []
        for tier_name, trs in sorted(tiers.items()):
            tr = {
                "tier": tier_name,
                "n": len(trs),
                # keep it minimal: p@1 and p@4 (and medAtt) are usually enough
                "p1": _pass_at_k(trs, 1, which="fix"),
                "p4": _pass_at_k(trs, max(k_values) if k_values else 4, which="fix"),
                "medAtt": _median_attempts_to_pass(trs, which="fix"),
                "imprRate": _improve_rate(trs),
                "thrash2": _rate(r.thrash_failcurve for r in trs),
            }
            tier_rows.append(tr)



        bugfix_pass_fix = {k: _pass_at_k(bugfix, k, which="fix") for k in k_values}

        bugfix_med_attempts = _median_attempts_to_pass(bugfix, which="fix")
        bugfix_first_fix_success = _rate((r.first_pass_fix_attempt == 1) for r in bugfix)
        bugfix_thrash_rate = _rate(r.thrash for r in bugfix)
        bugfix_med_failed1 = _median_failed_tests_attempt1(bugfix)

        # “progress” columns (these will collapse if most tasks pass)
        bugfix_best_failed_med = _median_best_failed(bugfix)
        bugfix_improve_rate = _improve_rate(bugfix)
        bugfix_improve_frac_med = _median_improve_frac(bugfix)
        bugfix_thrash_curve_rate = _rate(r.thrash_failcurve for r in bugfix)

        # FAILED-ONLY versions (these are what you want for model separation)
        bugfix_best_failed_med_F = _median_best_failed_failed_only(bugfix)
        bugfix_improve_rate_F = _improve_rate_failed_only(bugfix)
        bugfix_improve_frac_med_F = _median_improve_frac_failed_only(bugfix)

        stab_pass_at1 = _pass_at_k(stab, 1, which="test")
        stab_reg = None if stab_pass_at1 is None else (1.0 - stab_pass_at1)

        k_max = max(k_values) if k_values else 1
        hang_rec = _pass_at_k(hang, k_max, which="test")

        row: Dict[str, Any] = {
            "run_tag": run_dir.name,
            "model": (model_id[:28] if model_id else ""),
            "var": variant_id,
            "bugfix_n": len(bugfix),
            "stab_n": len(stab),
            "hang_n": len(hang),
            "rewrite": rewrite_rate,
            "lat_p50_ms": int(lat_p50) if lat_p50 is not None else None,
            "lat_p95_ms": int(lat_p95) if lat_p95 is not None else None,
            "bugfix_med_attempts": bugfix_med_attempts,
            "bugfix_med_failed1": bugfix_med_failed1,
            "bugfix_first_fix": bugfix_first_fix_success,
            "bugfix_thrash": bugfix_thrash_rate,
            "bugfix_best_failed_med": bugfix_best_failed_med,
            "bugfix_improve_rate": bugfix_improve_rate,
            "bugfix_improve_frac_med": bugfix_improve_frac_med,
            "bugfix_thrash_curve": bugfix_thrash_curve_rate,
            # failed-only
            "bugfix_best_failed_med_F": bugfix_best_failed_med_F,
            "bugfix_improve_rate_F": bugfix_improve_rate_F,
            "bugfix_improve_frac_med_F": bugfix_improve_frac_med_F,
            "stab_reg": stab_reg,
            "hang_rec": hang_rec,
            "tier_table": tier_rows,

        }
        for k in k_values:
            row[f"bugfix_fix_p@{k}"] = bugfix_pass_fix[k]
        out_rows.append(row)

    return out_rows


# -----------------------------
# Printing
# -----------------------------

def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "  n/a "
    return f"{100.0 * x:5.1f}%"


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

    # Strict required keys
    required = [
        "run_tag", "model", "var",
        "bugfix_n", "stab_n", "hang_n",
        "stab_reg", "hang_rec", "rewrite",
        "lat_p50_ms", "lat_p95_ms",
        "bugfix_med_attempts", "bugfix_med_failed1",
        "bugfix_first_fix", "bugfix_thrash",
        "bugfix_best_failed_med",
        "bugfix_improve_rate",
        "bugfix_improve_frac_med",
        "bugfix_thrash_curve",
        # failed-only
        "bugfix_best_failed_med_F",
        "bugfix_improve_rate_F",
        "bugfix_improve_frac_med_F",
    ]
    for k in args.k_values:
        required.append(f"bugfix_fix_p@{k}")

    for r in all_rows:
        missing = [k for k in required if k not in r]
        if missing:
            raise KeyError(f"Missing keys in row for run_tag={r.get('run_tag')}: {missing}")

        for rk in ["stab_reg", "hang_rec", "rewrite", "bugfix_first_fix", "bugfix_thrash",
                   "bugfix_improve_rate", "bugfix_thrash_curve",
                   "bugfix_improve_rate_F"] + [f"bugfix_fix_p@{k}" for k in args.k_values]:
            v = r.get(rk)
            if v is None:
                continue
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"Bad rate {rk}={v} in run_tag={r.get('run_tag')}")

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
        "bestFail".rjust(8),
        "imprRate".rjust(8),
        "imprFrac".rjust(8),
        "thrash2".rjust(8),
        "bestF_F".rjust(8),
        "imprR_F".rjust(8),
        "imprF_F".rjust(8),
        "firstFix".rjust(9),
        "thrash".rjust(8),
        "stab_reg".rjust(9),
        "hang_rec".rjust(9),
        "rewrite".rjust(8),
        "lat_p50".rjust(9),
        "lat_p95".rjust(9),
        sep="  ",
    )
    print("-" * 190)

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
            _fmt_num(r["bugfix_best_failed_med"], width=8, nd=1),
            _fmt_pct(r["bugfix_improve_rate"]).rjust(8),
            _fmt_num(r["bugfix_improve_frac_med"], width=8, nd=2),
            _fmt_pct(r["bugfix_thrash_curve"]).rjust(8),
            _fmt_num(r["bugfix_best_failed_med_F"], width=8, nd=1),
            _fmt_pct(r["bugfix_improve_rate_F"]).rjust(8),
            _fmt_num(r["bugfix_improve_frac_med_F"], width=8, nd=2),
            _fmt_pct(r["bugfix_first_fix"]).rjust(9),
            _fmt_pct(r["bugfix_thrash"]).rjust(8),
            _fmt_pct(r["stab_reg"]).rjust(9),
            _fmt_pct(r["hang_rec"]).rjust(9),
            _fmt_pct(r["rewrite"]).rjust(8),
            (str(r["lat_p50_ms"]) if r["lat_p50_ms"] is not None else "n/a").rjust(9),
            (str(r["lat_p95_ms"]) if r["lat_p95_ms"] is not None else "n/a").rjust(9),
            sep="  ",
        )

    # -----------------------------
    # Option A: Tier-wise table per run folder
    # -----------------------------
    for r in all_rows:
        tier_table = r.get("tier_table") or []
        if not tier_table:
            continue

        print("\n--- Tier breakdown:", r["run_tag"], "|", r["model"], "|", r["var"], "---")
        print(
            "tier".ljust(8),
            "n".rjust(4),
            "fix_p@1".rjust(9),
            f"fix_p@{max(args.k_values)}".rjust(9),
            "medAtt".rjust(8),
            "imprRate".rjust(9),
            "thrash2".rjust(9),
            sep="  ",
        )
        print("-" * 70)

        for tr in tier_table:
            print(
                str(tr["tier"]).ljust(8),
                f"{int(tr['n']):4d}",
                _fmt_rate(tr.get("p1")).rjust(9),
                _fmt_rate(tr.get("p4")).rjust(9),
                _fmt_num(tr.get("medAtt"), width=8, nd=2),
                _fmt_rate(tr.get("imprRate")).rjust(9),
                _fmt_rate(tr.get("thrash2")).rjust(9),
                sep="  ",
            )


if __name__ == "__main__":
    main(tyro.cli(Args))
