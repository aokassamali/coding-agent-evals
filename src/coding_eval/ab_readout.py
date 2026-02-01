from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import tyro


@dataclass
class Args:
    run_dir: str = "runs/latest"
    a_variant: str = "A_naive"
    b_variant: str = "B_debug"
    show_by_category: bool = True
    show_by_topic: bool = True
    max_list: int = 50  # max tasks to print in lists


def load_runs(path: str) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                runs.append(json.loads(line))
    return runs


def _rate(numer: int, denom: int) -> float:
    return 0.0 if denom == 0 else numer / denom


def summarize_variant(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rs)
    succ = sum(int(x.get("success", 0)) for x in rs)
    fm = Counter((x.get("failure_mode") or "success") for x in rs)

    latencies = []
    for x in rs:
        s = x.get("started_ms")
        e = x.get("ended_ms")
        if isinstance(s, int) and isinstance(e, int) and e >= s:
            latencies.append(e - s)

    out = {
        "n": n,
        "succ": succ,
        "completion_rate": _rate(succ, n),
        "failure_modes": fm,
        "lat_ms_mean": (sum(latencies) / len(latencies)) if latencies else None,
        "lat_ms_p50": None,
        "lat_ms_p95": None,
        "rewrite_rate": _rate(fm.get("rewrite_too_large", 0), n),
    }

    if latencies:
        lat_sorted = sorted(latencies)
        out["lat_ms_p50"] = lat_sorted[int(0.50 * (len(lat_sorted) - 1))]
        out["lat_ms_p95"] = lat_sorted[int(0.95 * (len(lat_sorted) - 1))]

    return out


def print_variant_summary(name: str, stats: Dict[str, Any]) -> None:
    n = stats["n"]
    succ = stats["succ"]
    cr = stats["completion_rate"]
    print(f"\nVariant: {name}")
    print(f"  runs: {n}")
    print(f"  completion_rate: {cr:.3f} ({succ}/{n})")
    print(f"  rewrite_too_large_rate: {stats['rewrite_rate']:.3f}")

    if stats["lat_ms_mean"] is not None:
        print(
            "  latency_ms: "
            f"mean={stats['lat_ms_mean']:.0f} "
            f"p50={stats['lat_ms_p50']:.0f} "
            f"p95={stats['lat_ms_p95']:.0f}"
        )

    fm: Counter = stats["failure_modes"]
    print("  top_failure_modes:")
    for k, c in fm.most_common(8):
        print(f"    - {k}: {c}")


def summarize(runs: List[Dict[str, Any]]) -> None:
    by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        by_variant[r["variant_id"]].append(r)

    for v, rs in by_variant.items():
        stats = summarize_variant(rs)
        print_variant_summary(v, stats)


def summarize_by_field(runs: List[Dict[str, Any]], field: str) -> None:
    """
    Prints per-variant breakdown by a categorical field: category or topic.
    Also prints the specialized rates asked for:
      - stability_noop regression rate = 1 - pass_rate within stability_noop
      - hang_timeout recovery rate = pass_rate within hang_timeout
    """
    by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        by_variant[r["variant_id"]].append(r)

    for v, rs in by_variant.items():
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rs:
            key = r.get(field) or "unknown"
            buckets[str(key)].append(r)

        print(f"\nVariant: {v} — by {field}")
        for key in sorted(buckets.keys()):
            brs = buckets[key]
            n = len(brs)
            succ = sum(int(x.get("success", 0)) for x in brs)
            cr = _rate(succ, n)

            fm = Counter((x.get("failure_mode") or "success") for x in brs)
            rewrite_rate = _rate(fm.get("rewrite_too_large", 0), n)

            line = f"  {key:18s}  completion={cr:.3f} ({succ:>3d}/{n:<3d})  rewrite={rewrite_rate:.3f}"

            # Specialized metrics
            if field == "category" and key == "stability_noop":
                reg = 1.0 - cr
                line += f"  regression={reg:.3f}"
            if field == "category" and key == "hang_timeout":
                line += f"  recovery={cr:.3f}"

            print(line)


def paired_table(
    runs: List[Dict[str, Any]],
    a: str = "A_naive",
    b: str = "B_debug",
    category_filter: Optional[str] = None,
    max_list: int = 50,
) -> None:
    by_task: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    a_only: List[str] = []
    b_only: List[str] = []
    both_fail: List[str] = []

    for r in runs:
        if category_filter is not None and (r.get("category") or "unknown") != category_filter:
            continue
        key = (r["task_id"], r["model_id"])
        by_task.setdefault(key, {})[r["variant_id"]] = r

    n11 = n10 = n01 = n00 = 0
    for (task_id, model_id), d in by_task.items():
        if a not in d or b not in d:
            continue
        a_s = int(d[a].get("success", 0))
        b_s = int(d[b].get("success", 0))

        if a_s == 1 and b_s == 1:
            n11 += 1
        elif a_s == 1 and b_s == 0:
            n10 += 1
            a_only.append(task_id)
        elif a_s == 0 and b_s == 1:
            n01 += 1
            b_only.append(task_id)
        else:
            n00 += 1
            both_fail.append(task_id)

    title = "Paired contingency"
    if category_filter is not None:
        title += f" (category={category_filter})"
    print(f"\n{title} (A rows vs B cols)")
    print(f"  n11 (both succeed): {n11}")
    print(f"  n10 (A only):       {n10}")
    print(f"  n01 (B only):       {n01}")
    print(f"  n00 (both fail):    {n00}")

    def _clip(xs: List[str]) -> List[str]:
        return xs[:max_list]

    print(f"\nA only tasks: {_clip(a_only)}")
    print(f"B only tasks: {_clip(b_only)}")
    print(f"Both fail tasks: {_clip(both_fail)}")


def main(args: Args) -> None:
    runs = load_runs(f"{args.run_dir}/fact_run.jsonl")

    # Overall
    summarize(runs)

    # Slices
    if args.show_by_category:
        summarize_by_field(runs, "category")
    if args.show_by_topic:
        summarize_by_field(runs, "topic")

    # Paired overall + paired by-category (super useful once you freeze prompts)
    paired_table(runs, a=args.a_variant, b=args.b_variant, max_list=args.max_list)
    if args.show_by_category:
        for cat in ["bugfix", "stability_noop", "hang_timeout", "from_scratch", "unknown"]:
            paired_table(
                runs,
                a=args.a_variant,
                b=args.b_variant,
                category_filter=cat,
                max_list=args.max_list,
            )


if __name__ == "__main__":
    main(tyro.cli(Args))
