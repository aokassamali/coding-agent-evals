# What this file does:
# Provide a clean CLI entrypoint for running an eval on any task JSONL and saving
# results to an isolated run directory (runs/<run_tag>/). This avoids editing code
# for each experiment and prevents different task suites from mixing in runs/latest.

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
import tyro
from llm import LLMConfig, fetch_server_models
import json

from tasks import load_tasks_jsonl
from runner import run_suite


@dataclass
class Args:
    tasks_path: str = ""
    out_root: str = "runs/"
    out_dir: str = ""  # Optional full output directory override
    run_tag: str = ""  # if empty, we auto-generate one

    llm_base_url: str = "http://127.0.0.1:8080/v1"
    model_id: str = "qwen2.5-coder"
    temperature: float = 0.0

    variants: tuple[str, ...] = ("A_naive", "B_debug")

    max_attempts: int = 4
    timeout_s: int = 4
    reject_no_progress: bool = True
    reject_rewrite_too_large: bool = True
    rewrite_min_prev_lines: int = 6
    rewrite_max_ratio: float = 1.5
    rewrite_max_abs_lines: int = 0  # 0 disables absolute line guardrail
    inject_prompt: str = ""
    inject_after_attempt: int = 0  # 0 disables; 2 means inject before attempt 3
    inject_on_rejection: bool = False
    inject_relax_guardrails: bool = False


def main(args: Args) -> None:
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        # If the user doesn't provide a run_tag, create a readable one.
        # We include a timestamp so repeated runs don't collide.
        if not args.run_tag:
            ts = time.strftime("%Y%m%d_%H%M%S")
            task_stem = Path(args.tasks_path).stem
            args.run_tag = f"{task_stem}_{args.model_id}_{ts}"
        out_dir = Path(args.out_root) / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks_jsonl(args.tasks_path)
    
    cfg = LLMConfig(
        base_url=args.llm_base_url,
        model=args.model_id,
        temperature=args.temperature,
        max_tokens=512,
        timeout_s=120,
    )

    (out_dir / "server_models.json").write_text(
        json.dumps(fetch_server_models(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Run each variant into the SAME run folder so readout can compare them,
    # but different experiments live in different run_tag directories.
    for v in args.variants:
        run_suite(
            tasks=tasks,
            out_dir=str(out_dir),
            variant_id=v,  # type: ignore
            model_id=args.model_id,
            max_attempts=args.max_attempts,
            timeout_s=args.timeout_s,
            llm_base_url=args.llm_base_url,
            temperature=args.temperature,
            reject_no_progress=args.reject_no_progress,
            reject_rewrite_too_large=args.reject_rewrite_too_large,
            rewrite_min_prev_lines=args.rewrite_min_prev_lines,
            rewrite_max_ratio=args.rewrite_max_ratio,
            rewrite_max_abs_lines=args.rewrite_max_abs_lines,
            inject_prompt=args.inject_prompt,
            inject_after_attempt=args.inject_after_attempt,
            inject_on_rejection=args.inject_on_rejection,
            inject_relax_guardrails=args.inject_relax_guardrails,
        )

    print(f"Done. Wrote logs to: {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
