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
    tasks_path: str = "data/tasks/v1_bugfix.jsonl"
    out_root: str = "runs"
    run_tag: str = ""  # if empty, we auto-generate one

    llm_base_url: str = "http://127.0.0.1:8080/v1"
    model_id: str = "qwen2.5-coder"

    variants: tuple[str, ...] = ("A_naive", "B_debug")

    max_attempts: int = 3
    timeout_s: int = 10


def main(args: Args) -> None:
    # If the user doesn't provide a run_tag, create a readable one.
    # We include a timestamp so repeated runs don't collide.
    if not args.run_tag:
        ts = time.strftime("%Y%m%d_%H%M%S")
        task_stem = Path(args.tasks_path).stem
        args.run_tag = f"{task_stem}_{args.model_id}_{ts}"

    out_dir = Path(args.out_root) / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks_jsonl(args.tasks_path)
    
    cfg = LLMConfig(base_url=args.llm_base_url, model=args.model_id, temperature=0.0, max_tokens=512, timeout_s=120)

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
        )

    print(f"Done. Wrote logs to: {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
