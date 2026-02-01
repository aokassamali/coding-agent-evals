"""
What this script does:
- Run ONE task against ONE model/variant and print everything (no files).
Why: fast debugging of prompt/extraction/fix-loop behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import tyro

from tasks import load_tasks_jsonl
from agent import LLMAgent
from llm import LLMConfig
from sandbox import run_pytest


@dataclass
class Args:
    tasks_path: str = "data/tasks/v3_split_task_categories.jsonl"
    task_id: str = "bug_precision_accum_19"

    variant: str = "B_debug"
    model_id: str = "local-model-label"

    llm_base_url: str = "http://127.0.0.1:8080/v1"
    max_attempts: int = 3
    timeout_s: int = 3
    max_tokens: int = 512


def main(args: Args) -> None:
    # Load tasks and pick exactly one by id
    tasks = load_tasks_jsonl(args.tasks_path)
    t = next((x for x in tasks if x.task_id == args.task_id), None)
    if t is None:
        raise SystemExit(f"Task_id not found: {args.task_id}")

    # Create agent
    agent = LLMAgent(
        variant=args.variant,
        cfg=LLMConfig(
            base_url=args.llm_base_url,
            model=args.model_id,         # label only; server may ignore
            temperature=0.0,
            max_tokens=args.max_tokens,
            timeout_s=120,
        ),
    )

    print("\n====================")
    print("TASK")
    print("====================")
    print(f"task_id:   {t.task_id}")
    print(f"category:  {getattr(t, 'category', None)}")
    print(f"topic:     {getattr(t, 'topic', None)}")
    print(f"signature: {t.signature}")
    print(f"prompt:\n{t.prompt}")
    print("--------------------")

    # Start code: starter_code (if present) else propose
    if getattr(t, "starter_code", "").strip():
        code = t.starter_code
        print("\n[starter_code present] Using starter_code as initial code.\n")
    else:
        out = agent.propose(signature=t.signature, prompt=t.prompt)
        code = out.code
        print("\n[propose] RAW RESPONSE:\n")
        print(out.response)
        print("\n[propose] EXTRACTED CODE:\n")
        print(code)
        print("\n--------------------")

    # Attempt loop: test -> fix
    for attempt in range(1, args.max_attempts + 1):
        print(f"\n====================")
        print(f"ATTEMPT {attempt}: TEST")
        print("====================")

        exec_res = run_pytest(code, t.tests, timeout_s=args.timeout_s)
        print(exec_res.stdout or "")
        if exec_res.stderr:
            print("\n[stderr]")
            print(exec_res.stderr)

        if exec_res.exit_code == 0:
            print("\n✅ PASSED")
            return

        if attempt == args.max_attempts:
            print("\n❌ FAILED (max attempts reached)")
            print("\nFinal code:\n")
            print(code)
            return

        print(f"\n====================")
        print(f"ATTEMPT {attempt}: FIX")
        print("====================")

        out = agent.fix(
            signature=t.signature,
            prompt=t.prompt,
            prev_code=code,
            test_output=(exec_res.stdout or "") + "\n" + (exec_res.stderr or ""),
        )

        print("\n[fix] RAW RESPONSE:\n")
        print(out.response)

        print("\n[fix] EXTRACTED CODE:\n")
        print(out.code)

        no_change = out.code.strip() == code.strip()
        print(f"\n[fix] no_change = {no_change}")

        code = out.code


if __name__ == "__main__":
    main(tyro.cli(Args))
