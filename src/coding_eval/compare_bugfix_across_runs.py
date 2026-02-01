# What this script does:
# Compare per-task success across multiple run folders to confirm whether models truly tie
# (same tasks pass/fail) or your summary is hiding differences.

from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

RUN_DIRS = [
    r"C:\Users\aokas\coding-agent-evals\runs\v4PromptB_Final_codegemma_7b_it_Q6_K",
    r"C:\Users\aokas\coding-agent-evals\runs\v4PromptB_Final_deepseek_coder_6_7b_instruct_Q6_K",
    r"C:\Users\aokas\coding-agent-evals\runs\v4PromptB_Final_qwen2_5_coder_3b_instruct_q6_k",
    r"C:\Users\aokas\coding-agent-evals\runs\v4PromptB_Final_qwen2_5_coder_7b_instruct_q4_k_m",
    r"C:\Users\aokas\coding-agent-evals\runs\v4PromptB_Final_qwen2_5_coder_7b_instruct_q6_k",
]

def load_runs(run_dir: str):
    p = Path(run_dir) / "fact_run.jsonl"
    runs = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                runs.append(json.loads(line))
    return runs

def main():
    # task_id -> model_name -> success
    by_task = defaultdict(dict)
    model_names = []

    for d in RUN_DIRS:
        runs = load_runs(d)
        if not runs:
            print(f"Empty runs: {d}")
            continue
        model = runs[0].get("model_id", Path(d).name)
        model_names.append(model)

        # filter to bugfix only + B_debug only to match your table
        for r in runs:
            if r.get("variant_id") != "B_debug":
                continue
            if r.get("category") != "bugfix":
                continue
            by_task[r["task_id"]][model] = int(r["success"])

    tasks = sorted(by_task.keys())
    if not tasks:
        print("No bugfix tasks found (check category/variant in RunLog).")
        return

    # Find differences
    diff_tasks = []
    all_fail = []
    all_pass = []
    for tid in tasks:
        vals = [by_task[tid].get(m) for m in model_names]
        if any(v is None for v in vals):
            continue
        s = set(vals)
        if len(s) > 1:
            diff_tasks.append((tid, dict(zip(model_names, vals))))
        elif list(s)[0] == 0:
            all_fail.append(tid)
        else:
            all_pass.append(tid)

    print(f"\nBugfix tasks compared: {len(tasks)}")
    print(f"All-pass tasks: {len(all_pass)}")
    print(f"All-fail tasks: {len(all_fail)}")
    print(f"Different-outcome tasks: {len(diff_tasks)}")

    if all_fail:
        print("\nALL FAIL (candidate hard-core set):")
        for t in all_fail:
            print(" -", t)

    if diff_tasks:
        print("\nDIFFERENT OUTCOMES:")
        for tid, m in diff_tasks:
            print("\n", tid)
            for k, v in m.items():
                print(f"   {k}: {v}")

if __name__ == "__main__":
    main()
