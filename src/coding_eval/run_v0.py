# What this script does:
# Run the same tasks under two variants (A_naive vs B_debug) against the same local model server,
# producing paired runs for A/B comparison.

from tasks import load_tasks_jsonl
from runner import run_suite

def main():
    tasks = load_tasks_jsonl("data/tasks/v1_bugfix.jsonl")

    # Variant A
    run_suite(
        tasks=tasks,
        out_dir="runs/latest",
        variant_id="A_naive",
        model_id="qwen2.5-coder",
        max_attempts=3,
        timeout_s=10,
        llm_base_url="http://127.0.0.1:8080/v1",
    )

    # Variant B
    run_suite(
        tasks=tasks,
        out_dir="runs/latest",
        variant_id="B_debug",
        model_id="qwen2.5-coder",
        max_attempts=3,
        timeout_s=10,
        llm_base_url="http://127.0.0.1:8080/v1",
    )

    print("Done. Logs in runs/latest/")

if __name__ == "__main__":
    main()
