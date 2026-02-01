# What this module does:
# Run tasks end-to-end with an agent loop (propose -> test -> fix),
# logging every step so we can compute metrics and diagnose failure modes.

from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from tasks import Task
from agent import LLMAgent, VariantId
from sandbox import run_pytest
from schema import StepLog, RunLog, now_ms, new_id
from llm import LLMConfig


def classify_failure(stderr: str, stdout: str, exit_code: int) -> Optional[str]:
    if exit_code == 0:
        return None
    s = (stderr or "") + "\n" + (stdout or "")
    s_low = s.lower()
    if "no module named pytest" in s_low:
        return "harness_missing_pytest"
    if "syntaxerror" in s_low:
        return "syntax_error"
    if "importerror" in s_low or "modulenotfounderror" in s_low:
        return "missing_dependency"
    if "assertionerror" in s_low or "assert " in s_low:
        return "assertion_failed"
    if "timeout" in s_low or "pytest_timeout" in s_low:
        return "timeout"
    if "no module named" in s_low:
        return "missing_dependency"
    return "other"

def has_third_party_import(code: str) -> Optional[str]:
    banned = ("numpy", "pandas", "scipy", "sklearn", "torch", "tensorflow")
    low = code.lower()
    for pkg in banned:
        if f"import {pkg}" in low or f"from {pkg} import" in low:
            return pkg
    return None

def run_suite(
    tasks: List[Task],
    out_dir: str,
    variant_id: VariantId,
    model_id: str,
    max_attempts: int = 3,
    timeout_s: int = 10,
    llm_base_url: str = "http://127.0.0.1:8080/v1",
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    steps_f = (out / "fact_step.jsonl").open("a", encoding="utf-8")
    runs_f = (out / "fact_run.jsonl").open("a", encoding="utf-8")

    agent = LLMAgent(
        variant=variant_id,
        cfg=LLMConfig(
            base_url=llm_base_url,
            model=model_id,
            temperature=0.0,
            max_tokens=512,
            timeout_s=120,
        ),
    )

    for t in tasks:
        run_id = new_id("run")
        run_start = now_ms()

        code = ""
        success = 0
        failure_mode: Optional[str] = None
        step_id = 0

        # Holds the first pytest result when we test starter_code before the loop.
        first_exec_res = None

        # -------------------------
        # Initialize code (starter or propose)
        # -------------------------
        if getattr(t, "starter_code", "").strip():
            code = t.starter_code

            # Log starter code snapshot
            step_id += 1
            steps_f.write(
                StepLog(
                    run_id=run_id,
                    step_id=step_id,
                    step_type="code_starter",
                    prompt=None,
                    response=None,
                    code=code,
                    stdout=None,
                    stderr=None,
                    exit_code=None,
                    started_ms=now_ms(),
                    ended_ms=now_ms(),
                    meta={"category": getattr(t, "category", None), "topic": getattr(t, "topic", None), "starter_check": getattr(t, "starter_check", None)},
                ).to_json()
                + "\n"
            )

            # Run pytest once on starter code
            step_id += 1
            s_start = now_ms()
            exec_res = run_pytest(code, t.tests, timeout_s=timeout_s)
            s_end = now_ms()

            steps_f.write(
                StepLog(
                    run_id=run_id,
                    step_id=step_id,
                    step_type="test",
                    prompt=None,
                    response=None,
                    code=None,
                    stdout=exec_res.stdout,
                    stderr=exec_res.stderr,
                    exit_code=exec_res.exit_code,
                    started_ms=s_start,
                    ended_ms=s_end,
                    meta={"phase": "starter"},
                ).to_json()
                + "\n"
            )

            # If it passes and the task is stability_noop, stop immediately (no LLM calls).
            if exec_res.exit_code == 0 and getattr(t, "category", None) == "stability_noop":
                success = 1
                failure_mode = None

                run_end = now_ms()
                runs_f.write(
                    RunLog(
                        run_id=run_id,
                        task_id=t.task_id,
                        variant_id=variant_id,
                        model_id=model_id,
                        success=success,
                        steps=step_id,
                        started_ms=run_start,
                        ended_ms=run_end,
                        failure_mode=failure_mode,
                        category=getattr(t, "category", None),
                        topic=getattr(t, "topic", None),
                    ).to_json()
                    + "\n"
                )
                continue

            # Otherwise: reuse this failing starter test as the first loop result.
            first_exec_res = exec_res

        else:
            # No starter code -> propose from scratch
            step_id += 1
            s_start = now_ms()
            out1 = agent.propose(signature=t.signature, prompt=t.prompt)
            s_end = now_ms()

            code = out1.code

            # Third-party import guardrail
            bad = has_third_party_import(code)
            if bad:
                failure_mode = "third_party_import"
                success = 0

                steps_f.write(
                    StepLog(
                        run_id=run_id,
                        step_id=step_id,
                        step_type="code_propose",
                        prompt=f"{t.prompt}\n\n{t.signature}",
                        response=out1.response,
                        code=code,
                        stdout=None,
                        stderr=None,
                        exit_code=None,
                        started_ms=s_start,
                        ended_ms=s_end,
                        meta={"third_party_pkg": bad, "category": getattr(t, "category", None), "topic": getattr(t, "topic", None)},
                    ).to_json()
                    + "\n"
                )

                run_end = now_ms()
                runs_f.write(
                    RunLog(
                        run_id=run_id,
                        task_id=t.task_id,
                        variant_id=variant_id,
                        model_id=model_id,
                        success=success,
                        steps=step_id,
                        started_ms=run_start,
                        ended_ms=run_end,
                        failure_mode=failure_mode,
                        category=getattr(t, "category", None),
                        topic=getattr(t, "topic", None),
                    ).to_json()
                    + "\n"
                )
                continue

            # Log propose output
            steps_f.write(
                StepLog(
                    run_id=run_id,
                    step_id=step_id,
                    step_type="code_propose",
                    prompt=f"{t.prompt}\n\n{t.signature}",
                    response=out1.response,
                    code=code,
                    stdout=None,
                    stderr=None,
                    exit_code=None,
                    started_ms=s_start,
                    ended_ms=s_end,
                    meta={"category": getattr(t, "category", None), "topic": getattr(t, "topic", None)},
                ).to_json()
                + "\n"
            )

        # -------------------------
        # Test + fix loop
        # -------------------------
        meta: Optional[dict] = None

        for attempt in range(1, max_attempts + 1):
            # Use starter test result exactly once as the first "test" outcome.
            if attempt == 1 and first_exec_res is not None:
                exec_res = first_exec_res
                first_exec_res = None
            else:
                step_id += 1
                s_start = now_ms()
                exec_res = run_pytest(code, t.tests, timeout_s=timeout_s)
                s_end = now_ms()

                steps_f.write(
                    StepLog(
                        run_id=run_id,
                        step_id=step_id,
                        step_type="test",
                        prompt=None,
                        response=None,
                        code=None,
                        stdout=exec_res.stdout,
                        stderr=exec_res.stderr,
                        exit_code=exec_res.exit_code,
                        started_ms=s_start,
                        ended_ms=s_end,
                        meta=None,
                    ).to_json()
                    + "\n"
                )

            if exec_res.exit_code == 0:
                success = 1
                failure_mode = None
                break

            failure_mode = classify_failure(exec_res.stderr, exec_res.stdout, exec_res.exit_code)

            # Stop if last attempt
            if attempt == max_attempts:
                break

            # Fix step
            step_id += 1
            s_start = now_ms()
            fix_out = agent.fix(
                signature=t.signature,
                prompt=f"[topic={t.topic}] {t.prompt}",
                prev_code=code,
                test_output=(exec_res.stdout or "") + "\n" + (exec_res.stderr or ""),
            )
            s_end = now_ms()

            prev_code = code
            new_code = fix_out.code

            prev_lines = prev_code.splitlines()
            new_lines = new_code.splitlines()

            churn_abs_lines = abs(len(new_lines) - len(prev_lines))
            churn_ratio = (len(new_lines) / max(1, len(prev_lines)))

            meta = {
                "prev_n_lines": len(prev_lines),
                "new_n_lines": len(new_lines),
                "churn_abs_lines": churn_abs_lines,
                "churn_ratio": churn_ratio,
                "category": getattr(t, "category", None),
                "topic": getattr(t, "topic", None),
            }
            
            no_change = new_code.strip() == prev_code.strip()
            meta["no_change"] = no_change
            if no_change:
                failure_mode = "no_progress"
                # log it and break early (otherwise you waste attempts)
                steps_f.write(
                    StepLog(
                        run_id=run_id,
                        step_id=step_id,
                        step_type="code_fix_rejected",
                        prompt=None,
                        response=fix_out.response,
                        code=new_code,
                        stdout=None,
                        stderr=None,
                        exit_code=None,
                        started_ms=s_start,
                        ended_ms=s_end,
                        meta=meta,
                    ).to_json() + "\n"
                )
                break

            # Reject huge rewrites
            if len(prev_lines) >= 6 and churn_ratio > 1.5:
                failure_mode = "rewrite_too_large"
                steps_f.write(
                    StepLog(
                        run_id=run_id,
                        step_id=step_id,
                        step_type="code_fix_rejected",
                        prompt=None,
                        response=fix_out.response,
                        code=new_code,
                        stdout=None,
                        stderr=None,
                        exit_code=None,
                        started_ms=s_start,
                        ended_ms=s_end,
                        meta=meta,
                    ).to_json()
                    + "\n"
                )
                break

            # Accept fix
            code = new_code

            # Third-party import guardrail
            bad = has_third_party_import(code)
            if bad:
                failure_mode = "third_party_import"
                success = 0
                steps_f.write(
                    StepLog(
                        run_id=run_id,
                        step_id=step_id,
                        step_type="code_fix_rejected",
                        prompt=None,
                        response=fix_out.response,
                        code=code,
                        stdout=None,
                        stderr=None,
                        exit_code=None,
                        started_ms=s_start,
                        ended_ms=s_end,
                        meta={**meta, "third_party_pkg": bad},
                    ).to_json()
                    + "\n"
                )
                break

            # Log accepted fix
            steps_f.write(
                StepLog(
                    run_id=run_id,
                    step_id=step_id,
                    step_type="code_fix",
                    prompt=None,
                    response=fix_out.response,
                    code=code,
                    stdout=None,
                    stderr=None,
                    exit_code=None,
                    started_ms=s_start,
                    ended_ms=s_end,
                    meta=meta,
                ).to_json()
                + "\n"
            )

        # -------------------------
        # Write RunLog
        # -------------------------
        run_end = now_ms()
        runs_f.write(
            RunLog(
                run_id=run_id,
                task_id=t.task_id,
                variant_id=variant_id,
                model_id=model_id,
                success=success,
                steps=step_id,
                started_ms=run_start,
                ended_ms=run_end,
                failure_mode=failure_mode,
                category=getattr(t, "category", None),
                topic=getattr(t, "topic", None),
            ).to_json()
            + "\n"
        )

    steps_f.close()
    runs_f.close()
