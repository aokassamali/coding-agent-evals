# Decision Rigidity in Local Code Models - Research Notes (v9/v11)

## Abstract
Do small-scale code models exhibit path dependence (decision rigidity) under iterative refinement? We evaluate local 3B-7B models on two synthetic suites and track approach changes across multiple fix attempts with a heuristic classifier. We find low approach switching under deterministic settings and strong sensitivity to temperature, but these findings are bounded by synthetic task design and guardrail effects.

**Research question:** Do small-scale local code models entrench on initial solution strategies when iteratively fixing code?

Results reported here were computed on Feb 4, 2026 using `runs/yyy_*`, `runs/zzz_*`, and `runs/temp2_*`.

## Key Findings

- **Under deterministic decoding (T=0.0) and the current harness/guardrails**, we observed **zero approach oscillation (A→B→A)** across **910 approach decisions** on **v11**.
- **In v9 at T=0.0**, additional attempts (**2–4**) add only **~1–3 percentage points** of success over attempt 1.
- **On the 33 v11 failures for Qwen2.5 Coder 7B (Q6_K)**, a **prompt-only strategy perturbation** recovered **0/33**; remaining failures were largely constrained by guardrail outcomes (e.g., `no_progress`, `rewrite_too_large`).
- **Temperature sensitivity (single run; not replicated):** at **T=0.2**, v11 success reached **100%** vs **83.5%** at **T=0.0**, and observed switching increased—suggesting decoding stochasticity can materially change “rigidity” signals in this setup.


## What This Project Does

- Measures fix success across multiple attempts on two synthetic task suites.
- Tracks approach changes between attempts using a heuristic, topic-aware classifier.
- Audits run integrity (missing tasks, fingerprints, cross-model diffs).
- Reports Kaplan-Meier survival curves to characterize entrenchment over attempts.
- Tests a limited **intervention** (strategy perturbation) on a failed subset.

## What This Project Does Not Do

- It does not establish generalization to external benchmarks (all tasks are synthetic).
- It does not provide ground-truth architectural intent for "approach" labels.
- It does not measure full model capability absent guardrails (guardrails shape outcomes).
- It does not run temperature sweeps or large statistical replications.
- It does not claim causal conclusions about "entrenchment" beyond these task suites.

## Hardware, Runtime, and Cost Transparency

- GPU: GTX 1080 Ti
- CPU: Ryzen 9 5900X
- RAM: 32GB
- Server: `llama-server` (OpenAI-compatible `/v1/chat/completions`)
- Context: 4096
- Default inference: `temperature=0.0`, `max_tokens=512`
- Attempts: `max_attempts=4`
- Per-test timeout: `timeout_s=3`

Observed wall time (this machine):
- 5-model ablation: ~3 hours
- Single model run: ~30 minutes
- Total ~15 runs completed over 6 days

Electricity cost was not estimated.

Guardrails applied during evaluation:

- `no_progress`: reject a fix if `new_code.strip() == prev_code.strip()`
- `rewrite_too_large`: reject if `len(new_lines)/len(prev_lines) > 1.5` and `len(prev_lines) >= 6`
- `rewrite_max_abs_lines=0` (disabled)

These guardrails are part of the measurement, not incidental. They materially affect failure modes.

## Task Suites

### v9_fixedtiers (bugfix suite)

- 200 tasks, stdlib-only
- Tier distribution:
  - Tier 1: 20 tasks (trivial)
  - Tier 2: 50 tasks (edge cases / parsing)
  - Tier 3: 60 tasks (two-stage fixes)
  - Tier 4: 50 tasks (structural fixes)
  - Tier 5: 20 tasks (cross-cutting reasoning)

File: `data/tasks/v9_fixedtiers.jsonl`

### v11_osc_total_200 (oscillation suite)

- 200 tasks, 8 topics, 25 tasks each
- Topics: `string_match`, `interval_merge`, `sorting`, `search`, `heap`,
  `priority_queue`, `topological_sort`, `trie`
- Designed to allow multiple plausible architectural approaches

File: `data/tasks/v11_osc_total_200.jsonl`

## Experimental Design

- v9 bugfix runs: `runs/yyy_*` (5 models)
- v11 oscillation runs: `runs/zzz_*` (5 models)
- Single temperature-0.2 run: `runs/temp2_qwen2_5_coder_7b_instruct_q6_k`
- Intervention testing runs: `runs/v2_inj_*`, `runs/v3_inj_*`

Each run directory contains:

- `fact_run.jsonl` (run-level results)
- `fact_step.jsonl` (step-level traces)
- `server_models.json` (LLM server fingerprint)

## Metrics

Bugfix metrics (from `ablation_readout.py`):

- `fix_p@k`: fraction solved after k fix attempts
- `medAtt`: median fix attempts among successful runs
- `imprRate`: fraction of runs where best failure count improves vs first
- `thrash2`: fail-curve oscillation (failure count decreases then increases)

Oscillation metrics (from `oscillation_metrics.py`):

- `switch rate`: approach changes across fix attempts
- `early deviation`: first fix approach != starter approach
- `stuck`: no success and no meaningful progress
- `NoDec`: no accepted decision points (guardrail rejections)
- `ApproachOsc`: A->B->A patterns (oscillation back to a previous approach)

**Switch rate normalization:** switch rate is computed as total approach changes divided by total tasks. Tasks with fewer than two decisions cannot contribute a switch (they count as 0).

Survival / entrenchment (from `multi_model_analysis.py`):

- Kaplan-Meier hazard by attempt
- Decreasing hazard suggests entrenchment (probability of success declines)

**Standard error:** For binomial proportions, SE = sqrt(p(1-p)/n), with n=200 unless otherwise noted.

## Classifier Validation (Sanity Check)

We do not yet have an exhaustive manual audit. As an automatic consistency check on v11, **685/685 code-fix decisions** that had explicit ground-truth `approaches` lists matched the allowed list. The remaining 225 decisions used heuristic-only labels. This is a sanity check, not a proof of ground-truth correctness.

### Manual Audit

To validate the heuristic classifier, we sampled 50 random decisions from v11 and manually labeled approaches blind to the classifier output.

| Metric | Value |
|---|---|
| Sample size | 50 |
| Agreement | 100% |
| Cohen's kappa | 1.00 |

This audit was performed by the author based on code inspection without viewing heuristic labels. The raw sample and labels are available in `data/results/manual_audit_v11_samples.jsonl` and `data/results/manual_audit_v11_labels.jsonl`. The perfect agreement likely reflects the fact that v11 tasks are intentionally separable and the classifier keys off obvious code cues; it should not be read as general validation for harder or less structured tasks.

## Results

### v9 bugfix suite (B_debug, 200 tasks/model)

Aggregate metrics:

| Model | fix_p@1 | fix_p@4 | medAtt | rewrite_too_large | lat_p50 (ms) |
|---|---:|---:|---:|---:|---:|
| CodeGemma 7B IT Q6_K | 81.0% +/- 2.8% | 82.0% +/- 2.7% | 1.00 | 3.5% | 3275 |
| DeepSeek Coder 6.7B Q6_K | 83.0% +/- 2.7% | 83.5% +/- 2.6% | 1.00 | 1.5% | 4400 |
| Qwen2.5 Coder 3B Q6_K | 80.5% +/- 2.8% | 82.0% +/- 2.7% | 1.00 | 3.0% | 4644 |
| Qwen2.5 Coder 7B Q4_K_M | 84.0% +/- 2.6% | 84.0% +/- 2.6% | 1.00 | 3.0% | 4207 |
| Qwen2.5 Coder 7B Q6_K | 81.5% +/- 2.7% | 84.0% +/- 2.6% | 1.00 | 1.5% | 5146 |

Tier ranges across models:

- Tier 1: fix_p@1 = 100%, fix_p@4 = 100%
- Tier 2: fix_p@1 = 84-86%, fix_p@4 = 84-86%
- Tier 3: fix_p@1 = 88.3-91.7%, fix_p@4 = 88.3-91.7%
- Tier 4: fix_p@1 = 70-78%, fix_p@4 = 72-78%
- Tier 5: fix_p@1 = 50-55%, fix_p@4 = 55-65%

Interpretation: extra attempts add only 1-3pp on aggregate; difficulty separation exists by tier, but cross-model differences are small.

### Quantization comparison (Qwen 7B Q4_K_M vs Q6_K)

| Suite | Metric | Q4_K_M | Q6_K |
|---|---|---:|---:|
| v9 bugfix | fix_p@1 | 84.0% | 81.5% |
| v9 bugfix | fix_p@4 | 84.0% | 84.0% |
| v11 oscillation | success | 83.5% | 83.5% |
| v11 oscillation | switch rate | 0.08/run | 0.09/run |

Interpretation: quantization has minimal impact on accuracy and switching in these suites.

### v11 oscillation suite (B_debug, 200 tasks/model)

Approach detection coverage: 100% on 910 decisions.

| Model | Success | Switch rate | Early dev | Stuck | NoDec |
|---|---:|---:|---:|---:|---:|
| CodeGemma 7B IT Q6_K | 83.5% +/- 2.6% | 0.09/run | 16.7% | 3.0% | 8.0% |
| DeepSeek Coder 6.7B Q6_K | 82.0% +/- 2.7% | 0.07/run | 13.3% | 2.5% | 9.5% |
| Qwen2.5 Coder 3B Q6_K | 83.0% +/- 2.7% | 0.09/run | 15.9% | 4.0% | 8.5% |
| Qwen2.5 Coder 7B Q4_K_M | 83.5% +/- 2.6% | 0.08/run | 15.0% | 3.0% | 8.5% |
| Qwen2.5 Coder 7B Q6_K | 83.5% +/- 2.6% | 0.09/run | 15.7% | 3.5% | 8.0% |

Approach oscillation (A->B->A) remained 0 across all models. Hazard rates decreased over attempts, consistent with entrenchment within this harness.

### Kaplan-Meier survival curves (v11)

![Kaplan-Meier survival curves](data/results/plots/kaplan_meier_v11.svg)

**Interpretation caveat:** Decreasing hazard over attempts is consistent with entrenchment but also consistent with task-level selection effects: if easier tasks are solved on early attempts, the remaining pool is harder by construction. These interpretations are not mutually exclusive, and the current design cannot distinguish them.

### Intervention testing (strategy perturbation, failed subset)

Subset: 33 failed v11 tasks for Qwen2.5 7B Q6_K.

Prompts:

- Strategy-shift prompt after attempt 2 (guardrails relaxed)
- Conservative prompt after attempt 2 (guardrails unchanged)

Results:

- Base failures: 0% success (no_progress 17, rewrite_too_large 16)
- Strategy shift: 0% success (no_progress 33)
- Conservative: 0% success (no_progress 17, harness_missing_pytest 16)

Interpretation: intervention did not recover failures; guardrails dominate this subset.

### Sensitivity run (temperature=0.2)

Single model: Qwen2.5 7B Q6_K on v11.

- Success = 100%
- Switch rate = 16.9%
- Early deviation = 8.9%

This is a single run and should not be over-interpreted. It demonstrates that temperature shifts dynamics materially.

## Discussion

- The harness detects low approach switching in deterministic runs.
- Guardrails are a major source of failure modes in the hardest tasks.
- The oscillation suite does not yet induce A->B->A behavior under deterministic settings.
- Temperature affects both success and switching, but this is not yet characterized systematically.

**Temperature hypothesis.** The T=0.2 result (100% success, 16.9% switch rate vs. 83.5% success, ~0.08 switch rate at T=0.0) suggests that apparent entrenchment may be partially or fully attributable to deterministic decoding rather than model-level strategy commitment. Under greedy decoding, a model that fails on attempt 1 will replay similar token sequences on subsequent attempts, mechanically producing low switching. Disambiguating "entrenchment as cognitive limitation" from "entrenchment as decoding artifact" requires temperature-controlled replication across multiple seeds, which is out of scope for this project but represents a natural follow-up.

## Limitations 

- Synthetic tasks only; no external benchmarks.
- Approach detection is heuristic, not ground truth.
- Low discriminative power in v9 (most tasks are all-pass or all-fail).
- Guardrails constrain what constitutes "progress."
- Limited replication and no temperature sweep.
- Hardware limits model size, context, and experiment scale.
- Classifier validation is partial. The heuristic classifier was sanity-checked against allowed approach lists, and a 50-sample manual audit showed 100% agreement (kappa 1.00). This is still single-rater and limited in scope.
- Temperature confound is unresolved. A single T=0.2 run suggests temperature materially affects both success and switching, but this project does not include a systematic temperature sweep. Entrenchment findings apply to deterministic (T=0.0) settings only.

These limitations are fundamental to the current results and are explicitly acknowledged here.

## Threats to Validity

**Internal validity:**
- Guardrails (`no_progress`, `rewrite_too_large`) mechanically prevent certain fix attempts, confounding "model won't change approach" with "harness rejected the change." The intervention results (0/33 recovery, dominated by guardrail rejections) illustrate this limitation.
- Deterministic decoding at T=0.0 may produce low switching as a mechanical artifact rather than a model property. The single T=0.2 run suggests this is a real concern.
- With `max_attempts=4`, there are only 3 opportunities to observe approach changes, limiting power to detect oscillation patterns.

**External validity:**
- All tasks are synthetic and stdlib-only; generalization to real-world codebases or external benchmarks (e.g., SWE-bench, HumanEval) is unknown.
- Models tested are 3B-7B parameter local models under 4096 context; findings may not transfer to larger frontier models or longer contexts.
- Guardrail settings are arbitrary and would affect results if changed.

## Future Work

- Temperature-controlled replication across multiple seeds to separate decoding artifacts from model-level entrenchment.
- External benchmark validation (e.g., SWE-bench or HumanEval subsets) to test generalization beyond synthetic tasks.
- Scale effects: evaluate larger models (13B+) or longer contexts to see whether entrenchment diminishes with capability.

## Related Work

This project builds on the growing literature on LLM self-refinement:

- **Madaan et al., "Self-Refine"** (arXiv:2303.17651): Iterative refinement with self-feedback on larger models (GPT-3.5/4). Our work differs in focusing on small local models (3B-7B) and measuring approach switching rather than output quality.
- **Shinn et al., "Reflexion"** (arXiv:2303.11366): Verbal reinforcement learning with explicit memory. Our harness does not provide cross-attempt memory, which may contribute to observed entrenchment.
- **Gou et al., "CRITIC"** (arXiv:2305.11738): Self-correction with external tool feedback. Our feedback is limited to test pass/fail signals.
- **Wang et al., "Self-Consistency"** (arXiv:2203.11171): Sampling multiple reasoning paths and marginalizing. Our deterministic (T=0.0) setting eliminates this mechanism by design.

A key difference from prior work is our focus on **strategy-level** behavior (do models switch approaches?) rather than **output-level** improvement (do outputs get better?). Our findings suggest that under deterministic decoding, small models exhibit low approach switching, but this may not generalize to the stochastic, larger-model settings studied in prior work.

## Reproduction

### Windows (PowerShell)

- The following scripts assume you've already loaded your model into the llama.cpp host.
- If you have multiple models, you can edit `run_ablations.ps1` and point it to your folder with local GGUF files.

```
python src/coding_eval/run_eval.py ^
  --tasks_path data/tasks/v11_osc_total_200.jsonl ^
  --model_id qwen2_5_coder_7b_instruct_q6_k ^
  --run_tag zzz_qwen2_5_coder_7b_instruct_q6_k ^
  --max_attempts 4 ^
  --timeout-s 3 ^
  --variants B_debug
```

### Unix (bash)

```
python src/coding_eval/run_eval.py \
  --tasks_path data/tasks/v11_osc_total_200.jsonl \
  --model_id qwen2_5_coder_7b_instruct_q6_k \
  --run_tag zzz_qwen2_5_coder_7b_instruct_q6_k \
  --max_attempts 4 \
  --timeout-s 3 \
  --variants B_debug
```

Ablation readout:

```
python src/coding_eval/ablation_readout.py --run_dirs runs/yyy_* --tasks_path data/tasks/v9_fixedtiers.jsonl
```

Oscillation metrics:

```
python src/coding_eval/oscillation_metrics.py --run_dirs runs/zzz_* --tasks_path data/tasks/v11_osc_total_200.jsonl
```

Multi-model analysis:

```
python src/coding_eval/multi_model_analysis.py --run_dirs runs/zzz_* --tasks_path data/tasks/v11_osc_total_200.jsonl --output data/results/multi_model_results_zzz
```

Audit runs:

```
python src/coding_eval/audit_runs.py --run_dirs runs/yyy_* --tasks_path data/tasks/v9_fixedtiers.jsonl --diff_tasks
```

## Artifacts

- v9 tasks: `data/tasks/v9_fixedtiers.jsonl`
- v11 tasks: `data/tasks/v11_osc_total_200.jsonl`
- v11 failed subset: `data/tasks/v11_osc_failed_zzz_qwen2_5_coder_7b_instruct_q6_k.jsonl`
- v9 runs: `runs/yyy_*`
- v11 runs: `runs/zzz_*`
- Temp=0.2 run: `runs/temp2_qwen2_5_coder_7b_instruct_q6_k`
- Multi-model results: `data/results/multi_model_results_zzz/`, `data/results/multi_model_results_temp2/`
- Reports: `data/results/reports/`
- Manual audit samples: `data/results/manual_audit_v11_samples.jsonl`
- Manual audit labels: `data/results/manual_audit_v11_labels.jsonl`
