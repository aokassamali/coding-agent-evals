# Decision Rigidity in Local Code Models: An Empirical Study

## Summary

This study evaluates decision-making patterns in local LLMs (3B-7B parameter range) during iterative code repair tasks. Across 1,000+ trajectories, we find that models exhibit **strong entrenchment bias**—when facing failures, they continue patching within their initial approach rather than exploring alternatives. This pattern holds even on tasks explicitly designed to induce architectural oscillation.

**Key Finding:** 0% architectural oscillation across all models and task types, with stuck rates reaching 22% on ambiguous tasks. Models need external intervention to recognize when to abandon a failing strategy.

---

## Methodology

### Models Evaluated

| Model | Parameters | Quantization | Family |
|-------|------------|--------------|--------|
| Qwen2.5-Coder 7B Instruct | 7B | Q6_K | Qwen |
| Qwen2.5-Coder 7B Instruct | 7B | Q4_K_M | Qwen |
| Qwen2.5-Coder 3B Instruct | 3B | Q6_K | Qwen |
| CodeGemma 7B IT | 7B | Q6_K | Gemma |
| DeepSeek Coder 6.7B Instruct | 6.7B | Q6_K | DeepSeek |

### Task Design

**Tiered Tasks (n=200):** Five difficulty tiers with progressively complex reasoning requirements:

| Tier | Description | Example Patterns |
|------|-------------|------------------|
| **Tier 1** | Trivial single-line fixes | Off-by-one errors, wrong operators, typos. Baseline sanity—should be near-perfect. |
| **Tier 2** | Parsing / edge-case clarifications | Small fixes requiring careful test reading. Multiple asserts to satisfy. |
| **Tier 3** | Two-stage fixes (fix reveals next failure) | Fix #1 passes first assert but reveals assert #2. Model must avoid "fixing the last failure by breaking the first." |
| **Tier 4** | Correct fix requires structural change | Data mutation bugs, wrong caching/memoization, ordering stability issues, subtle type coercion. Standard library only. |
| **Tier 5** | Cross-cutting reasoning | Context managers + exception paths, async/sync interop, weakref/finalizer/atexit lifecycle, pickle protocols. Requires understanding multiple interacting systems. |

**Oscillation Tasks (n=32):** Deliberately ambiguous problems where multiple valid architectural approaches exist:
- Queue: list with head pointer vs `collections.deque`
- Cache: LRU (recency) vs LFU (frequency) eviction
- Graph: adjacency list vs adjacency matrix
- Hash table: chaining vs open addressing

### Metrics

- `fix_p@k`: Pass rate after k fix attempts
- `stuck_rate`: Trajectories where model made no progress despite multiple attempts
- `thrash_rate`: Trajectories with repeated regressions (severity oscillation)
- `approach_osc`: Architectural strategy switches (A→B→A patterns)
- `imprRate`: Improvement rate (% of failing tasks that eventually pass)

---

## Results

### Performance by Tier (All Models)

#### CodeGemma 7B IT (Q6_K) — Total: 200 tasks
| Tier | n | fix_p@1 | fix_p@4 | Impr. Rate |
|------|---|---------|---------|------------|
| 1 | 30 | 86.7% | 90.0% | 90.0% |
| 2 | 30 | 63.3% | 80.0% | 80.0% |
| 3 | 30 | 56.7% | 60.0% | 60.0% |
| 4 | 50 | 46.0% | 54.0% | 54.2% |
| 5 | 60 | 21.7% | 28.3% | 39.5% |

#### DeepSeek Coder 6.7B Instruct (Q6_K) — Total: 200 tasks
| Tier | n | fix_p@1 | fix_p@4 | Impr. Rate |
|------|---|---------|---------|------------|
| 1 | 30 | 90.0% | 90.0% | 90.0% |
| 2 | 30 | 63.3% | 80.0% | 80.0% |
| 3 | 30 | 56.7% | 60.0% | 60.0% |
| 4 | 50 | 46.0% | 52.0% | 52.1% |
| 5 | 60 | 21.7% | 30.0% | 42.1% |

#### Qwen2.5-Coder 7B Instruct (Q6_K) — Total: 200 tasks
| Tier | n | fix_p@1 | fix_p@4 | Impr. Rate |
|------|---|---------|---------|------------|
| 1 | 30 | 90.0% | 90.0% | 90.0% |
| 2 | 30 | 63.3% | 80.0% | 80.0% |
| 3 | 30 | 56.7% | 60.0% | 60.0% |
| 4 | 50 | 44.0% | 50.0% | 50.0% |
| 5 | 60 | 21.7% | 26.7% | 39.5% |

#### Qwen2.5-Coder 7B Instruct (Q4_K_M) — Total: 200 tasks
| Tier | n | fix_p@1 | fix_p@4 | Impr. Rate |
|------|---|---------|---------|------------|
| 1 | 30 | 90.0% | 90.0% | 90.0% |
| 2 | 30 | 63.3% | 80.0% | 80.0% |
| 3 | 30 | 53.3% | 53.3% | 53.3% |
| 4 | 50 | 44.0% | 50.0% | 50.0% |
| 5 | 60 | 21.7% | 28.3% | 39.5% |

#### Qwen2.5-Coder 3B Instruct (Q6_K) — Total: 200 tasks
| Tier | n | fix_p@1 | fix_p@4 | Impr. Rate |
|------|---|---------|---------|------------|
| 1 | 30 | 90.0% | 90.0% | 90.0% |
| 2 | 30 | 63.3% | 80.0% | 80.0% |
| 3 | 30 | 56.7% | 60.0% | 60.0% |
| 4 | 50 | 44.0% | 52.0% | 52.1% |
| 5 | 60 | 29.4% | 38.2% | 39.4% |

---

### Comparative Analysis

#### Model Family Comparison (All Q6, 7B-class, n=200 each)

| Model | fix_p@1 | fix_p@4 | Tier 5 fix_p@4 | Median Latency |
|-------|---------|---------|----------------|----------------|
| CodeGemma 7B | 49.0% | 56.5% | 28.3% | 3,916ms |
| DeepSeek 6.7B | 49.5% | 56.5% | 30.0% | 6,879ms |
| Qwen 7B | 49.0% | 55.0% | 26.7% | 7,474ms |

**Finding:** Model families perform within 2pp of each other on aggregate metrics. DeepSeek shows slight edge on Tier 5 (+3pp over Qwen) but at higher latency. CodeGemma is fastest (3.9s median) while maintaining comparable accuracy.

#### Size Comparison (Qwen Q6_K, n=200 each)

| Model | fix_p@1 | fix_p@4 | Tier 5 fix_p@4 | Median Latency |
|-------|---------|---------|----------------|----------------|
| Qwen 7B | 49.0% | 55.0% | 26.7% | 7,474ms |
| Qwen 3B | 54.6% | 62.1% | 38.2% | 5,479ms |

**Finding:** Qwen 3B outperforms Qwen 7B by 7pp on fix_p@4 and 11pp on Tier 5. This may reflect differences in instruction tuning quality or task alignment. The 3B model also runs 27% faster (5.5s vs 7.5s median)—a case where smaller is both faster and better.

#### Quantization Comparison (Qwen 7B, n=200 each)

| Quantization | fix_p@1 | fix_p@4 | Tier 5 fix_p@4 | Median Latency |
|--------------|---------|---------|----------------|----------------|
| Q6_K | 49.0% | 55.0% | 26.7% | 7,474ms |
| Q4_K_M | 48.5% | 54.5% | 28.3% | 4,284ms |

**Finding:** No meaningful accuracy difference between Q4 and Q6 (<1pp on aggregate). Q4 runs 43% faster (4.3s vs 7.5s median). Teams can deploy Q4 quantizations for code repair tasks without accuracy loss.

---

### Oscillation Task Analysis

| Metric | Easy Tasks (v7) | Hard Tasks (v8) |
|--------|-----------------|-----------------|
| Success Rate | 84.6% | 64.9% |
| Stuck Rate | 7.7% | 22.8% |
| Thrash Rate | 0% | 0% |
| Approach Oscillation | 0% | 0% |
| Avg Decisions | 1.15 | 1.46 |

**Critical Finding:** Even on tasks designed to induce strategy switching, models show **0% architectural oscillation**. When task difficulty increases:
- Success drops from 85% → 65%
- Stuck rate nearly triples (8% → 23%)
- But models never backtrack to try alternative approaches

Models exhibit **entrenchment bias**: they commit to their initial approach and continue patching within that paradigm rather than exploring alternatives, even when repeated failures suggest the approach itself may be wrong.

---

## Implications for Agent Design

These findings suggest that local models lack the self-reflection capability to recognize when their current approach is failing and pivot to alternatives. This has direct implications for agent harness design:

1. **Entrenchment Detection:** Agent frameworks should monitor for "N consecutive failed patches without strategy change" as a trigger for intervention—not just error count.

2. **External Guidance Required:** Models benefit from explicit nudges to consider alternative approaches rather than continuing to patch within a failing paradigm. The stuck rate (22%) represents trajectories where decision-time guidance could unlock progress.

3. **Quantization is Not the Bottleneck:** Teams can deploy Q4 quantizations without meaningful accuracy loss, enabling 40%+ faster inference with equivalent results.

4. **Model Size vs. Quality:** Larger models don't guarantee better performance on iterative repair tasks. Instruction tuning quality and task alignment may matter more than raw parameter count.

---

## Reproduction

```bash
# run_ablations and run_ablations_oscillations are for when you have a folder with multiple local models downloaded. 
#For my specific machine I needed to remove the model from cache and restart the process with a new model.
# Run tiered evaluation
python run_eval.py --tasks-path data/tasks/v6_fixedtiers.jsonl --out-root runs/eval

# Run oscillation tasks  
python run_eval.py --tasks-path data/tasks/v7_oscillationtasks.jsonl --out-root runs/osc

# Generate ablation readout
python ablation_readout.py --run-dirs runs/eval

# Generate oscillation_metrics
python metrics.py --run_dirs runs/osc --output_json results.json
```

*Built as part of independent research on LLM agent evaluation frameworks, January 2026.*