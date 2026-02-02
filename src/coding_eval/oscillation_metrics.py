# decision_analysis.py
"""
Analyzes decision trajectories from agent evaluation runs.
Usage: python decision_analysis.py --run_dirs "C:\...\runs\w_qwen2_5_coder_7b_instruct_q6_k" "C:\...\runs\w_qwen2_5_coder_7b_instruct_q4_k_m"
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tyro


# ----------------------------
# Error Classification (from your runner)
# ----------------------------

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


# Map specific errors to coarse categories for stability analysis
ERROR_SEVERITY = {
    None: 0,  # PASS
    "harness_missing_pytest": 1,
    "syntax_error": 4,
    "missing_dependency": 3,
    "assertion_failed": 1,
    "timeout": 2,
    "other": 3,
}


# ----------------------------
# Data Structures
# ----------------------------

@dataclass
class DecisionPoint:
    fix_step_id: int
    fix_attempt_num: int
    code_before: str
    code_after: str
    code_changed: bool
    latency_ms: int
    error_type: Optional[str]  # Using your classification
    severity: int
    n_failed_tests: Optional[int]
    approach_class: Optional[str] = None  # NEW: "ring_hash", "modulo_list", etc.


@dataclass
class Trajectory:
    run_id: str
    model_id: str
    variant_id: str  # q4, q6, etc.
    task_id: str
    tier: str
    category: str
    decisions: List[DecisionPoint]
    
    final_success: bool
    n_decisions: int
    error_sequence: List[Optional[str]]
    severity_sequence: List[int]
    
    # Stability metrics
    oscillation_count: int
    n_regressions: int  # severity increased
    progression_type: str
    
    # Timing
    time_to_first_decision_ms: int
    total_duration_ms: int
    approach_sequence: List[Optional[str]] = field(default_factory=list)  # NEW
    approach_oscillation: int = 0  # NEW: count of A-B-A patterns


# ----------------------------
# Parsing Helpers
# ----------------------------

def normalize_code(code: str) -> str:
    if not code:
        return ""
    return '\n'.join(line.strip() for line in code.strip().splitlines())

def codes_are_different(code1: str, code2: str) -> bool:
    return normalize_code(code1) != normalize_code(code2)

def extract_tier(task_id: str) -> str:
    if task_id.startswith('tier'):
        return task_id[:5]  # 'tier1', 'tier2', etc.
    if task_id.startswith('osc'):
        return 'oscillation'
    return 'unknown'

def extract_quantization(variant_id: str) -> str:
    match = re.search(r'[qQ](\d+)', variant_id)
    return f"q{match.group(1)}" if match else variant_id

def parse_n_failed(stdout: str, stderr: str) -> Optional[int]:
    text = (stdout or "") + "\n" + (stderr or "")
    match = re.search(r'(\d+)\s+failed', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    if re.search(r'\d+\s+passed', text, re.IGNORECASE):
        return 0
    return None


# ----------------------------
# Trajectory Building
# ----------------------------

def parse_model_info(dirname: str) -> Tuple[str, str]:
    """Extract model_id and variant_id from directory name like w_qwen2_5_coder_7b_instruct_q6_k"""
    parts = dirname.replace('w_', '').replace('z_', '').split('_')
    
    model_parts = []
    variant_parts = []
    found_quant = False
    
    for part in parts:
        if 'q' in part.lower() and any(c.isdigit() for c in part):
            found_quant = True
        if not found_quant:
            model_parts.append(part)
        else:
            variant_parts.append(part)
    
    model_id = '_'.join(model_parts) if model_parts else dirname
    variant_id = extract_quantization('_'.join(variant_parts)) if variant_parts else 'unknown'
    return model_id, variant_id

def load_trajectories(run_dirs: List[Path]) -> List[Trajectory]:
    trajectories = []
    
    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"Warning: Directory not found: {run_dir}")
            continue
            
        dirname = run_dir.name
        model_id, variant_id = parse_model_info(dirname)
        
        run_file = run_dir / 'fact_run.jsonl'
        step_file = run_dir / 'fact_step.jsonl'
        
        if not run_file.exists() or not step_file.exists():
            print(f"Skipping {dirname}: missing fact files")
            continue
        
        # Load runs
        runs = []
        with open(run_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    runs.append(json.loads(line))
        
        # Load steps and group by run_id
        steps_by_run = defaultdict(list)
        with open(step_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    step = json.loads(line)
                    rid = step.get('run_id')
                    if rid:
                        steps_by_run[rid].append(step)
        
        # Build trajectories for each run
        for run in runs:
            run_id = run.get('run_id')
            if not run_id:
                continue
                
            task_id = run.get('task_id', 'unknown')
            category = run.get('category', 'unknown')
            success = run.get('success', 0)
            
            steps = steps_by_run.get(run_id, [])
            steps.sort(key=lambda x: int(x.get('step_id', 0)))
            
            # Build decision points by pairing code_fix with subsequent test
            decisions = []
            last_code = None
            fix_attempt = 0
            pending_decision = None
            
            for step in steps:
                step_type = step.get('step_type')
                
                if step_type == 'code_starter':
                    last_code = step.get('code', '')
                    
                elif step_type == 'code_fix':
                    fix_attempt += 1
                    new_code = step.get('code', '')
                    started = step.get('started_ms', 0)
                    ended = step.get('ended_ms', 0)
                    latency = ended - started if ended > started else 0
                    
                    code_changed = codes_are_different(last_code, new_code) if last_code else True
                    
                    pending_decision = {
                        'fix_step_id': step.get('step_id'),
                        'fix_attempt_num': fix_attempt,
                        'code_before': last_code or '',
                        'code_after': new_code,
                        'code_changed': code_changed,
                        'latency_ms': latency,
                    }
                    last_code = new_code
                    
                elif step_type == 'test' and pending_decision:
                    stdout = step.get('stdout', '')
                    stderr = step.get('stderr', '')
                    exit_code = step.get('exit_code', 1)
                    
                    error_type = classify_failure(stderr, stdout, exit_code)
                    severity = ERROR_SEVERITY.get(error_type, 3)

                                        
                    # NEW: Detect architectural approach from code
                    approach_class = None
                    code_to_analyze = new_code
                    if "self.ring" in code_to_analyze and "bisect" in code_to_analyze:
                        approach_class = "ring_hash"
                    elif "self.nodes" in code_to_analyze and "% len" in code_to_analyze:
                        approach_class = "modulo_list"
                    elif "deque" in code_to_analyze and "appendleft" in code_to_analyze:
                        approach_class = "deque_approach"
                    elif "list" in code_to_analyze and "pop(0)" in code_to_analyze:
                        approach_class = "list_approach"
                    # Add patterns for your specific oscillation tasks
                    
                    decision = DecisionPoint(
                        fix_step_id=pending_decision['fix_step_id'],
                        fix_attempt_num=pending_decision['fix_attempt_num'],
                        code_before=pending_decision['code_before'],
                        code_after=pending_decision['code_after'],
                        code_changed=pending_decision['code_changed'],
                        latency_ms=pending_decision['latency_ms'],
                        error_type=error_type,
                        severity=severity,
                        n_failed_tests=parse_n_failed(stdout, stderr),
                        approach_class=approach_class,
                    )
                    decisions.append(decision)
                    pending_decision = None
            
            if not decisions:
                continue
            
            # Compute metrics
            error_seq = [d.error_type for d in decisions]
            severity_seq = [d.severity for d in decisions]
            
            oscillations = count_severity_oscillations(severity_seq)
            regressions = count_regressions(severity_seq)
            prog_type = classify_progression(error_seq, severity_seq, bool(success))
            
            first_step = steps[0] if steps else {}
            last_step = steps[-1] if steps else {}
            time_first = decisions[0].latency_ms if decisions else 0
            total_time = last_step.get('ended_ms', 0) - first_step.get('started_ms', 0)

            approach_seq = [d.approach_class for d in decisions]
            approach_osc = count_approach_oscillations(approach_seq)  # create this function
            
            traj = Trajectory(
                run_id=run_id,
                model_id=model_id,
                variant_id=variant_id,
                task_id=task_id,
                tier=extract_tier(task_id),
                category=category,
                decisions=decisions,
                final_success=bool(success),
                n_decisions=len(decisions),
                error_sequence=error_seq,
                severity_sequence=severity_seq,
                oscillation_count=oscillations,
                n_regressions=regressions,
                progression_type=prog_type,
                time_to_first_decision_ms=time_first,
                total_duration_ms=total_time,
                approach_sequence=approach_seq,
                approach_oscillation=approach_osc,
            )
            trajectories.append(traj)
    
    return trajectories


# ----------------------------
# Stability Analysis
# ----------------------------

def count_severity_oscillations(severity_seq: List[int]) -> int:
    """Count A->B->A patterns in severity (instability)"""
    if len(severity_seq) < 3:
        return 0
    
    osc = 0
    for i in range(2, len(severity_seq)):
        a, b, c = severity_seq[i-2], severity_seq[i-1], severity_seq[i]
        # Got better then worse again (or vice versa)
        if a != b and b != c and a == c:
            osc += 1
    return osc

def count_regressions(severity_seq: List[int]) -> int:
    """Count times severity increased (got worse)"""
    if len(severity_seq) < 2:
        return 0
    return sum(1 for i in range(1, len(severity_seq)) 
               if severity_seq[i] > severity_seq[i-1])

def classify_progression(error_seq: List[Optional[str]], severity_seq: List[int], final_success: bool) -> str:
    """Classify trajectory pattern"""
    if not error_seq:
        return 'empty'
    
    if all(e is None for e in error_seq):
        return 'immediate_success'
    
    if final_success:
        if count_severity_oscillations(severity_seq) > 0:
            return 'oscillating_success'
        return 'monotonic_success'
    
    # Failed cases
    osc = count_severity_oscillations(severity_seq)
    if osc >= 2:
        return 'thrashing'
    elif osc == 1:
        return 'oscillating_failure'
    elif severity_seq[-1] == severity_seq[0] and len(severity_seq) > 1:
        return 'stuck'
    else:
        return 'degrading'

def count_approach_oscillations(approach_seq: List[Optional[str]]) -> int:
    """Count architectural A-B-A oscillations (not just severity flips)"""
    # Filter out None/unknown
    clean_seq = [a for a in approach_seq if a and a != "unknown"]
    if len(clean_seq) < 3:
        return 0
    
    osc = 0
    for i in range(2, len(clean_seq)):
        a, b, c = clean_seq[i-2], clean_seq[i-1], clean_seq[i]
        # A -> B -> A pattern (switched away then came back)
        if a != b and b != c and a == c:
            osc += 1
    return osc

# ----------------------------
# Aggregation & Reporting
# ----------------------------

@dataclass
class Args:
    run_dirs: List[str]  # List of directory paths
    output_json: Optional[str] = None

def aggregate_by_tier_quant_model(trajectories: List[Trajectory]) -> Dict:
    """Aggregate metrics by (tier, quantization, model)"""
    groups = defaultdict(lambda: {
        'count': 0,
        'successes': 0,
        'oscillations': [],
        'regressions': [],
        'thrashing_count': 0,
        'stuck_count': 0,
        'n_decisions': [],
        'latencies': [],
        'approach_osc': [],
    })
    
    for t in trajectories:
        key = (t.tier, t.variant_id, t.model_id)
        g = groups[key]
        
        g['count'] += 1
        g['successes'] += 1 if t.final_success else 0
        g['oscillations'].append(t.oscillation_count)
        g['regressions'].append(t.n_regressions)
        g['n_decisions'].append(t.n_decisions)
        g['latencies'].append(t.total_duration_ms)
        g['approach_osc'].append(t.approach_oscillation)
        
        if t.progression_type == 'thrashing':
            g['thrashing_count'] += 1
        if t.progression_type == 'stuck':
            g['stuck_count'] += 1
            
    # Compute summaries
    results = {}
    for key, g in groups.items():
        n = g['count']
        if n == 0:
            continue
            
        results[key] = {
            'tier': key[0],
            'quantization': key[1],
            'model': key[2],
            'n_runs': n,
            'success_rate': g['successes'] / n,
            'thrash_rate': g['thrashing_count'] / n,
            'stuck_rate': g['stuck_count'] / n,
            'avg_oscillations': statistics.mean(g['oscillations']) if g['oscillations'] else 0,
            'avg_regressions': statistics.mean(g['regressions']) if g['regressions'] else 0,
            'avg_decisions': statistics.mean(g['n_decisions']) if g['n_decisions'] else 0,
            'med_latency_ms': statistics.median(g['latencies']) if g['latencies'] else 0,
            'avg_approach_osc': statistics.mean(g['approach_osc']) if g['approach_osc'] else 0,
        }
    
    return results

def print_insights(results: Dict):
    """Print key findings"""
    print("\n" + "="*80)
    print("DECISION STABILITY ANALYSIS")
    print("="*80)
    

    print("\n--- Architectural Oscillation Analysis ---")
    osc_data = [d for d in results.values() if d.get('avg_approach_osc', 0) > 0]
    if osc_data:
        print(f"Detected approach oscillation in {len(osc_data)} configurations")
        for d in osc_data:
            print(f"  {d['tier']} {d['quantization']}: {d['avg_approach_osc']:.2f} osc/run")
    else:
        print("No architectural oscillation detected (0% A-B-A patterns)")
        print("Models exhibit entrenchment even on structurally ambiguous tasks")
        # Group by tier for comparison
        by_tier = defaultdict(list)
        for key, data in results.items():
            by_tier[data['tier']].append(data)
    
    print("\n--- Stability by Tier ---")
    print(f"{'Tier':<8} {'Quant':<8} {'Model':<25} {'Success':<10} {'Thrash%':<10} {'Osc/run':<10} {'AvgFixes':<10}")
    print("-" * 100)
    
    for tier in sorted(by_tier.keys()):
        for data in by_tier[tier]:
            print(f"{data['tier']:<8} {data['quantization']:<8} {data['model'][:25]:<25} "
                  f"{data['success_rate']:<10.2%} {data['thrash_rate']:<10.2%} "
                  f"{data['avg_oscillations']:<10.2f} {data['avg_decisions']:<10.1f}")
    
    # Specific insight: q4 vs q6 comparison
    print("\n--- Quantization Impact (Q4 vs Q6) ---")
    q4_data = [d for d in results.values() if d['quantization'] == 'q4']
    q6_data = [d for d in results.values() if d['quantization'] == 'q6']
    
    if q4_data and q6_data:
        avg_thrash_q4 = statistics.mean([d['thrash_rate'] for d in q4_data])
        avg_thrash_q6 = statistics.mean([d['thrash_rate'] for d in q6_data])
        avg_osc_q4 = statistics.mean([d['avg_oscillations'] for d in q4_data])
        avg_osc_q6 = statistics.mean([d['avg_oscillations'] for d in q6_data])
        
        print(f"Average thrashing rate - Q4: {avg_thrash_q4:.2%}, Q6: {avg_thrash_q6:.2%}")
        print(f"Average oscillations/run - Q4: {avg_osc_q4:.2f}, Q6: {avg_osc_q6:.2f}")
        
        if avg_thrash_q4 > avg_thrash_q6 * 1.2:
            print("FINDING: Q4 shows significantly higher decision instability than Q6")
        elif abs(avg_thrash_q4 - avg_thrash_q6) < 0.05:
            print("FINDING: No meaningful stability difference between Q4 and Q6")
        else:
            print("FINDING: Q4 and Q6 show comparable stability profiles")

def main(args: Args):
    run_dirs = [Path(d) for d in args.run_dirs]
    
    print(f"Loading trajectories from {len(run_dirs)} directories...")
    trajectories = load_trajectories(run_dirs)
    print(f"Loaded {len(trajectories)} trajectories")
    
    if not trajectories:
        print("No trajectories found.")
        return
    
    results = aggregate_by_tier_quant_model(trajectories)
    print_insights(results)
    
    if args.output_json:
        import json as jsonlib
        with open(args.output_json, 'w') as f:
            # Convert tuple keys to strings for JSON
            json_results = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in results.items()}
            jsonlib.dump(json_results, f, indent=2)
        print(f"\nSaved detailed results to {args.output_json}")

if __name__ == "__main__":
    main(tyro.cli(Args))
