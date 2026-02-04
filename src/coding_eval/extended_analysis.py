"""
Extended Analysis Suite for LLM Debugging Evaluation

This script adds four critical analyses:
1. Manual annotation interface for "undetected" approach switches
2. Starter deviation analysis (first-fix approach != starter approach)
3. Survival analysis with Kaplan-Meier curves
4. Enhanced oscillation metrics

Usage:
    # Manual annotation
    python extended_analysis.py annotate --run_dir runs/my_experiment --n_sample 50

    # Starter deviation analysis
    python extended_analysis.py deviation --run_dir runs/my_experiment
    
    # Survival analysis
    python extended_analysis.py survival --run_dir runs/my_experiment
    
    # All analyses
    python extended_analysis.py all --run_dir runs/my_experiment
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tyro
from approach_detection import detect_approach, load_tasks_metadata


# ============================================================================
# DATA LOADING
# ============================================================================

@dataclass
class StepLog:
    """Represents a single step in a trajectory"""
    run_id: str
    step_id: int
    step_type: str
    code: Optional[str]
    stdout: Optional[str]
    stderr: Optional[str]
    exit_code: Optional[int]
    meta: Optional[Dict[str, Any]]
    
    @classmethod
    def from_dict(cls, d: Dict) -> StepLog:
        return cls(
            run_id=d['run_id'],
            step_id=d['step_id'],
            step_type=d['step_type'],
            code=d.get('code'),
            stdout=d.get('stdout'),
            stderr=d.get('stderr'),
            exit_code=d.get('exit_code'),
            meta=d.get('meta', {})
        )


@dataclass
class RunLog:
    """Represents a complete run"""
    run_id: str
    task_id: str
    variant_id: str
    model_id: str
    success: int
    steps: int
    failure_mode: Optional[str]
    category: Optional[str]
    topic: Optional[str]
    
    @classmethod
    def from_dict(cls, d: Dict) -> RunLog:
        return cls(
            run_id=d['run_id'],
            task_id=d['task_id'],
            variant_id=d['variant_id'],
            model_id=d['model_id'],
            success=d['success'],
            steps=d['steps'],
            failure_mode=d.get('failure_mode'),
            category=d.get('category'),
            topic=d.get('topic')
        )


@dataclass
class Trajectory:
    """Complete trajectory with steps and metadata"""
    run: RunLog
    steps: List[StepLog]
    
    @property
    def starter_code(self) -> Optional[str]:
        """Get starter code if exists"""
        for step in self.steps:
            if step.step_type == 'code_starter':
                return step.code
        return None
    
    @property
    def starter_approach(self) -> Optional[str]:
        """Get starter approach class"""
        for step in self.steps:
            if step.step_type == 'code_starter' and step.meta:
                return step.meta.get('approach_class')
        return None
    
    @property
    def fix_steps(self) -> List[StepLog]:
        """Get all code_fix steps (excludes rejected fixes)"""
        return [s for s in self.steps if s.step_type == 'code_fix']
    
    @property
    def first_fix_approach(self) -> Optional[str]:
        """Get approach class of first accepted fix"""
        fixes = self.fix_steps
        if fixes and fixes[0].meta:
            return fixes[0].meta.get('approach_class')
        return None
    
    @property
    def final_fix_approach(self) -> Optional[str]:
        """Get approach class of last accepted fix"""
        fixes = self.fix_steps
        if fixes and fixes[-1].meta:
            return fixes[-1].meta.get('approach_class')
        return None
    
    @property
    def approach_sequence(self) -> List[Optional[str]]:
        """Get sequence of approach classes across all fixes"""
        seq = []
        if self.starter_approach:
            seq.append(self.starter_approach)
        for fix in self.fix_steps:
            if fix.meta:
                seq.append(fix.meta.get('approach_class'))
        return seq
    
    def had_approach_switch(self) -> bool:
        """Check if approach ever switched from starter"""
        seq = [a for a in self.approach_sequence if a and a != 'unknown']
        if len(seq) < 2:
            return False
        starter = seq[0]
        return any(a != starter for a in seq[1:])
    
    def had_early_deviation(self) -> bool:
        """Check if first fix deviated from starter approach"""
        starter = self.starter_approach
        first = self.first_fix_approach
        if not starter or not first or starter == 'unknown' or first == 'unknown':
            return False
        return starter != first


def load_trajectories(
    run_dir: Path,
    tasks_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    recompute_approach: bool = False,
) -> List[Trajectory]:
    """Load all trajectories from a run directory"""
    steps_file = run_dir / "fact_step.jsonl"
    runs_file = run_dir / "fact_run.jsonl"
    
    if not steps_file.exists() or not runs_file.exists():
        raise FileNotFoundError(f"Missing fact files in {run_dir}")
    
    # Load runs
    runs = {}
    with open(runs_file) as f:
        for line in f:
            run = RunLog.from_dict(json.loads(line))
            runs[run.run_id] = run
    
    # Load steps and group by run_id
    steps_by_run = defaultdict(list)
    with open(steps_file) as f:
        for line in f:
            step = StepLog.from_dict(json.loads(line))
            steps_by_run[step.run_id].append(step)
    
    # Build trajectories
    trajectories = []
    for run_id, steps in steps_by_run.items():
        if run_id in runs:
            run = runs[run_id]

            # Optionally recompute approach_class for code steps
            for step in steps:
                if step.step_type not in ("code_starter", "code_fix"):
                    continue
                if not step.code:
                    continue
                meta = step.meta or {}
                existing = meta.get("approach_class")
                if recompute_approach or not existing or existing == "unknown":
                    topic = run.topic or (tasks_meta.get(run.task_id, {}).get("topic") if tasks_meta else None)
                    approach = detect_approach(
                        step.code,
                        task_id=run.task_id,
                        topic=topic,
                        tasks_meta=tasks_meta,
                    )
                    if approach:
                        meta = dict(meta)
                        meta["approach_class"] = approach
                        step.meta = meta

            # Sort steps by step_id
            steps.sort(key=lambda s: s.step_id)
            trajectories.append(Trajectory(run=run, steps=steps))
    
    return trajectories


# ============================================================================
# PRIORITY 1: MANUAL ANNOTATION
# ============================================================================

def sample_undetected_trajectories(
    trajectories: List[Trajectory],
    n_sample: int = 50,
    seed: int = 42
) -> List[Trajectory]:
    """
    Sample trajectories where no approach switch was detected.
    
    "Undetected" means:
    - Has starter code and at least one fix
    - approach_sequence shows no switches OR
    - approach_class is null/unknown throughout
    """
    random.seed(seed)
    
    undetected = []
    for t in trajectories:
        if not t.starter_code or not t.fix_steps:
            continue
        
        # Check if approach stayed constant or unknown
        seq = [a for a in t.approach_sequence if a and a != 'unknown']
        
        if len(seq) < 2:
            # All unknown/null
            undetected.append(t)
        elif len(set(seq)) == 1:
            # Single approach throughout
            undetected.append(t)
    
    print(f"Found {len(undetected)} undetected trajectories out of {len(trajectories)} total")
    print(f"Undetected rate: {len(undetected)/len(trajectories):.1%}")
    
    if len(undetected) <= n_sample:
        return undetected
    
    return random.sample(undetected, n_sample)


def format_code_comparison(starter: str, final: str) -> str:
    """Format side-by-side code comparison"""
    starter_lines = starter.split('\n')
    final_lines = final.split('\n')
    
    output = []
    output.append("=" * 80)
    output.append("STARTER CODE:")
    output.append("=" * 80)
    for i, line in enumerate(starter_lines, 1):
        output.append(f"{i:3d} | {line}")
    
    output.append("\n" + "=" * 80)
    output.append("FINAL FIX:")
    output.append("=" * 80)
    for i, line in enumerate(final_lines, 1):
        output.append(f"{i:3d} | {line}")
    
    return '\n'.join(output)


def interactive_annotation(trajectories: List[Trajectory], output_file: Path):
    """
    Interactive annotation interface.
    
    Shows starter vs final code, asks user to label:
    1. Did approach actually switch? (Y/N)
    2. If yes, what was the switch? (free text)
    3. Why did regex miss it? (multiple choice)
    """
    annotations = []
    
    print("\n" + "="*80)
    print("MANUAL ANNOTATION INTERFACE")
    print("="*80)
    print(f"Annotating {len(trajectories)} trajectories")
    print("Press Ctrl+C to save and exit early\n")
    
    try:
        for idx, t in enumerate(trajectories, 1):
            print(f"\n{'='*80}")
            print(f"TRAJECTORY {idx}/{len(trajectories)}")
            print(f"{'='*80}")
            print(f"Run ID: {t.run.run_id}")
            print(f"Task: {t.run.task_id}")
            print(f"Topic: {t.run.topic}")
            print(f"Success: {bool(t.run.success)}")
            print(f"Detected approach sequence: {t.approach_sequence}")
            
            if not t.starter_code or not t.fix_steps:
                print("Skipping: no starter or fixes")
                continue
            
            final_code = t.fix_steps[-1].code
            print(format_code_comparison(t.starter_code, final_code))
            
            print("\n" + "-"*80)
            
            # Question 1: Did approach switch?
            while True:
                response = input("Did the approach actually switch? (y/n/skip): ").strip().lower()
                if response in ['y', 'n', 'skip']:
                    break
                print("Invalid input. Please enter y, n, or skip")
            
            if response == 'skip':
                continue
            
            actually_switched = (response == 'y')
            
            switch_description = None
            miss_reason = None
            
            if actually_switched:
                # Question 2: Describe the switch
                print("\nDescribe the switch (e.g., 'list to heapq', 'iterative to recursive'):")
                switch_description = input("> ").strip()
                
                # Question 3: Why missed?
                print("\nWhy did detection miss this?")
                print("1. Custom data structure names")
                print("2. Mixed paradigms")
                print("3. Idiosyncratic naming")
                print("4. Pattern too specific")
                print("5. Other")
                while True:
                    miss_choice = input("Enter number (1-5): ").strip()
                    if miss_choice in ['1', '2', '3', '4', '5']:
                        break
                    print("Invalid choice")
                
                miss_reasons_map = {
                    '1': 'custom_data_structure',
                    '2': 'mixed_paradigms',
                    '3': 'idiosyncratic_naming',
                    '4': 'pattern_too_specific',
                    '5': 'other'
                }
                miss_reason = miss_reasons_map[miss_choice]
            
            # Save annotation
            annotation = {
                'run_id': t.run.run_id,
                'task_id': t.run.task_id,
                'topic': t.run.topic,
                'detected_sequence': t.approach_sequence,
                'actually_switched': actually_switched,
                'switch_description': switch_description,
                'miss_reason': miss_reason,
                'starter_code_preview': t.starter_code[:200],
                'final_code_preview': final_code[:200] if final_code else None
            }
            annotations.append(annotation)
            
            # Save after each annotation (in case of crash)
            with open(output_file, 'w') as f:
                json.dump(annotations, f, indent=2)
            
            print(f"[SAVED] ({len(annotations)} total)")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving progress...")
    
    # Final save
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"ANNOTATION COMPLETE")
    print(f"{'='*80}")
    print(f"Annotated: {len(annotations)} trajectories")
    print(f"Saved to: {output_file}")
    
    # Summary statistics
    if annotations:
        switched = sum(1 for a in annotations if a['actually_switched'])
        print(f"\nActual switch rate: {switched}/{len(annotations)} ({switched/len(annotations):.1%})")
        
        if switched > 0:
            print("\nMiss reasons:")
            reason_counts = defaultdict(int)
            for a in annotations:
                if a['miss_reason']:
                    reason_counts[a['miss_reason']] += 1
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                print(f"  {reason}: {count}")


# ============================================================================
# PRIORITY 2: STARTER DEVIATION ANALYSIS
# ============================================================================

def analyze_starter_deviation(trajectories: List[Trajectory]) -> Dict[str, Any]:
    """
    Compute how often first-fix approach != starter approach.
    
    This measures early reconsideration regardless of strict oscillation definition.
    """
    stats = {
        'total_trajectories': len(trajectories),
        'with_starter_and_fix': 0,
        'early_deviations': 0,
        'deviation_rate': 0.0,
        'by_topic': defaultdict(lambda: {'total': 0, 'deviations': 0}),
        'by_model': defaultdict(lambda: {'total': 0, 'deviations': 0}),
        'examples': []
    }
    
    for t in trajectories:
        if not t.starter_approach or not t.first_fix_approach:
            continue
        
        # Filter out unknowns
        if t.starter_approach == 'unknown' or t.first_fix_approach == 'unknown':
            continue
        
        stats['with_starter_and_fix'] += 1
        
        deviated = t.had_early_deviation()
        
        if deviated:
            stats['early_deviations'] += 1
            
            # Collect examples
            if len(stats['examples']) < 10:
                stats['examples'].append({
                    'run_id': t.run.run_id,
                    'task_id': t.run.task_id,
                    'topic': t.run.topic,
                    'starter': t.starter_approach,
                    'first_fix': t.first_fix_approach,
                    'full_sequence': t.approach_sequence
                })
        
        # By topic
        if t.run.topic:
            stats['by_topic'][t.run.topic]['total'] += 1
            if deviated:
                stats['by_topic'][t.run.topic]['deviations'] += 1
        
        # By model
        stats['by_model'][t.run.model_id]['total'] += 1
        if deviated:
            stats['by_model'][t.run.model_id]['deviations'] += 1
    
    if stats['with_starter_and_fix'] > 0:
        stats['deviation_rate'] = stats['early_deviations'] / stats['with_starter_and_fix']
    
    return stats


def print_deviation_analysis(stats: Dict[str, Any]):
    """Print starter deviation analysis results"""
    print("\n" + "="*80)
    print("STARTER DEVIATION ANALYSIS")
    print("="*80)
    print(f"Trajectories analyzed: {stats['with_starter_and_fix']}")
    print(f"Early deviations: {stats['early_deviations']}")
    print(f"Deviation rate: {stats['deviation_rate']:.1%}")
    
    print("\n--- By Topic ---")
    for topic, data in sorted(stats['by_topic'].items()):
        rate = data['deviations'] / data['total'] if data['total'] > 0 else 0
        print(f"  {topic:20s}: {data['deviations']:3d}/{data['total']:3d} ({rate:.1%})")
    
    print("\n--- By Model ---")
    for model, data in sorted(stats['by_model'].items()):
        rate = data['deviations'] / data['total'] if data['total'] > 0 else 0
        print(f"  {model:40s}: {data['deviations']:3d}/{data['total']:3d} ({rate:.1%})")
    
    if stats['examples']:
        print("\n--- Example Deviations ---")
        for ex in stats['examples'][:5]:
            print(f"\n  {ex['task_id']} ({ex['topic']}):")
            print(f"    Starter: {ex['starter']}")
            print(f"    First fix: {ex['first_fix']}")
            print(f"    Full sequence: {ex['full_sequence']}")


# ============================================================================
# PRIORITY 3: SURVIVAL ANALYSIS
# ============================================================================

@dataclass
class SurvivalData:
    """Per-trajectory record for survival analysis."""
    run_id: str
    time: int   # Attempt index of fix or censoring (1..max_attempts)
    event: bool # True if fixed at `time`, False if censored at `time`


def extract_survival_data(trajectories: List[Trajectory], max_attempts: int = 4) -> List[SurvivalData]:
    """
    Extract survival data from trajectories.

    Each trajectory contributes ONE record:
    - event=True at the first successful attempt (if within max_attempts)
    - event=False (censored) at the last observed attempt otherwise
    """
    survival_data: List[SurvivalData] = []

    for t in trajectories:
        test_steps = [s for s in t.steps if s.step_type == 'test']
        if not test_steps:
            continue

        # Cap observation window at max_attempts
        observed = min(len(test_steps), max_attempts)

        event_time: Optional[int] = None
        for attempt_idx, test in enumerate(test_steps[:max_attempts], 1):
            if test.exit_code == 0:
                event_time = attempt_idx
                break

        if event_time is not None:
            survival_data.append(SurvivalData(
                run_id=t.run.run_id,
                time=event_time,
                event=True
            ))
        else:
            survival_data.append(SurvivalData(
                run_id=t.run.run_id,
                time=observed,
                event=False
            ))

    return survival_data


def compute_kaplan_meier(survival_data: List[SurvivalData], max_attempts: int = 4) -> Dict[int, Dict[str, float]]:
    """
    Compute Kaplan-Meier survival curve and hazard rates.

    Returns dict mapping attempt -> {survival_prob, hazard_rate, at_risk, events}
    """
    results: Dict[int, Dict[str, float]] = {}
    survival_prob = 1.0

    for attempt in range(1, max_attempts + 1):
        # At risk = trajectories with time >= attempt
        n_at_risk = sum(1 for d in survival_data if d.time >= attempt)
        # Events = trajectories that fix exactly at this attempt
        n_events = sum(1 for d in survival_data if d.event and d.time == attempt)

        if n_at_risk == 0:
            hazard_rate = 0.0
            survival_prob_new = survival_prob
        else:
            hazard_rate = n_events / n_at_risk
            survival_prob_new = survival_prob * (1 - hazard_rate)

        results[attempt] = {
            'survival_prob': survival_prob_new,
            'cumulative_fix_prob': 1 - survival_prob_new,
            'hazard_rate': hazard_rate,
            'at_risk': n_at_risk,
            'events': n_events
        }

        survival_prob = survival_prob_new

    return results


def print_survival_analysis(km_results: Dict[int, Dict[str, float]]):
    """Print survival analysis results"""
    print("\n" + "="*80)
    print("SURVIVAL ANALYSIS (Kaplan-Meier)")
    print("="*80)
    print("\nP(fixed by attempt k) = 1 - S(k)")
    print("Hazard rate h(k) = P(fix at k | survived to k)")
    print()
    print(f"{'Attempt':<10} {'At Risk':<10} {'Fixes':<10} {'Hazard':<12} {'Cum Fix %':<12} {'Survival %':<12}")
    print("-" * 80)
    
    for attempt in sorted(km_results.keys()):
        r = km_results[attempt]
        print(f"{attempt:<10} {r['at_risk']:<10} {r['events']:<10} "
              f"{r['hazard_rate']:<12.1%} {r['cumulative_fix_prob']:<12.1%} "
              f"{r['survival_prob']:<12.1%}")
    
    print("\nKey insights:")
    if 1 in km_results:
        print(f"- First-attempt fix rate: {km_results[1]['hazard_rate']:.1%}")
    
    max_attempt = max(km_results.keys())
    print(f"- Overall fix rate by attempt {max_attempt}: {km_results[max_attempt]['cumulative_fix_prob']:.1%}")
    
    # Check if hazard is increasing or decreasing
    hazards = [km_results[k]['hazard_rate'] for k in sorted(km_results.keys())]
    if len(hazards) >= 2:
        if hazards[-1] > hazards[0]:
            print(f"- Hazard rate INCREASING ({hazards[0]:.1%} -> {hazards[-1]:.1%}): models improve with feedback")
        else:
            print(f"- Hazard rate DECREASING ({hazards[0]:.1%} -> {hazards[-1]:.1%}): early fixes more likely")


# ============================================================================
# PRIORITY 4: ENHANCED METRICS
# ============================================================================

def compute_enhanced_metrics(trajectories: List[Trajectory]) -> Dict[str, Any]:
    """
    Compute all enhanced metrics in one pass.
    
    Combines:
    - Approach switch detection
    - Starter deviation
    - Per-attempt fix rates
    """
    metrics = {
        'total': len(trajectories),
        'approach_switches': 0,
        'approach_switch_rate': 0.0,
        'early_deviations': 0,
        'early_deviation_rate': 0.0,
        'fix_by_attempt': defaultdict(int),
        'attempts_tried': defaultdict(int),
        'approach_coverage': {
            'detected': 0,
            'total': 0,
            'rate': 0.0
        }
    }
    
    trajectories_with_approach_meta = 0
    trajectories_with_known_switch_scope = 0
    trajectories_with_known_early_scope = 0
    
    for t in trajectories:
        # Approach coverage
        seq = t.approach_sequence
        if seq:
            trajectories_with_approach_meta += 1
            detected_approaches = [a for a in seq if a and a != 'unknown']
            if detected_approaches:
                metrics['approach_coverage']['detected'] += 1

        # Approach switch detection (only when switch is observable)
        known_seq = [a for a in seq if a and a != 'unknown']
        if len(known_seq) >= 2:
            trajectories_with_known_switch_scope += 1
            if t.had_approach_switch():
                metrics['approach_switches'] += 1

        # Early deviation (starter + first fix must be known)
        starter = t.starter_approach
        first_fix = t.first_fix_approach
        if starter and first_fix and starter != 'unknown' and first_fix != 'unknown':
            trajectories_with_known_early_scope += 1
            if t.had_early_deviation():
                metrics['early_deviations'] += 1
        
        # Fix rates by attempt
        test_steps = [s for s in t.steps if s.step_type == 'test']
        for attempt, test in enumerate(test_steps, 1):
            metrics['attempts_tried'][attempt] += 1
            if test.exit_code == 0:
                metrics['fix_by_attempt'][attempt] += 1
                break  # Count only first successful fix
    
    if trajectories_with_known_switch_scope > 0:
        metrics['approach_switch_rate'] = metrics['approach_switches'] / trajectories_with_known_switch_scope
    if trajectories_with_known_early_scope > 0:
        metrics['early_deviation_rate'] = metrics['early_deviations'] / trajectories_with_known_early_scope

    metrics['approach_coverage']['total'] = trajectories_with_approach_meta
    if trajectories_with_approach_meta > 0:
        metrics['approach_coverage']['rate'] = metrics['approach_coverage']['detected'] / trajectories_with_approach_meta
    
    return metrics


def print_enhanced_metrics(metrics: Dict[str, Any]):
    """Print enhanced metrics"""
    print("\n" + "="*80)
    print("ENHANCED METRICS SUMMARY")
    print("="*80)
    
    print(f"\nTotal trajectories: {metrics['total']}")
    
    print("\n--- Approach Switching ---")
    print(f"Approach switches detected: {metrics['approach_switches']}")
    print(f"Switch rate: {metrics['approach_switch_rate']:.1%}")
    
    print("\n--- Early Deviation (First Fix ≠ Starter) ---")
    print(f"Early deviations: {metrics['early_deviations']}")
    print(f"Deviation rate: {metrics['early_deviation_rate']:.1%}")
    
    print("\n--- Fix Rates by Attempt ---")
    print(f"{'Attempt':<10} {'Fixes':<10} {'Tried':<10} {'Fix %':<10}")
    print("-" * 40)
    for attempt in sorted(metrics['attempts_tried'].keys()):
        fixes = metrics['fix_by_attempt'][attempt]
        tried = metrics['attempts_tried'][attempt]
        rate = fixes / tried if tried > 0 else 0
        print(f"{attempt:<10} {fixes:<10} {tried:<10} {rate:<10.1%}")
    
    print("\n--- Approach Coverage ---")
    print(f"Trajectories with detected approaches: {metrics['approach_coverage']['detected']}/{metrics['approach_coverage']['total']}")
    print(f"Coverage rate: {metrics['approach_coverage']['rate']:.1%}")


# ============================================================================
# MAIN CLI
# ============================================================================

@dataclass
class AnnotateArgs:
    """Arguments for manual annotation"""
    run_dir: str
    n_sample: int = 50
    output: str = "annotations.json"
    seed: int = 42
    tasks_path: str = ""
    recompute_approach: bool = True


@dataclass
class DeviationArgs:
    """Arguments for starter deviation analysis"""
    run_dir: str
    output: str = "deviation_analysis.json"
    tasks_path: str = ""
    recompute_approach: bool = True


@dataclass
class SurvivalArgs:
    """Arguments for survival analysis"""
    run_dir: str
    max_attempts: int = 4
    output: str = "survival_analysis.json"
    tasks_path: str = ""
    recompute_approach: bool = True


@dataclass
class AllArgs:
    """Arguments for running all analyses"""
    run_dir: str
    n_sample: int = 50
    max_attempts: int = 4
    output_dir: str = "extended_analysis"
    tasks_path: str = ""
    recompute_approach: bool = True


def cmd_annotate(args: AnnotateArgs):
    """Run manual annotation"""
    run_dir = Path(args.run_dir)
    tasks_meta = load_tasks_metadata(Path(args.tasks_path)) if args.tasks_path else None
    trajectories = load_trajectories(
        run_dir,
        tasks_meta=tasks_meta,
        recompute_approach=args.recompute_approach,
    )
    
    sampled = sample_undetected_trajectories(trajectories, args.n_sample, args.seed)
    
    output_file = Path(args.output)
    interactive_annotation(sampled, output_file)


def cmd_deviation(args: DeviationArgs):
    """Run starter deviation analysis"""
    run_dir = Path(args.run_dir)
    tasks_meta = load_tasks_metadata(Path(args.tasks_path)) if args.tasks_path else None
    trajectories = load_trajectories(
        run_dir,
        tasks_meta=tasks_meta,
        recompute_approach=args.recompute_approach,
    )
    
    stats = analyze_starter_deviation(trajectories)
    print_deviation_analysis(stats)
    
    # Save to file
    output_file = Path(args.output)
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\nSaved to: {output_file}")


def cmd_survival(args: SurvivalArgs):
    """Run survival analysis"""
    run_dir = Path(args.run_dir)
    tasks_meta = load_tasks_metadata(Path(args.tasks_path)) if args.tasks_path else None
    trajectories = load_trajectories(
        run_dir,
        tasks_meta=tasks_meta,
        recompute_approach=args.recompute_approach,
    )
    
    survival_data = extract_survival_data(trajectories, args.max_attempts)
    km_results = compute_kaplan_meier(survival_data, args.max_attempts)
    print_survival_analysis(km_results)
    
    # Save to file
    output_file = Path(args.output)
    with open(output_file, 'w') as f:
        json.dump(km_results, f, indent=2)
    print(f"\nSaved to: {output_file}")


def cmd_all(args: AllArgs):
    """Run all analyses"""
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Loading trajectories...")
    tasks_meta = load_tasks_metadata(Path(args.tasks_path)) if args.tasks_path else None
    trajectories = load_trajectories(
        run_dir,
        tasks_meta=tasks_meta,
        recompute_approach=args.recompute_approach,
    )
    print(f"Loaded {len(trajectories)} trajectories\n")
    
    # 1. Enhanced metrics
    print("\n" + "="*80)
    print("1/4: COMPUTING ENHANCED METRICS")
    print("="*80)
    metrics = compute_enhanced_metrics(trajectories)
    print_enhanced_metrics(metrics)
    with open(output_dir / "enhanced_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    # 2. Starter deviation
    print("\n" + "="*80)
    print("2/4: STARTER DEVIATION ANALYSIS")
    print("="*80)
    deviation_stats = analyze_starter_deviation(trajectories)
    print_deviation_analysis(deviation_stats)
    with open(output_dir / "deviation_analysis.json", 'w') as f:
        json.dump(deviation_stats, f, indent=2, default=str)
    
    # 3. Survival analysis
    print("\n" + "="*80)
    print("3/4: SURVIVAL ANALYSIS")
    print("="*80)
    survival_data = extract_survival_data(trajectories, args.max_attempts)
    km_results = compute_kaplan_meier(survival_data, args.max_attempts)
    print_survival_analysis(km_results)
    with open(output_dir / "survival_analysis.json", 'w') as f:
        json.dump(km_results, f, indent=2)
    
    # 4. Sample for manual annotation
    print("\n" + "="*80)
    print("4/4: SAMPLING FOR MANUAL ANNOTATION")
    print("="*80)
    sampled = sample_undetected_trajectories(trajectories, args.n_sample)
    print(f"\nSampled {len(sampled)} trajectories for annotation")
    print(f"To annotate, run:")
    print(f"  python extended_analysis.py annotate --run_dir {args.run_dir}")
    
    # Save sample metadata
    sample_metadata = [
        {
            'run_id': t.run.run_id,
            'task_id': t.run.task_id,
            'topic': t.run.topic,
            'approach_sequence': t.approach_sequence
        }
        for t in sampled
    ]
    with open(output_dir / "annotation_sample.json", 'w') as f:
        json.dump(sample_metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print("ALL ANALYSES COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    tyro.cli({
        "annotate": cmd_annotate,
        "deviation": cmd_deviation,
        "survival": cmd_survival,
        "all": cmd_all,
    })
