"""
Multi-Model Analysis

Analyzes multiple model run directories at once and generates comparative reports.

Usage:
    # Analyze all models matching a pattern
    python multi_model_analysis.py --run_dirs "runs/osc_*"
    
    # Or specify directories explicitly
    python multi_model_analysis.py --run_dirs runs/osc_qwen runs/osc_llama runs/osc_deepseek
    
    # With glob pattern
    python multi_model_analysis.py --run_pattern "runs/osc_*" --output multi_model_results
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any
import glob
import hashlib
import json
import statistics
import tyro

from extended_analysis import (
    load_trajectories,
    analyze_starter_deviation,
    compute_enhanced_metrics,
    extract_survival_data,
    compute_kaplan_meier,
    Trajectory
)
from approach_detection import load_tasks_metadata


def extract_model_name(run_dir: Path) -> str:
    """Extract model name from directory path"""
    # Assumes format: runs/{tag}_{model_name} or similar
    dir_name = run_dir.name
    
    # Try to extract after last underscore
    parts = dir_name.split('_')
    if len(parts) >= 2:
        # Join everything after the tag
        # E.g., "osc_qwen2_5_coder" -> "qwen2_5_coder"
        return '_'.join(parts[1:])
    
    return dir_name


def compute_data_fingerprint(trajectories: List[Trajectory]) -> str:
    """Compute a short fingerprint of run/task identities to detect duplicates."""
    h = hashlib.sha1()
    rows = sorted(
        (t.run.run_id, t.run.task_id, t.run.model_id, t.run.variant_id, t.run.success)
        for t in trajectories
    )
    for row in rows:
        h.update("|".join(map(str, row)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:12]


@dataclass
class ModelResults:
    """Results for a single model"""
    model_name: str
    run_dir: Path
    n_trajectories: int
    enhanced_metrics: Dict[str, Any]
    deviation_stats: Dict[str, Any]
    survival_results: Dict[int, Dict[str, float]]
    data_fingerprint: str
    model_ids: List[str]
    task_count: int


def analyze_single_model(
    run_dir: Path,
    model_name: str = None,
    *,
    tasks_meta: Dict[str, Dict[str, Any]] | None = None,
    recompute_approach: bool = False,
) -> ModelResults:
    """Run complete analysis for a single model"""
    if model_name is None:
        model_name = extract_model_name(run_dir)
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {model_name}")
    print(f"Directory: {run_dir}")
    print(f"{'='*70}")
    
    # Load trajectories
    trajectories = load_trajectories(
        run_dir,
        tasks_meta=tasks_meta,
        recompute_approach=recompute_approach,
    )
    print(f"Loaded {len(trajectories)} trajectories")
    
    if not trajectories:
        print(f"[!] Warning: No trajectories found in {run_dir}")
        return None

    model_ids = sorted({t.run.model_id for t in trajectories if t.run.model_id})
    task_ids = sorted({t.run.task_id for t in trajectories if t.run.task_id})
    data_fingerprint = compute_data_fingerprint(trajectories)

    if len(model_ids) > 1:
        print(f"[!] Warning: Multiple model_ids in {run_dir}: {model_ids}")
    print(f"Data fingerprint: {data_fingerprint}")
    
    # Run analyses
    enhanced_metrics = compute_enhanced_metrics(trajectories)
    deviation_stats = analyze_starter_deviation(trajectories)
    
    survival_data = extract_survival_data(trajectories, max_attempts=4)
    survival_results = compute_kaplan_meier(survival_data, max_attempts=4)
    
    return ModelResults(
        model_name=model_name,
        run_dir=run_dir,
        n_trajectories=len(trajectories),
        enhanced_metrics=enhanced_metrics,
        deviation_stats=deviation_stats,
        survival_results=survival_results,
        data_fingerprint=data_fingerprint,
        model_ids=model_ids,
        task_count=len(task_ids),
    )


def print_comparison_table(results: List[ModelResults]):
    """Print comparison table across all models"""
    print("\n" + "="*100)
    print("MULTI-MODEL COMPARISON")
    print("="*100)
    
    # Header
    print(f"\n{'Model':<30} {'N':<8} {'Success':<10} {'Switch':<10} {'Early Dev':<10} {'Hazard':<10}")
    print("-"*100)
    
    # Sort by success rate
    results_sorted = sorted(
        results,
        key=lambda r: r.enhanced_metrics['fix_by_attempt'].get(1, 0) / r.n_trajectories if r.n_trajectories > 0 else 0,
        reverse=True
    )
    
    for r in results_sorted:
        model = r.model_name[:28]
        n = r.n_trajectories
        
        # Success rate (at least one fix)
        total_fixes = sum(r.enhanced_metrics['fix_by_attempt'].values())
        success_rate = total_fixes / n if n > 0 else 0
        
        # Switch rate
        switch_rate = r.enhanced_metrics['approach_switch_rate']
        
        # Early deviation rate
        dev_rate = r.deviation_stats['deviation_rate']
        
        # Hazard trend (first -> last)
        attempts = sorted(r.survival_results.keys())
        if len(attempts) >= 2:
            h_first = r.survival_results[attempts[0]]['hazard_rate']
            h_last = r.survival_results[attempts[-1]]['hazard_rate']
            hazard_trend = f"{h_first:.1%}->{h_last:.1%}"
        else:
            hazard_trend = "N/A"
        
        print(f"{model:<30} {n:<8} {success_rate:<10.1%} {switch_rate:<10.1%} {dev_rate:<10.1%} {hazard_trend:<10}")
    
    print("\n" + "="*100)
    print("INTERPRETATION GUIDE")
    print("="*100)
    print("\nSuccess:   Overall fix rate across all attempts")
    print("Switch:    Strict oscillation (approach != starter)")
    print("Early Dev: First fix != starter (looser reconsideration)")
    print("Hazard:   First -> Last attempt hazard rate")
    print("           - Decreasing = entrenchment (models get worse)")
    print("           - Increasing = learning (models get better)")


def print_detailed_breakdowns(results: List[ModelResults]):
    """Print detailed breakdowns for each model"""
    
    for r in results:
        print(f"\n{'='*100}")
        print(f"DETAILED BREAKDOWN: {r.model_name}")
        print(f"{'='*100}")
        
        # Fix rates by attempt
        print(f"\nFix Rates by Attempt (n={r.n_trajectories}):")
        print(f"{'Attempt':<12} {'Fixes':<10} {'Rate':<10} {'Cum. Rate':<12}")
        print("-"*50)
        
        attempts_tried = r.enhanced_metrics['attempts_tried']
        fix_by_attempt = r.enhanced_metrics['fix_by_attempt']
        cumulative_fixes = 0
        
        for attempt in sorted(attempts_tried.keys()):
            fixes = fix_by_attempt.get(attempt, 0)
            tried = attempts_tried[attempt]
            rate = fixes / tried if tried > 0 else 0
            cumulative_fixes += fixes
            cum_rate = cumulative_fixes / r.n_trajectories
            
            print(f"{attempt:<12} {fixes:<10} {rate:<10.1%} {cum_rate:<12.1%}")
        
        # Survival analysis
        print(f"\nSurvival Analysis:")
        print(f"{'Attempt':<12} {'At Risk':<10} {'Events':<10} {'Hazard':<10} {'Survival':<12}")
        print("-"*60)
        
        for attempt in sorted(r.survival_results.keys()):
            sr = r.survival_results[attempt]
            print(f"{attempt:<12} {sr['at_risk']:<10} {sr['events']:<10} "
                  f"{sr['hazard_rate']:<10.1%} {sr['survival_prob']:<12.1%}")
        
        # Approach switching details
        print(f"\nApproach Analysis:")
        print(f"  Total switches:    {r.enhanced_metrics['approach_switches']} ({r.enhanced_metrics['approach_switch_rate']:.1%})")
        print(f"  Early deviations:  {r.deviation_stats['early_deviations']} ({r.deviation_stats['deviation_rate']:.1%})")
        print(f"  Coverage:          {r.enhanced_metrics['approach_coverage']['rate']:.1%}")
        
        # By topic breakdown
        if r.deviation_stats['by_topic']:
            print(f"\nEarly Deviation by Topic:")
            by_topic = sorted(
                r.deviation_stats['by_topic'].items(),
                key=lambda x: x[1]['deviations'] / x[1]['total'] if x[1]['total'] > 0 else 0,
                reverse=True
            )
            for topic, data in by_topic[:10]:  # Top 10
                if data['total'] > 0:
                    rate = data['deviations'] / data['total']
                    print(f"  {topic:20s}: {data['deviations']:3}/{data['total']:3} ({rate:.1%})")


def generate_json_export(results: List[ModelResults], output_file: Path):
    """Export all results to JSON"""
    export = {
        'models': {},
        'summary': {
            'n_models': len(results),
            'total_trajectories': sum(r.n_trajectories for r in results)
        }
    }
    
    for r in results:
        export['models'][r.model_name] = {
            'n_trajectories': r.n_trajectories,
            'run_dir': str(r.run_dir),
            'data_fingerprint': r.data_fingerprint,
            'model_ids': r.model_ids,
            'task_count': r.task_count,
            'enhanced_metrics': r.enhanced_metrics,
            'deviation_stats': {
                'total': r.deviation_stats['with_starter_and_fix'],
                'deviations': r.deviation_stats['early_deviations'],
                'rate': r.deviation_stats['deviation_rate'],
                'by_topic': dict(r.deviation_stats['by_topic'])
            },
            'survival': {str(k): v for k, v in r.survival_results.items()}
        }
    
    with open(output_file, 'w') as f:
        json.dump(export, f, indent=2, default=str)
    
    print(f"\nExported results to: {output_file}")


def find_key_insights(results: List[ModelResults]) -> List[str]:
    """Automatically identify key insights across models"""
    insights = []
    
    # Check for consistent patterns
    all_switch_rates = [r.enhanced_metrics['approach_switch_rate'] for r in results]
    all_dev_rates = [r.deviation_stats['deviation_rate'] for r in results]
    
    avg_switch = sum(all_switch_rates) / len(all_switch_rates) if all_switch_rates else 0
    avg_dev = sum(all_dev_rates) / len(all_dev_rates) if all_dev_rates else 0
    
    # Insight 1: Low switching
    if avg_switch < 0.02:
        insights.append(
            f"[!] Very low average switch rate ({avg_switch:.1%}) across all models - "
            "manual annotation critical to validate detection"
        )
    
    # Insight 2: High deviation despite low switching
    if avg_dev > avg_switch * 5:
        insights.append(
            f"[+] Early deviation ({avg_dev:.1%}) >> strict switching ({avg_switch:.1%}) - "
            "models DO reconsider but commit to alternatives"
        )
    
    # Insight 3: Check for decreasing hazards
    decreasing_hazards = 0
    for r in results:
        attempts = sorted(r.survival_results.keys())
        if len(attempts) >= 2:
            h_first = r.survival_results[attempts[0]]['hazard_rate']
            h_last = r.survival_results[attempts[-1]]['hazard_rate']
            if h_last < h_first * 0.8:  # 20% decrease
                decreasing_hazards += 1
    
    if decreasing_hazards >= len(results) * 0.5:  # Majority of models
        insights.append(
            f"[+] {decreasing_hazards}/{len(results)} models show decreasing hazard rates - "
            "evidence of entrenchment over time"
        )
    
    # Insight 4: Model variation
    if len(all_switch_rates) >= 2:
        switch_range = max(all_switch_rates) - min(all_switch_rates)
        if switch_range > 0.05:  # >5% variation
            best_model = results[all_switch_rates.index(max(all_switch_rates))].model_name
            insights.append(
                f"[+] Large variation in switch rates ({min(all_switch_rates):.1%} - {max(all_switch_rates):.1%}) - "
                f"{best_model} most willing to reconsider"
            )
    
    return insights


@dataclass
class Args:
    run_dirs: tuple[str, ...] = ()
    run_pattern: str = ""  # Glob pattern like "runs/osc_*"
    output: str = "multi_model_results"
    detailed: bool = True  # Show detailed breakdowns
    tasks_path: str = ""
    recompute_approach: bool = True
    validate: bool = False
    expected_n: int = 0
    min_variance: float = 0.01
    min_approach_coverage: float = 0.0


def main(args: Args):
    """Run multi-model analysis"""
    
    # Collect directories
    directories = []
    
    if args.run_pattern:
        # Use glob pattern
        matched = glob.glob(args.run_pattern)
        directories.extend([Path(p) for p in matched if Path(p).is_dir()])
        print(f"Found {len(directories)} directories matching pattern: {args.run_pattern}")
    
    if args.run_dirs:
        # Add explicitly specified directories
        directories.extend([Path(d) for d in args.run_dirs])
    
    if not directories:
        print("Error: No directories specified. Use --run_dirs or --run_pattern")
        return
    
    # Remove duplicates
    directories = list(set(directories))
    
    print(f"\n{'='*100}")
    print(f"MULTI-MODEL ANALYSIS")
    print(f"{'='*100}")
    print(f"Analyzing {len(directories)} model runs:")
    for d in directories:
        print(f"  - {d}")
    
    # Load task metadata (optional, improves approach detection)
    tasks_meta = None
    if args.tasks_path:
        tasks_meta = load_tasks_metadata(Path(args.tasks_path))
    else:
        default_tasks = Path("data/tasks/v8_osc_total_200.jsonl")
        if default_tasks.exists():
            tasks_meta = load_tasks_metadata(default_tasks)

    # Analyze each model
    results = []
    for run_dir in directories:
        try:
            result = analyze_single_model(
                run_dir,
                tasks_meta=tasks_meta,
                recompute_approach=args.recompute_approach,
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"[!] Error analyzing {run_dir}: {e}")
            continue
    
    if not results:
        print("Error: No valid results obtained")
        return
    
    print(f"\n\nSuccessfully analyzed {len(results)}/{len(directories)} models")

    # Data integrity checks
    if results:
        fingerprints: Dict[str, str] = {}
        for r in results:
            if r.data_fingerprint in fingerprints:
                msg = (
                    f"Duplicate data fingerprint: {r.model_name} and {fingerprints[r.data_fingerprint]} "
                    f"share {r.data_fingerprint}"
                )
                if args.validate:
                    raise AssertionError(msg)
                print(f"[!] Warning: {msg}")
            else:
                fingerprints[r.data_fingerprint] = r.model_name

        if args.expected_n > 0:
            for r in results:
                if r.n_trajectories != args.expected_n:
                    msg = f"{r.model_name} has {r.n_trajectories} trajectories (expected {args.expected_n})"
                    if args.validate:
                        raise AssertionError(msg)
                    print(f"[!] Warning: {msg}")

        # Variance checks across models
        if len(results) >= 2 and args.min_variance > 0:
            switch_rates = [r.enhanced_metrics['approach_switch_rate'] for r in results]
            dev_rates = [r.deviation_stats['deviation_rate'] for r in results]
            if statistics.pstdev(switch_rates) <= args.min_variance:
                msg = f"Low variance in switch rates (std={statistics.pstdev(switch_rates):.3f})"
                if args.validate:
                    raise AssertionError(msg)
                print(f"[!] Warning: {msg}")
            if statistics.pstdev(dev_rates) <= args.min_variance:
                msg = f"Low variance in early deviation rates (std={statistics.pstdev(dev_rates):.3f})"
                if args.validate:
                    raise AssertionError(msg)
                print(f"[!] Warning: {msg}")

        # Coverage checks
        if args.min_approach_coverage > 0:
            for r in results:
                coverage = r.enhanced_metrics['approach_coverage']['rate']
                if coverage < args.min_approach_coverage:
                    msg = f"{r.model_name} approach coverage {coverage:.1%} < {args.min_approach_coverage:.1%}"
                    if args.validate:
                        raise AssertionError(msg)
                    print(f"[!] Warning: {msg}")
    
    # Generate outputs
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Comparison table
    print_comparison_table(results)
    
    # Detailed breakdowns
    if args.detailed:
        print_detailed_breakdowns(results)
    
    # Key insights
    insights = find_key_insights(results)
    if insights:
        print("\n" + "="*100)
        print("KEY INSIGHTS")
        print("="*100)
        for i, insight in enumerate(insights, 1):
            print(f"\n{i}. {insight}")
    
    # Export JSON
    json_file = output_dir / "multi_model_results.json"
    generate_json_export(results, json_file)
    
    # Summary file
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("MULTI-MODEL ANALYSIS SUMMARY\n")
        f.write("="*100 + "\n\n")
        f.write(f"Models analyzed: {len(results)}\n")
        f.write(f"Total trajectories: {sum(r.n_trajectories for r in results)}\n\n")
        
        for r in results:
            f.write(f"\n{r.model_name}:\n")
            f.write(f"  Trajectories: {r.n_trajectories}\n")
            f.write(f"  Switch rate: {r.enhanced_metrics['approach_switch_rate']:.1%}\n")
            f.write(f"  Early dev rate: {r.deviation_stats['deviation_rate']:.1%}\n")
        
        if insights:
            f.write("\n\nKey Insights:\n")
            for i, insight in enumerate(insights, 1):
                f.write(f"{i}. {insight}\n")
    
    print(f"\nSummary written to: {summary_file}")
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - multi_model_results.json (full data)")
    print(f"  - summary.txt (human-readable summary)")


if __name__ == "__main__":
    main(tyro.cli(Args))
