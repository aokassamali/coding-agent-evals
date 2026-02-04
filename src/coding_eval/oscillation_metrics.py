# decision_analysis.py
r"""
Analyzes decision trajectories from agent evaluation runs.
Usage: python decision_analysis.py --run_dirs r"C:\...\runs\w_qwen2_5_coder_7b_instruct_q6_k" r"C:\...\runs\w_qwen2_5_coder_7b_instruct_q4_k_m"
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
from approach_detection import detect_approach as shared_detect_approach, load_tasks_metadata


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

def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator
    spread = z * ((p*(1-p) + z**2/(4*n)) / n) ** 0.5 / denominator
    return (max(0.0, center - spread), min(1.0, center + spread))


# Insert after line 53 (after ERROR_SEVERITY dict)

# ----------------------------
# Approach Detection Patterns
# ----------------------------

APPROACH_PATTERNS = {
    # === Data Structures ===
    'queue_impl': [
        ('collections_deque', lambda c: 'deque' in c and ('popleft' in c or 'appendleft' in c)),
        ('list_with_indices', lambda c: ('self.front' in c or 'self.rear' in c) and 'deque' not in c),
        ('circular_buffer', lambda c: '% self.capacity' in c or '% len(' in c or '% self.size' in c),
        ('list_only', lambda c: 'pop(0)' in c and 'deque' not in c),
    ],
    'stack_impl': [
        ('list_append_pop', lambda c: '.append(' in c and '.pop()' in c and 'self.top' not in c and 'self.head' not in c),
        ('index_tracking', lambda c: 'top_idx' in c or 'self.top' in c),
        ('linked_list', lambda c: ('self.head' in c or "['next']" in c or '.next' in c) and 'list' not in c.lower()),
    ],
    'linked_list': [
        ('dict_nodes', lambda c: "['next']" in c or "['prev']" in c),
        ('class_nodes', lambda c: '.next' in c and "['next']" not in c),
        ('sentinel_nodes', lambda c: 'dummy' in c or 'sentinel' in c),
        ('iterative', lambda c: 'while' in c and ('curr' in c or 'current' in c)),
        ('recursive', lambda c: 'return' in c and 'def ' in c and 'while' not in c),
    ],
    'bst': [
        ('dict_nodes', lambda c: "['left']" in c or "['right']" in c),
        ('class_nodes', lambda c: '.left' in c or '.right' in c),
        ('iterative', lambda c: 'while' in c and ('.left' in c or '.right' in c or "['left']" in c)),
        ('recursive', lambda c: 'return' in c and ('self.insert' in c or 'self.delete' in c or 'self.search' in c)),
    ],
    'heap': [
        ('heapq', lambda c: 'heapq' in c or 'heappush' in c or 'heappop' in c),
        ('manual_sift', lambda c: 'sift' in c or '_sift' in c or 'bubble' in c),
        ('sorted_list', lambda c: '.sort()' in c or 'bisect' in c),
    ],
    'priority_queue': [
        ('heapq_module', lambda c: 'heapq' in c or 'heappush' in c),
        ('sorted_list', lambda c: '.sort()' in c or 'sorted(' in c),
        ('manual_heap', lambda c: 'sift' in c or '_sift' in c),
    ],
    'hashtable': [
        ('open_addressing', lambda c: ('% self.capacity' in c or '% self.size' in c or '% len(' in c) and 'bucket' not in c),
        ('chaining', lambda c: 'bucket' in c or ('append' in c and 'self.table' in c)),
        ('tombstone', lambda c: 'tombstone' in c or 'DELETED' in c),
        ('python_dict', lambda c: 'self.data = {}' in c or 'self._dict' in c),
    ],
    'trie': [
        ('dict_children', lambda c: 'self.children = {}' in c or 'children = {}' in c or 'dict()' in c),
        ('array_children', lambda c: '[None] * 26' in c or '[None]*26' in c or 'children = [' in c),
        ('defaultdict', lambda c: 'defaultdict' in c),
        ('nested_dict', lambda c: 'defaultdict(dict)' in c or 'lambda: {}' in c),
    ],
    'disjoint_set': [
        ('path_compression', lambda c: 'self.parent[x] = ' in c and 'find' in c),
        ('union_by_rank', lambda c: 'rank' in c or 'size' in c),
        ('simple_union', lambda c: 'self.parent[' in c and 'rank' not in c and 'size' not in c),
        ('iterative_find', lambda c: 'while' in c and 'parent' in c),
        ('recursive_find', lambda c: 'return self.find' in c or 'return find' in c),
    ],
    'ring_buffer': [
        ('deque', lambda c: 'deque' in c),
        ('modulo_index', lambda c: '% self.capacity' in c or '% self.size' in c or '% len(' in c),
        ('list_rotate', lambda c: '.rotate(' in c or 'self.head' in c),
    ],
    'skip_list': [
        ('linked_levels', lambda c: 'self.level' in c and '.next' in c),
        ('array_levels', lambda c: 'self.forward' in c or 'forward[' in c),
        ('sorted_list_baseline', lambda c: 'bisect' in c or '.sort()' in c),
    ],
    'graph': [
        ('adjacency_list', lambda c: 'self.adj' in c and 'matrix' not in c),
        ('adjacency_matrix', lambda c: 'matrix' in c or '[[' in c and 'self.adj' not in c),
        ('edge_set', lambda c: 'self.edges' in c and 'set' in c),
        ('hybrid', lambda c: 'self.adj' in c and 'matrix' in c),
    ],
    
    # === Caching ===
    'lru_cache': [
        ('ordereddict', lambda c: 'OrderedDict' in c),
        ('dict_with_dll', lambda c: ('self.head' in c or 'self.tail' in c) and 'cache' in c),
        ('dict_with_timestamp', lambda c: 'time' in c and 'self.cache' in c),
        ('plain_dict_list', lambda c: 'self.order' in c or 'access_order' in c),
    ],
    'caching': [
        ('ordereddict', lambda c: 'OrderedDict' in c),
        ('functools', lambda c: 'functools' in c or '@lru_cache' in c),
        ('lock_protected', lambda c: 'Lock()' in c or 'self.lock' in c),
        ('thread_local', lambda c: 'threading.local' in c or 'local()' in c),
    ],
    'memoize_lru': [
        ('ordereddict', lambda c: 'OrderedDict' in c),
        ('functools_wrapper', lambda c: 'functools.lru_cache' in c or 'functools.wraps' in c),
        ('dict_linked_list', lambda c: 'self.head' in c or 'self.tail' in c),
        ('timestamps', lambda c: 'time.time()' in c or 'timestamp' in c),
    ],
    'ttl_cache': [
        ('heap_expiry', lambda c: 'heapq' in c or 'heappush' in c),
        ('lazy_cleanup', lambda c: 'time.time()' in c and 'heapq' not in c),
        ('dict_expiry', lambda c: 'expir' in c and 'self.cache' in c),
    ],
    'write_through_cache': [
        ('write_store_first', lambda c: 'self.store' in c and 'self.cache' in c),
        ('invalidate_on_set', lambda c: 'del self.cache' in c or '.pop(' in c),
        ('delegate', lambda c: '__getitem__' in c or '__setitem__' in c),
    ],
    
    # === Algorithms ===
    'sorting': [
        ('quicksort', lambda c: 'pivot' in c),
        ('mergesort', lambda c: 'merge' in c and ('left' in c or 'mid' in c)),
        ('builtin', lambda c: '.sort()' in c or 'sorted(' in c),
        ('iterative', lambda c: 'stack' in c and 'pivot' in c),
    ],
    'search': [
        ('binary_iterative', lambda c: 'while' in c and ('low' in c or 'left' in c) and 'mid' in c),
        ('binary_recursive', lambda c: 'return' in c and 'mid' in c and 'self.' in c),
        ('bisect_module', lambda c: 'bisect' in c),
        ('linear', lambda c: 'for ' in c and 'if ' in c and 'mid' not in c and 'bisect' not in c),
    ],
    'dijkstra': [
        ('heapq', lambda c: 'heapq' in c or 'heappush' in c),
        ('sorted_list', lambda c: '.sort()' in c or 'min(' in c and 'heapq' not in c),
        ('bellman_ford', lambda c: 'for _ in range' in c and 'relax' in c.lower()),
    ],
    'topological_sort': [
        ('dfs_postorder', lambda c: 'visited' in c and ('append' in c or 'stack' in c) and 'indegree' not in c),
        ('kahns_bfs', lambda c: 'indegree' in c or 'in_degree' in c),
        ('recursive', lambda c: 'def dfs' in c or 'def visit' in c),
    ],
    'string_match': [
        ('kmp', lambda c: 'lps' in c or 'prefix' in c.lower() and 'suffix' in c.lower()),
        ('rabin_karp', lambda c: 'hash' in c and 'rolling' in c.lower() or 'mod' in c and 'base' in c),
        ('naive', lambda c: 'for i in range' in c and 'for j in range' in c),
        ('builtin', lambda c: '.find(' in c or '.index(' in c),
    ],
    'interval_merge': [
        ('sort_and_scan', lambda c: '.sort(' in c and 'result' in c),
        ('sweep_line', lambda c: 'event' in c or 'point' in c.lower() and 'start' in c and 'end' in c),
        ('reduce', lambda c: 'reduce(' in c or 'functools.reduce' in c),
    ],
    
    # === Concurrency ===
    'semaphore': [
        ('threading_semaphore', lambda c: 'threading.Semaphore' in c or 'Semaphore(' in c),
        ('counter_condition', lambda c: 'Condition' in c and 'self.count' in c),
        ('queue_tokens', lambda c: 'Queue' in c and 'token' in c.lower()),
    ],
    'barrier': [
        ('threading_barrier', lambda c: 'threading.Barrier' in c or 'Barrier(' in c),
        ('condition_notify', lambda c: 'Condition' in c and 'notify_all' in c),
        ('event_counter', lambda c: 'Event' in c and 'count' in c),
        ('generation_counter', lambda c: 'generation' in c),
    ],
    'read_write_lock': [
        ('condition_based', lambda c: 'Condition' in c and ('reader' in c or 'writer' in c)),
        ('single_lock_counter', lambda c: 'Lock' in c and 'self.readers' in c),
        ('two_locks', lambda c: 'read_lock' in c or 'write_lock' in c),
        ('rlock_counter', lambda c: 'RLock' in c),
    ],
    'actor_mailbox': [
        ('deque_per_actor', lambda c: 'deque' in c and ('self.mailbox' in c or 'self.inbox' in c)),
        ('queue_module', lambda c: 'queue.Queue' in c or 'Queue()' in c),
        ('list_with_index', lambda c: 'self.head' in c or 'self.index' in c),
    ],
    'thread_pool': [
        ('futures', lambda c: 'concurrent.futures' in c or 'ThreadPoolExecutor' in c),
        ('queue_workers', lambda c: 'Queue' in c and 'Thread' in c and 'worker' in c.lower()),
        ('dynamic_threads', lambda c: 'Thread(' in c and 'start()' in c),
    ],
    'connection_pool': [
        ('queue_blocking', lambda c: 'Queue' in c and ('get(' in c or 'put(' in c)),
        ('semaphore', lambda c: 'Semaphore' in c),
        ('list_with_lock', lambda c: 'Lock' in c and ('self.pool' in c or 'self.connections' in c)),
    ],
    
    # === Rate Limiting / Flow Control ===
    'rate_limiting': [
        ('token_bucket', lambda c: 'token' in c and ('refill' in c or 'capacity' in c)),
        ('sliding_window', lambda c: 'window' in c or ('timestamp' in c and 'request' in c.lower())),
        ('fixed_counter', lambda c: 'count' in c and 'reset' in c and 'window' not in c),
        ('leaky_bucket', lambda c: 'leak' in c or 'drain' in c),
    ],
    'backpressure': [
        ('semaphore', lambda c: 'Semaphore' in c),
        ('counter', lambda c: 'self.count' in c or 'self.current' in c and 'Semaphore' not in c),
        ('deque', lambda c: 'deque' in c),
        ('circular_buffer', lambda c: '% self.capacity' in c or 'maxlen' in c),
    ],
    'circuit_breaker': [
        ('state_machine', lambda c: 'state' in c and ('OPEN' in c or 'CLOSED' in c or 'HALF' in c)),
        ('counter_threshold', lambda c: 'failure' in c and 'threshold' in c and 'state' not in c),
        ('time_based', lambda c: 'time.time()' in c and 'reset' in c),
        ('sliding_window', lambda c: 'window' in c and 'failure' in c),
    ],
    'request_coalescing': [
        ('dict_of_futures', lambda c: 'Future' in c or 'asyncio.Future' in c),
        ('dict_of_events', lambda c: 'Event' in c and 'self.pending' in c),
        ('lock_per_key', lambda c: 'Lock' in c and ('self.locks' in c or 'locks[' in c)),
    ],
    
    # === Retry / Timer ===
    'retry': [
        ('fixed_delay', lambda c: 'sleep' in c and 'attempt' in c and '**' not in c and '* 2' not in c),
        ('exponential_backoff', lambda c: 'sleep' in c and ('**' in c or '* 2' in c or '*2' in c)),
        ('jitter', lambda c: 'random' in c and 'sleep' in c),
    ],
    'timer': [
        ('threading_timer', lambda c: 'threading.Timer' in c or 'Timer(' in c),
        ('thread_sleep', lambda c: 'Thread' in c and 'time.sleep' in c),
        ('event_flag', lambda c: 'Event' in c and ('wait(' in c or 'set()' in c)),
        ('loop_thread', lambda c: 'while' in c and 'sleep' in c and 'self.running' in c),
    ],
    
    # === Events / PubSub ===
    'events': [
        ('list_handlers', lambda c: 'append' in c and ('handler' in c or 'listener' in c or 'callback' in c)),
        ('dict_handlers', lambda c: 'self.listeners' in c or 'self.handlers' in c),
        ('weakref', lambda c: 'weakref' in c or 'WeakSet' in c),
    ],
}

def detect_approach(code: str, topic: Optional[str], task_id: Optional[str], tasks_meta: Optional[Dict[str, Dict[str, Any]]]) -> Optional[str]:
    """Detect implementation approach using shared heuristics."""
    return shared_detect_approach(code, topic=topic, task_id=task_id, tasks_meta=tasks_meta)

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
    topic: Optional[str]
    failure_mode: Optional[str]
    decisions: List[DecisionPoint]
    
    final_success: bool
    n_decisions: int
    error_sequence: List[Optional[str]]
    severity_sequence: List[int]
    
    # Stability metrics
    oscillation_count: int
    n_regressions: int  # severity increased
    progression_type: str
    no_decision: bool
    
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

def _tier_from_meta(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, int):
        return f"tier{value}"
    if isinstance(value, str):
        v = value.strip().lower()
        if v.startswith("tier"):
            return v
        if v.isdigit():
            return f"tier{v}"
        return v
    return None


def extract_tier(task_id: str, tasks_meta: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    if tasks_meta and task_id in tasks_meta:
        tier = _tier_from_meta(tasks_meta.get(task_id, {}).get("tier"))
        if tier:
            return tier
    if task_id.startswith('tier'):
        return task_id[:5]  # 'tier1', 'tier2', etc.
    if task_id.startswith('osc'):
        return 'oscillation'
    return 'unknown'

def extract_quantization(variant_id: str) -> str:
    """Normalize quantization string (preserve suffix like q4_k_m)."""
    if not variant_id:
        return "unknown"
    match = re.search(r'([qQ]\d+(?:_[kKmMsS]+)*)', str(variant_id))
    return match.group(1).lower() if match else str(variant_id).lower()

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

def parse_model_info(name: str) -> Tuple[str, str]:
    """
    Extract (model_id, quantization) from a run dir or model_id string.
    Examples:
      yy_qwen2_5_coder_7b_instruct_q6_k -> (qwen2_5_coder_7b_instruct, q6_k)
      qwen2_5_coder_7b_instruct_q6_k   -> (qwen2_5_coder_7b_instruct, q6_k)
    """
    # Remove common run tag prefixes
    clean = name or ""
    for prefix in ['yy_', 'zz_', 'zb_', 'zc_', 'w_', 'z_', 'xx_']:
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            break

    # Look for quantization pattern at the end
    quant_pattern = re.compile(r'_([qQ]\d+(?:_[kKmMsS]+)*)$')

    match = quant_pattern.search(clean)
    if match:
        quant_part = match.group(1)
        model_id = clean[:match.start()]
        variant_id = extract_quantization(quant_part)
    else:
        model_id = clean
        variant_id = 'unknown'

    return model_id, variant_id

def load_trajectories(run_dirs: List[Path], tasks_meta: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Trajectory]:
    trajectories = []
    
    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"Warning: Directory not found: {run_dir}")
            continue
            
        dirname = run_dir.name
        
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
            if category == 'unknown' and tasks_meta:
                category = tasks_meta.get(task_id, {}).get("category") or category
            topic = run.get('topic') or (tasks_meta.get(task_id, {}).get("topic") if tasks_meta else None)
            success = run.get('success', 0)
            failure_mode = run.get('failure_mode')

            # Prefer model_id from the run log; fall back to directory name
            model_id_raw = run.get('model_id') or dirname
            model_id, variant_id = parse_model_info(str(model_id_raw))
            if variant_id == 'unknown' and model_id_raw != dirname:
                # If quantization not in model_id, try the directory name
                _, dir_variant = parse_model_info(dirname)
                if dir_variant != 'unknown':
                    variant_id = dir_variant
            
            steps = steps_by_run.get(run_id, [])
            steps.sort(key=lambda x: int(x.get('step_id', 0)))
            
            # Build decision points by pairing code_fix with subsequent test
            decisions = []
            last_code = None
            fix_attempt = 0
            pending_decision = None
            starter_code = None
            starter_approach = None
            
            for step in steps:
                step_type = step.get('step_type')
                
                if step_type == 'code_starter':
                    last_code = step.get('code', '')
                    starter_code = last_code
                    step_topic = topic
                    if not step_topic:
                        meta = step.get('meta') or {}
                        step_topic = meta.get('topic')
                    if starter_code:
                        starter_approach = detect_approach(starter_code, step_topic, task_id, tasks_meta)
                    
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

                                        
                    # Detect architectural approach from code using topic-aware patterns
                    approach_class = None
                    code_to_analyze = pending_decision['code_after']
                    
                    # Get topic from run metadata, fallback to starter meta
                    step_topic = topic
                    if not step_topic:
                        for s in steps:
                            if s.get('step_type') == 'code_starter':
                                meta = s.get('meta') or {}
                                step_topic = meta.get('topic')
                                break
                    
                    if code_to_analyze:
                        approach_class = detect_approach(code_to_analyze, step_topic, task_id, tasks_meta)
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
            
            # Compute metrics
            error_seq = [d.error_type for d in decisions]
            severity_seq = [d.severity for d in decisions]

            if decisions:
                oscillations = count_severity_oscillations(severity_seq)
                regressions = count_regressions(severity_seq)
                prog_type = classify_progression(error_seq, severity_seq, bool(success))
            else:
                oscillations = 0
                regressions = 0
                if success:
                    prog_type = "no_decision_success"
                else:
                    prog_type = "no_decision_failure"
            
            first_step = steps[0] if steps else {}
            last_step = steps[-1] if steps else {}
            time_first = decisions[0].latency_ms if decisions else 0
            total_time = last_step.get('ended_ms', 0) - first_step.get('started_ms', 0)

            approach_seq: List[Optional[str]] = []
            if starter_code is not None:
                approach_seq.append(starter_approach)
            approach_seq.extend(d.approach_class for d in decisions)
            approach_osc = count_approach_oscillations(approach_seq)  # create this function
            
            traj = Trajectory(
                run_id=run_id,
                model_id=model_id,
                variant_id=variant_id,
                task_id=task_id,
                tier=extract_tier(task_id, tasks_meta),
                category=category,
                topic=topic,
                failure_mode=failure_mode,
                decisions=decisions,
                final_success=bool(success),
                n_decisions=len(decisions),
                error_sequence=error_seq,
                severity_sequence=severity_seq,
                oscillation_count=oscillations,
                n_regressions=regressions,
                progression_type=prog_type,
                no_decision=(len(decisions) == 0),
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
    tasks_path: Optional[str] = None
    min_approach_coverage: float = 0.0
    validate: bool = False

def aggregate_by_tier_quant_model(trajectories: List[Trajectory]) -> Tuple[Dict, Dict]:
    """Aggregate metrics by (tier, quantization, model). Returns (results, coverage_stats)."""
    groups = defaultdict(lambda: {
        'count': 0,
        'successes': 0,
        'oscillations': [],
        'regressions': [],
        'thrashing_count': 0,
        'stuck_count': 0,
        'no_decision_count': 0,
        'n_decisions': [],
        'latencies': [],
        'approach_osc': [],
        'decisions_with_approach': 0,
        'decisions_total': 0,
        'approach_switches': 0,  # A->B (not necessarily A->B->A)
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
        if t.no_decision and not t.final_success:
            g['no_decision_count'] += 1
        
        # Track approach detection coverage
        for d in t.decisions:
            g['decisions_total'] += 1
            if d.approach_class and d.approach_class != 'unknown':
                g['decisions_with_approach'] += 1
        
        # Count approach switches (A->B, regardless of whether it goes back)
        clean_seq = [a for a in t.approach_sequence if a and a != 'unknown']
        for i in range(1, len(clean_seq)):
            if clean_seq[i] != clean_seq[i-1]:
                g['approach_switches'] += 1
            
    # Compute summaries
    results = {}
    coverage_stats = {}
    
    for key, g in groups.items():
        n = g['count']
        if n == 0:
            continue
        
        coverage = g['decisions_with_approach'] / g['decisions_total'] if g['decisions_total'] > 0 else 0
        
        results[key] = {
            'tier': key[0],
            'quantization': key[1],
            'model': key[2],
            'n_runs': n,
            'success_rate': g['successes'] / n,
            'thrash_rate': g['thrashing_count'] / n,
            'stuck_rate': g['stuck_count'] / n,
            'no_decision_rate': g['no_decision_count'] / n,
            'avg_oscillations': statistics.mean(g['oscillations']) if g['oscillations'] else 0,
            'avg_regressions': statistics.mean(g['regressions']) if g['regressions'] else 0,
            'avg_decisions': statistics.mean(g['n_decisions']) if g['n_decisions'] else 0,
            'med_latency_ms': statistics.median(g['latencies']) if g['latencies'] else 0,
            'avg_approach_osc': statistics.mean(g['approach_osc']) if g['approach_osc'] else 0,
            'approach_detection_coverage': coverage,
            'total_approach_switches': g['approach_switches'],
            'approach_switch_rate': g['approach_switches'] / n,
        }
        
        coverage_stats[key] = {
            'decisions_with_approach': g['decisions_with_approach'],
            'decisions_total': g['decisions_total'],
            'coverage_pct': coverage,
        }

    # Coverage by topic (debug)
    topic_coverage = defaultdict(lambda: {'detected': 0, 'total': 0})
    for t in trajectories:
        topic = t.topic or 'unknown'
        for d in t.decisions:
            topic_coverage[topic]['total'] += 1
            if d.approach_class and d.approach_class != 'unknown':
                topic_coverage[topic]['detected'] += 1

    if topic_coverage:
        print("\n--- Coverage by Topic ---")
        for topic, stats in sorted(topic_coverage.items()):
            pct = stats['detected'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {topic}: {stats['detected']}/{stats['total']} ({pct:.0%})")
    
    return results, coverage_stats

def print_insights(results: Dict, coverage_stats: Dict):
    """Print key findings with confidence intervals."""
    print("\n" + "="*80)
    print("DECISION STABILITY ANALYSIS")
    print("="*80)
    
    # Coverage warning
    print("\n--- Approach Detection Coverage ---")
    total_detected = sum(c['decisions_with_approach'] for c in coverage_stats.values())
    total_decisions = sum(c['decisions_total'] for c in coverage_stats.values())
    overall_coverage = total_detected / total_decisions if total_decisions > 0 else 0
    
    print(f"Overall: {total_detected}/{total_decisions} decisions ({overall_coverage:.1%})")
    if overall_coverage < 0.5:
        print("⚠️  WARNING: Low coverage - approach oscillation metrics may be unreliable")
    
    for key, cov in coverage_stats.items():
        print(f"  {key[0]} {key[1]}: {cov['coverage_pct']:.1%}")

    print("\n--- Approach Switching Analysis ---")
    for key, data in results.items():
        if data['tier'] == 'oscillation':
            n = data['n_runs']
            switches = data['total_approach_switches']
            print(f"{data['model'][:20]} ({data['quantization']}): "
                  f"{switches} switches across {n} runs "
                  f"({data['approach_switch_rate']:.2f}/run)")
    
    print("\n--- Success Rates with 95% CI ---")
    print(f"{'Tier':<12} {'Quant':<6} {'Success':<12} {'95% CI':<16} {'N':<6}")
    print("-" * 60)
    
    by_tier = defaultdict(list)
    for key, data in results.items():
        by_tier[data['tier']].append(data)
    
    for tier in sorted(by_tier.keys()):
        for data in by_tier[tier]:
            n = data['n_runs']
            successes = int(data['success_rate'] * n)
            low, high = wilson_ci(successes, n)
            print(f"{data['tier']:<12} {data['quantization']:<6} "
                  f"{data['success_rate']:<12.1%} [{low:.1%}, {high:.1%}]  {n:<6}")
    
    print("\n--- Stability Metrics ---")
    print(f"{'Tier':<12} {'Stuck%':<10} {'NoDec%':<10} {'Thrash%':<10} {'AvgOsc':<10} {'ApproachOsc':<12}")
    print("-" * 60)
    
    for tier in sorted(by_tier.keys()):
        for data in by_tier[tier]:
            print(f"{data['tier']:<12} {data['stuck_rate']:<10.1%} "
                  f"{data['no_decision_rate']:<10.1%} {data['thrash_rate']:<10.1%} {data['avg_oscillations']:<10.2f} "
                  f"{data['avg_approach_osc']:<12.2f}")

def main(args: Args):
    run_dirs = [Path(d) for d in args.run_dirs]
    
    print(f"Loading trajectories from {len(run_dirs)} directories...")
    tasks_meta = None
    if args.tasks_path:
        tasks_meta = load_tasks_metadata(Path(args.tasks_path))
    else:
        default_tasks = Path("data/tasks/v8_osc_total_200.jsonl")
        if default_tasks.exists():
            tasks_meta = load_tasks_metadata(default_tasks)

    trajectories = load_trajectories(run_dirs, tasks_meta=tasks_meta)
    print(f"Loaded {len(trajectories)} trajectories")
    
    if not trajectories:
        print("No trajectories found.")
        return
    
    results, coverage_stats = aggregate_by_tier_quant_model(trajectories)
    print_insights(results, coverage_stats)

    # Validation: approach coverage
    total_detected = sum(c['decisions_with_approach'] for c in coverage_stats.values())
    total_decisions = sum(c['decisions_total'] for c in coverage_stats.values())
    overall_coverage = total_detected / total_decisions if total_decisions > 0 else 0.0
    if args.validate and args.min_approach_coverage > 0:
        if overall_coverage < args.min_approach_coverage:
            raise AssertionError(
                f"Approach coverage too low: {overall_coverage:.1%} < {args.min_approach_coverage:.1%}"
            )
    
    if args.output_json:
        import json as jsonlib
        output = {
            'results': {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in results.items()},
            'coverage': {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in coverage_stats.items()},
        }
        with open(args.output_json, 'w') as f:
            jsonlib.dump(output, f, indent=2)
        print(f"\nSaved detailed results to {args.output_json}")

if __name__ == "__main__":
    main(tyro.cli(Args))
