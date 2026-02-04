"""
Shared approach detection utilities used by analysis and runner.

This module centralizes heuristic approach classification so that
oscillation metrics and multi-model analysis are consistent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ----------------------------
# Task metadata helpers
# ----------------------------

def load_tasks_metadata(tasks_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load task metadata keyed by task_id."""
    tasks: Dict[str, Dict[str, Any]] = {}
    if not tasks_path.exists():
        return tasks
    with tasks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            task_id = d.get("task_id")
            if not task_id:
                continue
            tasks[str(task_id)] = {
                "topic": d.get("topic"),
                "approaches": d.get("approaches") or [],
                "tier": d.get("tier"),
                "category": d.get("category"),
            }
    return tasks


_TASK_ID_TOPIC_HINTS = {
    "osc_queue": "queue_impl",
    "osc_stack": "stack_impl",
    "osc_cache": "lru_cache",
    "osc_graph": "graph",
    "osc_sort": "sorting",
    "osc_search": "search",
    "osc_hash": "hashtable",
    "osc_pq": "priority_queue",
    "osc_tree": "bst",
    "osc_dll": "linked_list",
    "osc_timer": "timer",
    "osc_rate_limit": "rate_limiting",
    "osc_event": "events",
    "osc_thread_pool": "thread_pool",
    "osc_connection_pool": "connection_pool",
    "osc_trie": "trie",
    "osc_heap": "heap",
    "osc_disjoint": "disjoint_set",
    "osc_ring": "ring_buffer",
    "osc_skip": "skip_list",
    "osc_dijkstra": "dijkstra",
    "osc_topological_sort": "topological_sort",
    "osc_string_match": "string_match",
    "osc_interval_merge": "interval_merge",
    "osc_semaphore": "semaphore",
    "osc_barrier": "barrier",
    "osc_rwlock": "read_write_lock",
    "osc_actor": "actor_mailbox",
    "osc_write_through": "write_through_cache",
    "osc_ttl": "ttl_cache",
    "osc_memoize": "memoize_lru",
    "osc_circuit_breaker": "circuit_breaker",
    "osc_request_coalescing": "request_coalescing",
    "osc_backpressure": "backpressure",
    "osc_caching": "caching",
    "osc_retry": "retry",
}


def infer_topic_from_task_id(task_id: Optional[str]) -> Optional[str]:
    if not task_id:
        return None
    tid = task_id.lower()
    for prefix, topic in _TASK_ID_TOPIC_HINTS.items():
        if prefix in tid:
            return topic
    return None


# ----------------------------
# Pattern library (topic-specific)
# ----------------------------

APPROACH_PATTERNS = {
    # === Data Structures ===
    'queue_impl': [
        ('collections_deque', lambda c: 'deque' in c and ('popleft' in c or 'appendleft' in c)),
        ('list_with_indices', lambda c: ('self.front' in c or 'self.rear' in c or 'front' in c or 'rear' in c) and 'deque' not in c),
        ('circular_buffer', lambda c: '% self.capacity' in c or '% len(' in c or '% self.size' in c or 'circular' in c),
        ('list_only', lambda c: 'pop(0)' in c and 'deque' not in c),
    ],
    'stack_impl': [
        ('list_append_pop', lambda c: '.append(' in c and '.pop(' in c and 'self.top' not in c and 'self.head' not in c),
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
        ('chaining', lambda c: 'bucket' in c or ('append' in c and 'self.table' in c) or 'buckets' in c),
        ('tombstone', lambda c: 'tombstone' in c or 'DELETED' in c or 'deleted' in c),
        ('python_dict', lambda c: 'self.data = {}' in c or 'self._dict' in c or 'dict(' in c),
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
        ('builtin', lambda c: '.sort(' in c or 'sorted(' in c or 'cmp_to_key' in c),
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
        ('sort_and_scan', lambda c: ('.sort(' in c or 'sorted(' in c) and ('result' in c or 'merged' in c or 'out' in c or 'interval' in c)),
        ('sweep_line', lambda c: ('event' in c or 'point' in c or 'sweep' in c) and ('start' in c and 'end' in c)),
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
        ('simple_loop', lambda c: 'try' in c and 'except' in c and ('for ' in c or 'while ' in c) and 'sleep' not in c),
    ],
    'timer': [
        ('threading_timer', lambda c: 'threading.Timer' in c or 'Timer(' in c),
        ('thread_sleep', lambda c: 'Thread' in c and 'time.sleep' in c),
        ('event_flag', lambda c: 'Event' in c and ('wait(' in c or 'set()' in c)),
        ('loop_thread', lambda c: 'while' in c and 'sleep' in c and 'self.running' in c),
        ('manual_elapsed', lambda c: ('elapsed' in c or 'interval' in c) and ('tick' in c or 'delta' in c or 'ms' in c)),
    ],
    
    # === Events / PubSub ===
    'events': [
        ('list_handlers', lambda c: 'append' in c and ('handler' in c or 'listener' in c or 'callback' in c)),
        ('dict_handlers', lambda c: 'self.listeners' in c or 'self.handlers' in c),
        ('weakref', lambda c: 'weakref' in c or 'WeakSet' in c),
    ],
}


# ----------------------------
# Fallback heuristics
# ----------------------------

_STRING_RE = re.compile(r"('''.*?'''|\"\"\".*?\"\"\"|'.*?'|\".*?\")", re.DOTALL)
_COMMENT_RE = re.compile(r"#.*")


def _normalize_code(code: str) -> str:
    text = _STRING_RE.sub(" ", code)
    text = _COMMENT_RE.sub(" ", text)
    return text.lower()


def _code_tokens(code_norm: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", code_norm))


def _has_recursion(code_norm: str) -> bool:
    funcs = re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code_norm)
    for name in funcs:
        if code_norm.count(name + "(") > 1:
            return True
    return False


def _build_features(code_norm: str, tokens: set[str]) -> Dict[str, bool]:
    return {
        "deque": "deque" in tokens or "popleft" in tokens or "appendleft" in tokens,
        "heapq": "heapq" in tokens or "heappush" in tokens or "heappop" in tokens,
        "bisect": "bisect" in tokens,
        "ordereddict": "ordereddict" in code_norm,
        "list": "append" in tokens or "pop" in tokens or "list" in tokens,
        "set": "set" in tokens,
        "dict": "dict" in tokens or "{" in code_norm,
        "linked": "next" in tokens or "prev" in tokens or "node" in tokens,
        "node": "node" in tokens or "next" in tokens,
        "recursive": _has_recursion(code_norm),
        "iterative": "while" in tokens or "for" in tokens,
        "matrix": "[[" in code_norm or "matrix" in tokens,
        "adj": "adj" in tokens or "adjacency" in tokens,
        "sorted": "sorted" in tokens or ".sort(" in code_norm,
        "heap": "heap" in tokens or "sift" in tokens,
        "token": "token" in tokens,
        "window": "window" in tokens or "timestamp" in tokens,
        "semaphore": "semaphore" in tokens,
        "barrier": "barrier" in tokens,
        "event": "event" in tokens,
        "timer": "timer" in tokens,
        "thread": "thread" in tokens or "threading" in tokens,
        "queue": "queue" in tokens or "deque" in tokens,
        "lock": "lock" in tokens or "rlock" in tokens,
        "state": "state" in tokens,
        "frequency": "freq" in tokens or "frequency" in tokens,
        "time": "time" in tokens,
        "ttl": "ttl" in tokens or "expire" in code_norm,
        "bucket": "bucket" in tokens,
        "leaky": "leak" in tokens or "drain" in tokens,
        "hash": "hash" in tokens,
        "rolling": "rolling" in tokens,
        "dp": "dp" in tokens,
    }


_NAME_HINTS: Dict[str, Tuple[str, int]] = {
    "deque": ("deque", 3),
    "heapq": ("heapq", 3),
    "bisect": ("bisect", 3),
    "ordereddict": ("ordereddict", 3),
    "linked": ("linked", 2),
    "node": ("node", 2),
    "recursive": ("recursive", 2),
    "iterative": ("iterative", 2),
    "matrix": ("matrix", 2),
    "adj": ("adj", 2),
    "sorted": ("sorted", 2),
    "heap": ("heap", 2),
    "list": ("list", 1),
    "dict": ("dict", 1),
    "set": ("set", 1),
    "token": ("token", 2),
    "window": ("window", 2),
    "semaphore": ("semaphore", 3),
    "barrier": ("barrier", 3),
    "event": ("event", 2),
    "timer": ("timer", 2),
    "thread": ("thread", 2),
    "queue": ("queue", 1),
    "lock": ("lock", 2),
    "state": ("state", 2),
    "frequency": ("frequency", 2),
    "ttl": ("ttl", 2),
    "bucket": ("bucket", 2),
    "leaky": ("leaky", 2),
    "hash": ("hash", 1),
    "rolling": ("rolling", 2),
    "dp": ("dp", 2),
}


def _score_approach_name(name: str, features: Dict[str, bool]) -> int:
    score = 0
    n = name.lower()
    for token, (feature_key, weight) in _NAME_HINTS.items():
        if token in n and features.get(feature_key):
            score += weight
    return score


def _detect_from_approach_names(code: str, approaches: Iterable[str]) -> Optional[str]:
    code_norm = _normalize_code(code)
    tokens = _code_tokens(code_norm)
    features = _build_features(code_norm, tokens)

    best_name: Optional[str] = None
    best_score = 0

    for name in approaches:
        score = _score_approach_name(str(name), features)
        if score > best_score:
            best_score = score
            best_name = str(name)

    return best_name if best_score > 0 else None


# ----------------------------
# Public API
# ----------------------------

def detect_approach(
    code: str,
    *,
    topic: Optional[str] = None,
    task_id: Optional[str] = None,
    tasks_meta: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[str]:
    """Detect implementation approach using topic patterns and task metadata."""
    if not code:
        return None

    if not topic and tasks_meta and task_id:
        topic = tasks_meta.get(task_id, {}).get("topic")
    if not topic:
        topic = infer_topic_from_task_id(task_id)

    if topic:
        patterns = APPROACH_PATTERNS.get(topic, [])
        for name, detector in patterns:
            try:
                if detector(code):
                    return name
            except Exception:
                continue

    # Fallback: infer by approach names from tasks metadata if available
    if tasks_meta and task_id in tasks_meta:
        approaches = tasks_meta.get(task_id, {}).get("approaches") or []
        inferred = _detect_from_approach_names(code, approaches)
        if inferred:
            return inferred

    return None
