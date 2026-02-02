# approach_classifier.py
"""
Classifies code into architectural approaches for oscillation detection.
Add this to your runner.py and call classify_approach() on each code snapshot.
"""

from typing import Optional


def classify_approach(code: str, task_id: str) -> Optional[str]:
    """
    Detect which architectural approach the code uses based on task_id and code patterns.
    Returns approach name or None if unclassifiable.
    """
    if not code or not task_id:
        return None
    
    code_lower = code.lower()
    
    # osc_queue_list_vs_deque_01
    if "osc_queue" in task_id:
        if "from collections import deque" in code or "collections.deque" in code or "deque()" in code:
            return "collections_deque"
        elif "self.head" in code and "self.items" in code:
            return "list_with_head"
        elif "self.items" in code or "self.data" in code:
            return "list_simple"
        return None
    
    # osc_stack_array_vs_linked_01
    if "osc_stack" in task_id:
        if '["item"]' in code or "['item']" in code or '["next"]' in code or "['next']" in code:
            return "linked_list_dict"
        elif "self.top" in code and "node" in code_lower:
            return "linked_list"
        elif "self.items" in code and ".append" in code and ".pop()" in code:
            return "python_list"
        return None
    
    # osc_cache_lru_vs_lfu_01
    if "osc_cache_lru_vs_lfu" in task_id:
        if "ordereddict" in code_lower:
            return "lru_ordereddict"
        elif "access_count" in code or "frequency" in code_lower or "freq" in code_lower:
            return "lfu_frequency"
        elif "access_order" in code or "move_to_end" in code:
            return "lru_manual"
        return "basic_dict"
    
    # osc_sort_recursive_vs_iterative_01
    if "osc_sort" in task_id:
        if "sort_list(left)" in code or "sort_list(right)" in code:
            return "quicksort_recursive"
        elif "sorted(" in code:
            return "builtin_sorted"
        elif ".sort()" in code:
            return "inplace_sort"
        elif "while" in code and ("swap" in code_lower or "arr[" in code):
            return "iterative_sort"
        return None
    
    # osc_search_linear_vs_binary_01
    if "osc_search" in task_id:
        if "mid" in code_lower and ("//" in code or "// 2" in code):
            return "binary_search"
        elif "bisect" in code_lower:
            return "bisect_module"
        elif "for " in code and "range" in code:
            return "linear_scan"
        return None
    
    # osc_graph_adj_list_vs_matrix_01
    if "osc_graph" in task_id:
        if "self.matrix" in code and "[u][v]" in code:
            return "adjacency_matrix"
        elif "self.edges" in code and (".add(" in code or "set()" in code):
            return "adjacency_list"
        return None
    
    # osc_set_bitmap_vs_hash_01
    if "osc_set" in task_id:
        if "self.data[" in code and (">> " in code or "<< " in code or "& " in code or "| " in code):
            return "bitmap"
        elif "self.values" in code or "set()" in code or "{}" in code:
            return "hash_set"
        return None
    
    # osc_json_recursive_vs_iterative_01
    if "osc_json" in task_id:
        if "to_json(" in code and ("to_json(v)" in code or "to_json(x)" in code):
            return "recursive_descent"
        elif "json.dumps" in code:
            return "builtin_json"
        elif "stack" in code_lower and "while" in code:
            return "iterative_stack"
        return None
    
    # osc_parser_regex_vs_fsm_01 (email validation)
    if "osc_parser" in task_id or "email" in task_id.lower():
        if "import re" in code or "re.match" in code or "re.search" in code:
            return "regex_pattern"
        elif "state" in code_lower or "for char in" in code_lower:
            return "state_machine"
        return None
    
    # osc_dedup_sort_vs_set_01
    if "osc_dedup" in task_id:
        if "seen" in code and ".add(" in code and "result.append" in code:
            return "order_preserving_seen"
        elif "set(" in code and "sorted(" in code:
            return "set_then_sort"
        elif "list(set(" in code:
            return "set_unordered"
        return None
    
    # osc_rate_limit_token_vs_leaky_01
    if "osc_rate_limit" in task_id:
        if "tokens" in code and "self.tokens" in code:
            return "token_bucket"
        elif "window" in code_lower or "timestamps" in code:
            return "sliding_window"
        return None
    
    # osc_priority_heap_vs_sorted_01
    if "osc_priority" in task_id:
        if "heapq" in code or "heappush" in code or "heappop" in code:
            return "heapq"
        elif ".sort(" in code:
            return "sorted_list"
        return None
    
    # osc_event_pub_sub_vs_observer_01
    if "osc_event" in task_id:
        if "self.listeners" in code and "dict" in code_lower:
            return "pub_sub_dict"
        elif "self.listener" in code and "list" not in code_lower:
            return "single_observer"
        elif "append" in code and ("listeners" in code or "callbacks" in code):
            return "observer_list"
        return None
    
    # osc_timer_call_later_vs_loop_01
    if "osc_timer" in task_id:
        if "threading.Timer" in code:
            return "threading_timer"
        elif "time.sleep" in code and "while" in code:
            return "manual_loop"
        elif "self.thread" in code:
            return "custom_thread"
        return None
    
    # osc_file_line_buffered_vs_chunk_01
    if "osc_file" in task_id:
        if ".read()" in code and ".split" in code:
            return "read_all_split"
        elif "for line in f" in code or ".readlines()" in code:
            return "buffered_iterator"
        return None
    
    # osc_matrix_dense_vs_sparse_01
    if "osc_matrix" in task_id:
        if "self.data = {}" in code or "self.data.get(" in code:
            return "dict_sparse"
        elif "[[0]" in code or "[[" in code:
            return "dense_2d_list"
        return None
    
    # osc_thread_pool_fixed_vs_dynamic_01
    if "osc_thread_pool" in task_id:
        if "for _ in range(max_workers)" in code:
            return "fixed_threads"
        elif "if len(self.threads) < self.max" in code:
            return "dynamic_grow"
        return None
    
    # osc_csv_parser_pandas_vs_manual_01
    if "osc_csv" in task_id:
        if "csv.reader" in code or "import csv" in code:
            return "csv_module"
        elif "state" in code_lower or "in_quotes" in code:
            return "manual_state_machine"
        elif ".split(',')" in code and '"' not in code[code.find("split"):]:
            return "naive_split"
        return None
    
    # osc_uuid_random_vs_time_01
    if "osc_uuid" in task_id:
        if "uuid.uuid4" in code or "import uuid" in code:
            return "uuid_module"
        elif "time.time" in code or "time.time_ns" in code:
            return "time_based"
        elif "random.choice" in code:
            return "random_chars"
        return None
    
    # osc_regex_nfa_vs_dfa_01 (wildcard matching)
    if "osc_regex" in task_id:
        if "match(pattern[1:]" in code:
            return "recursive_backtrack"
        elif "dp[" in code or "dp =" in code:
            return "dynamic_programming"
        return None
    
    # osc_bloom_counter_vs_bitarr_01
    if "osc_bloom" in task_id:
        if "+= 1" in code and "self.bit_array[h]" in code:
            return "counter_array"
        elif "= 1" in code and "self.bit_array[h]" in code:
            return "bit_array"
        return None
    
    # osc_counter_exact_vs_hyperloglog_01
    if "osc_counter" in task_id:
        if "set(" in code or "self.seen = set" in code:
            return "exact_set"
        elif "hash" in code_lower and "leading" in code_lower:
            return "hyperloglog"
        elif "self.items" in code and "list" in code_lower:
            return "exact_list"
        return None
    
    # osc_metrics_counter_vs_gauge_01
    if "osc_metrics" in task_id:
        if "+=" in code and "counter" in code_lower:
            return "cumulative_counter"
        elif "self.values[name] = value" in code:
            return "always_set"
        return None
    
    # osc_id_gen_seq_vs_snowflake_01
    if "osc_id_gen" in task_id:
        if "time.time" in code or "time.time_ns" in code:
            return "timestamp_based"
        elif "self.counter += 1" in code:
            return "sequential_counter"
        return None
    
    # osc_lock_mutex_vs_rwlock_01
    if "osc_lock" in task_id:
        if "self.readers" in code or "reader_count" in code:
            return "reader_writer_separate"
        elif "self.lock.acquire" in code and "self.lock.release" in code:
            return "mutex_all"
        return None
    
    # osc_hash_md5_vs_sha_01
    if "osc_hash" in task_id:
        if "sha256" in code or "sha512" in code:
            return "sha256"
        elif "md5" in code:
            return "md5"
        return None
    
    # osc_timeout_poll_vs_callback_01
    if "osc_timeout" in task_id:
        if "threading.Timer" in code or "threading.Thread" in code:
            return "threading_timer"
        elif "signal.alarm" in code or "signal.SIGALRM" in code:
            return "signal_based"
        elif "time.time()" in code and "elapsed" in code:
            return "polling_elapsed"
        return None
    
    # osc_retry_linear_vs_exponential_01
    if "osc_retry" in task_id:
        if "* 2" in code or "** " in code or "2 **" in code:
            return "exponential_backoff"
        elif "time.sleep(delay)" in code:
            return "fixed_delay"
        return None
    
    # osc_cache_ttl_vs_lru_01
    if "osc_cache_ttl" in task_id:
        if "time.time()" in code and ("expir" in code_lower or "timestamp" in code_lower):
            return "strict_expiry"
        elif "ordereddict" in code_lower or "move_to_end" in code:
            return "lru_with_ttl"
        return None
    
    # osc_writer_buffer_vs_stream_01
    if "osc_writer" in task_id:
        if "self.file = open" in code or "self.f = open" in code:
            return "streaming_immediate"
        elif "self.buffer" in code and ".append(" in code:
            return "buffer_all"
        return None
    
    return None


# Quick test when run directly
if __name__ == "__main__":
    test_cases = [
        ("osc_queue_list_vs_deque_01", "from collections import deque\nself.items = deque()", "collections_deque"),
        ("osc_queue_list_vs_deque_01", "self.items = []\nself.head = 0", "list_with_head"),
        ("osc_cache_lru_vs_lfu_01", "from collections import OrderedDict", "lru_ordereddict"),
        ("osc_search_linear_vs_binary_01", "mid = (low + high) // 2", "binary_search"),
    ]
    
    for task_id, code, expected in test_cases:
        result = classify_approach(code, task_id)
        status = "✓" if result == expected else "✗"
        print(f"{status} {task_id}: expected={expected}, got={result}")