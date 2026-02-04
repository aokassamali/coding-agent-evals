from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional


def _dedent(text: str) -> str:
    return textwrap.dedent(text).strip() + "\n"


def _make_task(
    *,
    task_id: str,
    prompt: str,
    signature: str,
    starter_code: str,
    tests: str,
    category: str,
    topic: str,
    tier: Any,
    starter_check: str = "fail",
    approaches: Optional[List[str]] = None,
) -> Dict[str, Any]:
    task: Dict[str, Any] = {
        "task_id": task_id,
        "prompt": prompt,
        "signature": signature,
        "starter_code": starter_code,
        "tests": tests,
        "category": category,
        "topic": topic,
        "tier": tier,
        "starter_check": starter_check,
    }
    if approaches is not None:
        task["approaches"] = approaches
    return task


def _write_jsonl(path: Path, tasks: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=True) + "\n")


# -----------------------------
# Helper implementations for expected outputs
# -----------------------------

def _rotate_right(items: List[Any], k: int) -> List[Any]:
    if not items:
        return []
    k = k % len(items)
    if k == 0:
        return list(items)
    return items[-k:] + items[:-k]


def _chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _flatten_once(items: List[Any]) -> List[Any]:
    out: List[Any] = []
    for x in items:
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out


def _dedupe_preserve(items: List[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _merge_intervals(intervals: List[tuple]) -> List[tuple]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [tuple(x) for x in merged]


def _group_anagrams(words: List[str]) -> List[List[str]]:
    groups: Dict[str, List[str]] = {}
    for w in words:
        key = "".join(sorted(w))
        groups.setdefault(key, []).append(w)
    normalized = [sorted(g) for g in groups.values()]
    return sorted(normalized, key=lambda g: g[0])


def _top_k_frequent(items: List[Any], k: int) -> List[Any]:
    freq: Dict[Any, int] = {}
    for x in items:
        freq[x] = freq.get(x, 0) + 1
    ordered = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [x for x, _ in ordered[:k]]


def _normalize_path(path: str) -> str:
    parts: List[str] = []
    for part in path.split("/"):
        if part == "" or part == ".":
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/" + "/".join(parts)


def _rle_encode(s: str) -> str:
    if not s:
        return ""
    out: List[str] = []
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            out.append(f"{s[i - 1]}{count}")
            count = 1
    out.append(f"{s[-1]}{count}")
    return "".join(out)


def _find_all_overlapping(text: str, pattern: str) -> List[int]:
    if not pattern:
        return []
    out: List[int] = []
    plen = len(pattern)
    for i in range(0, len(text) - plen + 1):
        if text[i : i + plen] == pattern:
            out.append(i)
    return out


def _is_balanced(text: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: List[str] = []
    for ch in text:
        if ch in pairs.values():
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack


def _bfs_shortest_path_len(graph: Dict[Any, List[Any]], start: Any, end: Any) -> int:
    if start == end:
        return 0
    queue = [(start, 0)]
    visited = {start}
    while queue:
        node, dist = queue.pop(0)
        for nbr in graph.get(node, []):
            if nbr == end:
                return dist + 1
            if nbr not in visited:
                visited.add(nbr)
                queue.append((nbr, dist + 1))
    return -1


def _topological_sort(graph: Dict[Any, List[Any]]) -> List[Any]:
    indegree: Dict[Any, int] = {}
    for node in graph:
        indegree.setdefault(node, 0)
        for nbr in graph[node]:
            indegree[nbr] = indegree.get(nbr, 0) + 1
    queue = [n for n, d in indegree.items() if d == 0]
    order: List[Any] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for nbr in graph.get(node, []):
            indegree[nbr] -= 1
            if indegree[nbr] == 0:
                queue.append(nbr)
    return order


# -----------------------------
# Bugfix task set (v9)
# -----------------------------

def generate_v9_bugfix() -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    def add(
        *,
        tier: int,
        topic: str,
        task_id: str,
        prompt: str,
        signature: str,
        starter_code: str,
        tests: str,
    ) -> None:
        tasks.append(
            _make_task(
                task_id=task_id,
                prompt=prompt,
                signature=signature,
                starter_code=starter_code,
                tests=tests,
                category="bugfix",
                topic=topic,
                tier=tier,
                starter_check="fail",
            )
        )
        tier_counts[tier] += 1

    # ---- Tier 1 (20) ----
    between_variants = [
        (1, 3),
        (0, 10),
        (-5, 5),
        (2, 4),
        (10, 20),
    ]
    for idx, (low, high) in enumerate(between_variants, start=1):
        mid = (low + high) // 2
        add(
            tier=1,
            topic="conditionals",
            task_id=f"v9_t1_between_{idx:02d}",
            prompt="Fix this function so it returns True when x is between low and high inclusive.",
            signature="def is_between(x: int, low: int, high: int) -> bool:",
            starter_code=_dedent(
                """
                def is_between(x: int, low: int, high: int) -> bool:
                    return low < x < high
                """
            ),
            tests=_dedent(
                f"""
                def test_is_between_{idx}():
                    assert is_between({low}, {low}, {high}) is True
                    assert is_between({mid}, {low}, {high}) is True
                    assert is_between({high}, {low}, {high}) is True
                    assert is_between({low - 1}, {low}, {high}) is False
                    assert is_between({high + 1}, {low}, {high}) is False
                """
            ),
        )

    safe_div_variants = [
        (10, 2),
        (7, -2),
        (-9, 3),
        (5, 4),
        (0, 5),
    ]
    for idx, (a, b) in enumerate(safe_div_variants, start=1):
        add(
            tier=1,
            topic="math",
            task_id=f"v9_t1_safe_div_{idx:02d}",
            prompt="Fix this function to return None when dividing by zero.",
            signature="def safe_divide(a: float, b: float):",
            starter_code=_dedent(
                """
                def safe_divide(a: float, b: float):
                    if b == 0:
                        return 0
                    return a / b
                """
            ),
            tests=_dedent(
                f"""
                def test_safe_divide_{idx}():
                    assert safe_divide({a}, {b}) == {a / b}
                    assert safe_divide({a}, 0) is None
                """
            ),
        )

    last_char_variants = ["alpha", "Z", "hello", "12345", "mixEd"]
    for idx, s in enumerate(last_char_variants, start=1):
        add(
            tier=1,
            topic="strings",
            task_id=f"v9_t1_last_char_{idx:02d}",
            prompt="Return the last character of s, or '' if s is empty.",
            signature="def last_char(s: str) -> str:",
            starter_code=_dedent(
                """
                def last_char(s: str) -> str:
                    return s[-1]
                """
            ),
            tests=_dedent(
                f"""
                def test_last_char_{idx}():
                    assert last_char({s!r}) == {s[-1]!r}
                    assert last_char("") == ""
                """
            ),
        )

    clamp_variants = [
        (0, 10, 5),
        (-3, 3, 0),
        (1, 4, 2),
        (10, 20, 15),
        (-10, -1, -5),
    ]
    for idx, (low, high, mid) in enumerate(clamp_variants, start=1):
        add(
            tier=1,
            topic="conditionals",
            task_id=f"v9_t1_clamp_{idx:02d}",
            prompt="Clamp x to the inclusive range [low, high].",
            signature="def clamp_value(x: int, low: int, high: int) -> int:",
            starter_code=_dedent(
                """
                def clamp_value(x: int, low: int, high: int) -> int:
                    if x < low:
                        return high
                    if x > high:
                        return low
                    return x
                """
            ),
            tests=_dedent(
                f"""
                def test_clamp_value_{idx}():
                    assert clamp_value({mid}, {low}, {high}) == {mid}
                    assert clamp_value({low - 1}, {low}, {high}) == {low}
                    assert clamp_value({high + 1}, {low}, {high}) == {high}
                """
            ),
        )


    # ---- Tier 2 (50) ----
    count_variants = [
        ([1, 2, 1, 1, 3], 1),
        (["a", "b", "a", "c", "a"], "a"),
        ([5, 5, 5, 2, 5], 5),
        ([0, 1, 0, 2, 0, 3], 0),
        (["x", "y", "z", "y"], "y"),
        ([3, 4, 3, 4, 3, 4], 4),
        (["cat", "dog", "cat"], "cat"),
        ([9, 8, 7, 8, 8], 8),
        (["p", "q", "p", "q", "p"], "p"),
        ([True, False, True, True], True),
    ]
    for idx, (items, target) in enumerate(count_variants, start=1):
        expected = items.count(target)
        add(
            tier=2,
            topic="lists",
            task_id=f"v9_t2_count_occ_{idx:02d}",
            prompt="Return how many times target appears in items.",
            signature="def count_occurrences(items: list, target) -> int:",
            starter_code=_dedent(
                """
                def count_occurrences(items: list, target) -> int:
                    count = 0
                    for item in items:
                        if item == target:
                            count = 1
                    return count
                """
            ),
            tests=_dedent(
                f"""
                def test_count_occurrences_{idx}():
                    assert count_occurrences({items!r}, {target!r}) == {expected}
                """
            ),
        )

    rotate_variants = []
    for i in range(10):
        items = list(range(i, i + 6))
        k = i + 1
        rotate_variants.append((items, k))
    for idx, (items, k) in enumerate(rotate_variants, start=1):
        expected = _rotate_right(items, k)
        add(
            tier=2,
            topic="lists",
            task_id=f"v9_t2_rotate_{idx:02d}",
            prompt="Rotate items to the right by k (k may be larger than the list length).",
            signature="def rotate_right(items: list, k: int) -> list:",
            starter_code=_dedent(
                """
                def rotate_right(items: list, k: int) -> list:
                    if not items:
                        return []
                    return items[-k:] + items[:-k]
                """
            ),
            tests=_dedent(
                f"""
                def test_rotate_right_{idx}():
                    assert rotate_right({items!r}, {k}) == {expected!r}
                """
            ),
        )

    dedupe_variants = [
        [1, 2, 1, 3, 2, 4],
        ["a", "b", "a", "c", "b"],
        [5, 5, 5, 1, 2],
        ["x", "x", "y", "x"],
        [True, False, True, True, False],
        ["cat", "dog", "cat", "bird"],
        [9, 8, 9, 7, 8],
        ["p", "q", "p", "q", "r"],
        [0, 1, 0, 2, 0],
        ["aa", "aa", "bb", "aa"],
    ]
    for idx, items in enumerate(dedupe_variants, start=1):
        expected = _dedupe_preserve(items)
        add(
            tier=2,
            topic="lists",
            task_id=f"v9_t2_dedupe_{idx:02d}",
            prompt="Remove duplicates while preserving original order.",
            signature="def dedupe_preserve_order(items: list) -> list:",
            starter_code=_dedent(
                """
                def dedupe_preserve_order(items: list) -> list:
                    return list(set(items))
                """
            ),
            tests=_dedent(
                f"""
                def test_dedupe_preserve_order_{idx}():
                    assert dedupe_preserve_order({items!r}) == {expected!r}
                """
            ),
        )

    chunk_variants = []
    for i in range(10):
        size = (i % 3) + 2
        items = list(range(1, 9 + i))
        chunk_variants.append((items, size))
    for idx, (items, size) in enumerate(chunk_variants, start=1):
        expected = _chunk_list(items, size)
        add(
            tier=2,
            topic="lists",
            task_id=f"v9_t2_chunk_{idx:02d}",
            prompt="Split items into chunks of size N (final chunk may be shorter).",
            signature="def chunk_list(items: list, size: int) -> list:",
            starter_code=_dedent(
                """
                def chunk_list(items: list, size: int) -> list:
                    chunks = []
                    for i in range(0, len(items) - size, size):
                        chunks.append(items[i:i + size])
                    return chunks
                """
            ),
            tests=_dedent(
                f"""
                def test_chunk_list_{idx}():
                    assert chunk_list({items!r}, {size}) == {expected!r}
                """
            ),
        )

    flatten_variants = [
        [1, [2, 3], 4],
        [[1, 2], 3, [4, 5]],
        ["a", ["b", "c"], "d"],
        [[], [1], 2],
        [[1], [2], [3]],
        [1, 2, [3, 4, 5]],
        ["x", ["y"], ["z"]],
        [[1, 2], [3], 4, [5, 6]],
        [[], "a", ["b"]],
        [[0], 1, 2],
    ]
    for idx, items in enumerate(flatten_variants, start=1):
        expected = _flatten_once(items)
        add(
            tier=2,
            topic="lists",
            task_id=f"v9_t2_flatten_{idx:02d}",
            prompt="Flatten one nesting level of lists.",
            signature="def flatten_once(items: list) -> list:",
            starter_code=_dedent(
                """
                def flatten_once(items: list) -> list:
                    out = []
                    for x in items:
                        if isinstance(x, list):
                            out.append(x)
                        else:
                            out.append(x)
                    return out
                """
            ),
            tests=_dedent(
                f"""
                def test_flatten_once_{idx}():
                    assert flatten_once({items!r}) == {expected!r}
                """
            ),
        )

    # ---- Tier 3 (60) ----
    interval_variants = [
        [(5, 7), (1, 3), (2, 4)],
        [(6, 8), (1, 2), (3, 5)],
        [(1, 4), (0, 0), (5, 7), (2, 3)],
        [(10, 12), (2, 4), (3, 8)],
        [(1, 2), (2, 6), (8, 10), (9, 11)],
        [(3, 5), (1, 2), (6, 7)],
        [(4, 5), (1, 10), (2, 3)],
        [(0, 1), (1, 2), (2, 3)],
        [(7, 9), (2, 6), (1, 3)],
        [(15, 18), (1, 2), (2, 4)],
    ]
    for idx, intervals in enumerate(interval_variants, start=1):
        expected = _merge_intervals(intervals)
        add(
            tier=3,
            topic="intervals",
            task_id=f"v9_t3_merge_intervals_{idx:02d}",
            prompt="Merge overlapping intervals. Input may be unsorted.",
            signature="def merge_intervals(intervals: list) -> list:",
            starter_code=_dedent(
                """
                def merge_intervals(intervals: list) -> list:
                    if not intervals:
                        return []
                    merged = [intervals[0]]
                    for start, end in intervals[1:]:
                        last_start, last_end = merged[-1]
                        if start <= last_end:
                            merged[-1] = (last_start, max(last_end, end))
                        else:
                            merged.append((start, end))
                    return merged
                """
            ),
            tests=_dedent(
                f"""
                def test_merge_intervals_{idx}():
                    assert merge_intervals({intervals!r}) == {expected!r}
                """
            ),
        )

    anagram_variants = [
        ["eat", "tea", "tan", "ate", "nat", "bat"],
        ["listen", "silent", "enlist", "google"],
        ["rat", "tar", "art", "car"],
        ["abc", "bca", "cab", "xyz"],
        ["a", "b", "a"],
        ["dusty", "study", "night", "thing"],
        ["evil", "vile", "veil", "live"],
        ["loop", "polo", "pool", "lopo"],
        ["state", "taste", "tates"],
        ["one", "neo", "eon", "two"],
    ]
    for idx, words in enumerate(anagram_variants, start=1):
        expected = _group_anagrams(words)
        add(
            tier=3,
            topic="strings",
            task_id=f"v9_t3_group_anagrams_{idx:02d}",
            prompt="Group words that are anagrams of each other.",
            signature="def group_anagrams(words: list) -> list:",
            starter_code=_dedent(
                """
                def group_anagrams(words: list) -> list:
                    groups = {}
                    for w in words:
                        key = w
                        groups.setdefault(key, []).append(w)
                    return list(groups.values())
                """
            ),
            tests=_dedent(
                f"""
                def test_group_anagrams_{idx}():
                    result = group_anagrams({words!r})
                    normalized = sorted([sorted(g) for g in result], key=lambda g: g[0])
                    assert normalized == {expected!r}
                """
            ),
        )

    topk_variants = [
        ([1, 1, 1, 2, 2, 3], 2),
        (["a", "b", "a", "c", "b", "a"], 1),
        ([4, 4, 5, 6, 6, 6], 2),
        (["x", "y", "x", "z", "z", "z"], 2),
        ([9, 9, 8, 7, 7, 7, 8], 1),
        (["cat", "dog", "cat", "bird", "dog", "cat"], 2),
        ([10, 11, 10, 12, 12, 12, 11], 3),
        (["p", "q", "p", "r", "p", "q"], 2),
        ([0, 0, 0, 1, 2, 2], 2),
        (["aa", "bb", "aa", "cc", "bb", "bb"], 2),
    ]
    for idx, (items, k) in enumerate(topk_variants, start=1):
        expected = _top_k_frequent(items, k)
        add(
            tier=3,
            topic="frequency",
            task_id=f"v9_t3_topk_{idx:02d}",
            prompt="Return the k most frequent items.",
            signature="def top_k_frequent(items: list, k: int) -> list:",
            starter_code=_dedent(
                """
                def top_k_frequent(items: list, k: int) -> list:
                    freq = {}
                    for x in items:
                        freq[x] = freq.get(x, 0) + 1
                    ordered = sorted(freq.items(), key=lambda kv: kv[1])
                    return [x for x, _ in ordered[:k]]
                """
            ),
            tests=_dedent(
                f"""
                def test_top_k_frequent_{idx}():
                    assert top_k_frequent({items!r}, {k}) == {expected!r}
                """
            ),
        )

    bracket_variants = [
        "([]){}",
        "([)]",
        "{[()]}",
        "((()))",
        "([{}])",
        "(]",
        "(()",
        "())",
        "{[}]",
        "[]{}()",
    ]
    for idx, s in enumerate(bracket_variants, start=1):
        expected = _is_balanced(s)
        add(
            tier=3,
            topic="stack",
            task_id=f"v9_t3_brackets_{idx:02d}",
            prompt="Return True if the brackets are balanced for (), [], {}.",
            signature="def is_balanced(s: str) -> bool:",
            starter_code=_dedent(
                """
                def is_balanced(s: str) -> bool:
                    pairs = {')': '(', ']': '[', '}': '{'}
                    stack = []
                    for ch in s:
                        if ch in pairs.values():
                            stack.append(ch)
                        elif ch in pairs:
                            if not stack:
                                return False
                            stack.pop()
                    return not stack
                """
            ),
            tests=_dedent(
                f"""
                def test_is_balanced_{idx}():
                    assert is_balanced({s!r}) is {expected}
                """
            ),
        )

    path_variants = [
        "/a/b/../c",
        "/a/./b/./c",
        "/../a/b",
        "/a/b/c/..",
        "/a//b///c",
        "/a/b/../../d",
        "/x/y/z/../..",
        "/a/../b/../c",
        "/a/b/././c",
        "/a/b/c",
    ]
    for idx, path in enumerate(path_variants, start=1):
        expected = _normalize_path(path)
        add(
            tier=3,
            topic="parsing",
            task_id=f"v9_t3_norm_path_{idx:02d}",
            prompt="Normalize a Unix-style path by resolving '.' and '..'.",
            signature="def normalize_path(path: str) -> str:",
            starter_code=_dedent(
                """
                def normalize_path(path: str) -> str:
                    parts = []
                    for part in path.split('/'):
                        if part == '' or part == '.':
                            continue
                        if part == '..':
                            continue
                        parts.append(part)
                    return '/' + '/'.join(parts)
                """
            ),
            tests=_dedent(
                f"""
                def test_normalize_path_{idx}():
                    assert normalize_path({path!r}) == {expected!r}
                """
            ),
        )

    rle_variants = [
        "aaabbc",
        "abcd",
        "aabbaa",
        "zzzz",
        "a",
        "abccccc",
        "wwwwx",
        "ppqqq",
        "mmmmnn",
        "yyyz",
    ]
    for idx, s in enumerate(rle_variants, start=1):
        expected = _rle_encode(s)
        add(
            tier=3,
            topic="strings",
            task_id=f"v9_t3_rle_{idx:02d}",
            prompt="Run-length encode the string (e.g., 'aaab' -> 'a3b1').",
            signature="def rle_encode(s: str) -> str:",
            starter_code=_dedent(
                """
                def rle_encode(s: str) -> str:
                    if not s:
                        return ""
                    out = []
                    count = 1
                    for i in range(1, len(s)):
                        if s[i] == s[i - 1]:
                            count += 1
                        else:
                            out.append(f"{s[i - 1]}{count}")
                            count = 1
                    return "".join(out)
                """
            ),
            tests=_dedent(
                f"""
                def test_rle_encode_{idx}():
                    assert rle_encode({s!r}) == {expected!r}
                """
            ),
        )

    # ---- Tier 4 (50) ----
    lru_variants = [
        (2, [("a", 1), ("b", 2), ("c", 3)]),
        (2, [("x", 1), ("y", 2), ("z", 3)]),
        (3, [("k1", 1), ("k2", 2), ("k3", 3), ("k4", 4)]),
        (2, [("p", 7), ("q", 8), ("r", 9)]),
        (3, [("u", 1), ("v", 2), ("w", 3), ("t", 4)]),
        (2, [("m", 1), ("n", 2), ("o", 3)]),
        (3, [("aa", 1), ("bb", 2), ("cc", 3), ("dd", 4)]),
        (2, [("s", 1), ("t", 2), ("u", 3)]),
        (3, [("h", 5), ("i", 6), ("j", 7), ("k", 8)]),
        (2, [("r1", 1), ("r2", 2), ("r3", 3)]),
    ]
    for idx, (capacity, ops) in enumerate(lru_variants, start=1):
        key1, _ = ops[0]
        key2, _ = ops[1]
        new_key, _ = ops[2]
        add(
            tier=4,
            topic="caching",
            task_id=f"v9_t4_lru_{idx:02d}",
            prompt="Fix this LRU cache so recent gets update the eviction order.",
            signature="class LRUCache:",
            starter_code=_dedent(
                """
                from collections import OrderedDict

                class LRUCache:
                    def __init__(self, capacity: int):
                        self.capacity = capacity
                        self.data = OrderedDict()

                    def get(self, key):
                        if key not in self.data:
                            return None
                        return self.data[key]

                    def put(self, key, value):
                        if key in self.data:
                            self.data[key] = value
                            return
                        self.data[key] = value
                        if len(self.data) > self.capacity:
                            self.data.popitem(last=False)
                """
            ),
            tests=_dedent(
                f"""
                def test_lru_cache_{idx}():
                    cache = LRUCache({capacity})
                    for k, v in {ops!r}[:2]:
                        cache.put(k, v)
                    assert cache.get({key1!r}) == {ops[0][1]!r}
                    cache.put({new_key!r}, {ops[2][1]!r})
                    assert cache.get({key2!r}) is None
                    assert cache.get({key1!r}) == {ops[0][1]!r}
                """
            ),
        )

    trie_variants = [
        ("cat", "car"),
        ("apple", "app"),
        ("dog", "do"),
        ("tree", "tr"),
        ("house", "hou"),
        ("train", "tra"),
        ("plane", "pla"),
        ("note", "no"),
        ("ring", "ri"),
        ("road", "ro"),
    ]
    for idx, (word, prefix) in enumerate(trie_variants, start=1):
        add(
            tier=4,
            topic="trie",
            task_id=f"v9_t4_trie_{idx:02d}",
            prompt="Fix this Trie so search returns True only for whole words.",
            signature="class Trie:",
            starter_code=_dedent(
                """
                class TrieNode:
                    def __init__(self):
                        self.children = {}
                        self.is_end = False

                class Trie:
                    def __init__(self):
                        self.root = TrieNode()

                    def insert(self, word: str) -> None:
                        node = self.root
                        for ch in word:
                            node = node.children.setdefault(ch, TrieNode())

                    def search(self, word: str) -> bool:
                        node = self.root
                        for ch in word:
                            if ch not in node.children:
                                return False
                            node = node.children[ch]
                        return True
                """
            ),
            tests=_dedent(
                f"""
                def test_trie_{idx}():
                    trie = Trie()
                    trie.insert({word!r})
                    assert trie.search({word!r}) is True
                    assert trie.search({prefix!r}) is False
                """
            ),
        )

    uf_variants = [
        ([1, 2, 3], [(1, 2), (2, 3)], (1, 3)),
        ([4, 5, 6], [(4, 5), (5, 6)], (4, 6)),
        ([10, 11, 12], [(10, 11), (11, 12)], (10, 12)),
        ([7, 8, 9], [(7, 8), (8, 9)], (7, 9)),
        ([20, 21, 22], [(20, 21), (21, 22)], (20, 22)),
        ([30, 31, 32], [(30, 31), (31, 32)], (30, 32)),
        ([40, 41, 42], [(40, 41), (41, 42)], (40, 42)),
        ([50, 51, 52], [(50, 51), (51, 52)], (50, 52)),
        ([60, 61, 62], [(60, 61), (61, 62)], (60, 62)),
        ([70, 71, 72], [(70, 71), (71, 72)], (70, 72)),
    ]
    for idx, (nodes, unions, query) in enumerate(uf_variants, start=1):
        a, b = query
        add(
            tier=4,
            topic="disjoint_set",
            task_id=f"v9_t4_union_find_{idx:02d}",
            prompt="Fix UnionFind so find returns the true root for chained unions.",
            signature="class UnionFind:",
            starter_code=_dedent(
                """
                class UnionFind:
                    def __init__(self, items):
                        self.parent = {x: x for x in items}

                    def find(self, x):
                        if self.parent[x] == x:
                            return x
                        return self.parent[x]

                    def union(self, a, b):
                        ra = self.find(a)
                        rb = self.find(b)
                        if ra != rb:
                            self.parent[rb] = ra
                """
            ),
            tests=_dedent(
                f"""
                def test_union_find_{idx}():
                    uf = UnionFind({nodes!r})
                    for a, b in {unions!r}:
                        uf.union(a, b)
                    assert uf.find({a}) == uf.find({b})
                """
            ),
        )

    graph_variants = []
    for i in range(10):
        start = f"S{i}"
        mid_short = f"M{i}"
        mid_long = f"L{i}"
        long2 = f"L{i}b"
        end = f"E{i}"
        graph = {
            start: [mid_short, mid_long],
            mid_short: [end],
            mid_long: [long2],
            long2: [end],
            end: [],
        }
        graph_variants.append((graph, start, end))
    for idx, (graph, start, end) in enumerate(graph_variants, start=1):
        expected = _bfs_shortest_path_len(graph, start, end)
        add(
            tier=4,
            topic="graph",
            task_id=f"v9_t4_shortest_path_{idx:02d}",
            prompt="Return the length of the shortest path in an unweighted graph.",
            signature="def shortest_path_len(graph: dict, start, end) -> int:",
            starter_code=_dedent(
                """
                def shortest_path_len(graph: dict, start, end) -> int:
                    stack = [(start, 0)]
                    visited = set()
                    while stack:
                        node, dist = stack.pop()
                        if node == end:
                            return dist
                        if node in visited:
                            continue
                        visited.add(node)
                        for nbr in graph.get(node, []):
                            stack.append((nbr, dist + 1))
                    return -1
                """
            ),
            tests=_dedent(
                f"""
                def test_shortest_path_len_{idx}():
                    graph = {graph!r}
                    assert shortest_path_len(graph, {start!r}, {end!r}) == {expected}
                """
            ),
        )

    rate_variants = [
        (5, [0, 5, 10]),
        (3, [0, 3, 6]),
        (7, [0, 7, 14]),
        (2, [0, 2, 4]),
        (4, [0, 4, 8]),
        (6, [0, 6, 12]),
        (8, [0, 8, 16]),
        (9, [0, 9, 18]),
        (10, [0, 10, 20]),
        (12, [0, 12, 24]),
    ]
    for idx, (interval, times) in enumerate(rate_variants, start=1):
        add(
            tier=4,
            topic="rate_limiting",
            task_id=f"v9_t4_rate_limit_{idx:02d}",
            prompt="Allow calls only if at least interval_ms has elapsed since the last allowed call.",
            signature="class RateLimiter:",
            starter_code=_dedent(
                """
                class RateLimiter:
                    def __init__(self, interval_ms: int):
                        self.interval_ms = interval_ms
                        self.last_ms = None

                    def allow(self, now_ms: int) -> bool:
                        if self.last_ms is None:
                            self.last_ms = now_ms
                            return True
                        if now_ms - self.last_ms > self.interval_ms:
                            self.last_ms = now_ms
                            return True
                        return False
                """
            ),
            tests=_dedent(
                f"""
                def test_rate_limiter_{idx}():
                    rl = RateLimiter({interval})
                    assert rl.allow({times[0]}) is True
                    assert rl.allow({times[1]}) is True
                    assert rl.allow({times[1]}) is False
                """
            ),
        )

    # ---- Tier 5 (20) ----
    descriptor_variants = ["balance", "score", "level", "quota", "limit"]
    for idx, attr in enumerate(descriptor_variants, start=1):
        add(
            tier=5,
            topic="descriptors",
            task_id=f"v9_t5_descriptor_{idx:02d}",
            prompt="Fix this descriptor so values are stored per instance (not shared).",
            signature="class PositiveInt:",
            starter_code=_dedent(
                f"""
                class PositiveInt:
                    def __init__(self, name: str):
                        self.name = name
                        self.value = None

                    def __get__(self, obj, objtype=None):
                        return self.value

                    def __set__(self, obj, value: int):
                        if value < 0:
                            raise ValueError("must be non-negative")
                        self.value = value

                class Account:
                    {attr} = PositiveInt("{attr}")

                    def __init__(self, {attr}: int):
                        self.{attr} = {attr}
                """
            ),
            tests=_dedent(
                f"""
                def test_descriptor_{idx}():
                    a = Account(1)
                    b = Account(2)
                    assert a.{attr} == 1
                    assert b.{attr} == 2
                """
            ),
        )

    toggle_variants = ["enabled", "active", "flag", "visible", "ready"]
    for idx, attr in enumerate(toggle_variants, start=1):
        add(
            tier=5,
            topic="context_manager",
            task_id=f"v9_t5_context_{idx:02d}",
            prompt="Fix this context manager so it restores the previous flag value on exit.",
            signature="class ToggleFlag:",
            starter_code=_dedent(
                f"""
                class ToggleFlag:
                    def __init__(self, obj, attr: str):
                        self.obj = obj
                        self.attr = attr
                        self.prev = None

                    def __enter__(self):
                        self.prev = getattr(self.obj, self.attr)
                        setattr(self.obj, self.attr, True)
                        return self

                    def __exit__(self, exc_type, exc, tb):
                        pass

                class Box:
                    def __init__(self):
                        self.{attr} = False
                """
            ),
            tests=_dedent(
                f"""
                def test_toggle_flag_{idx}():
                    box = Box()
                    assert box.{attr} is False
                    with ToggleFlag(box, "{attr}"):
                        assert box.{attr} is True
                    assert box.{attr} is False
                """
            ),
        )

    iter_variants = [
        (0, 3),
        (1, 5),
        (2, 6),
        (5, 9),
        (10, 13),
    ]
    for idx, (start, end) in enumerate(iter_variants, start=1):
        add(
            tier=5,
            topic="iterators",
            task_id=f"v9_t5_iter_{idx:02d}",
            prompt="Fix this iterator so it yields values from start up to end (exclusive).",
            signature="class RangeIter:",
            starter_code=_dedent(
                """
                class RangeIter:
                    def __init__(self, start: int, end: int):
                        self.cur = start
                        self.end = end

                    def __iter__(self):
                        return self

                    def __next__(self):
                        if self.cur > self.end:
                            raise StopIteration
                        val = self.cur
                        self.cur += 1
                        return val
                """
            ),
            tests=_dedent(
                f"""
                def test_range_iter_{idx}():
                    it = RangeIter({start}, {end})
                    assert list(it) == list(range({start}, {end}))
                """
            ),
        )

    pickle_variants = [
        ("alice", "secret1"),
        ("bob", "secret2"),
        ("carol", "secret3"),
        ("dave", "secret4"),
        ("eve", "secret5"),
    ]
    for idx, (user, secret) in enumerate(pickle_variants, start=1):
        add(
            tier=5,
            topic="pickle_protocol",
            task_id=f"v9_t5_pickle_{idx:02d}",
            prompt="Fix __getstate__ so sensitive data is not stored in the pickle.",
            signature="class SecureNote:",
            starter_code=_dedent(
                """
                class SecureNote:
                    def __init__(self, user: str, secret: str):
                        self.user = user
                        self.secret = secret

                    def __getstate__(self):
                        return self.__dict__
                """
            ),
            tests=_dedent(
                f"""
                def test_secure_note_{idx}():
                    import pickle
                    note = SecureNote({user!r}, {secret!r})
                    blob = pickle.dumps(note)
                    restored = pickle.loads(blob)
                    assert restored.user == {user!r}
                    assert restored.secret is None
                """
            ),
        )

    assert sum(tier_counts.values()) == 200, f"Expected 200 tasks, got {sum(tier_counts.values())}"
    assert tier_counts == {1: 20, 2: 50, 3: 60, 4: 50, 5: 20}, f"Tier counts off: {tier_counts}"
    return tasks


# -----------------------------
# Oscillation task set (v10)
# -----------------------------

OSC_APPROACHES: Dict[str, List[str]] = {
    "backpressure": ["semaphore", "counter", "deque", "circular_buffer"],
    "barrier": ["threading_barrier", "condition_notify", "event_counter", "generation_counter"],
    "bst": ["dict_nodes", "class_nodes", "iterative", "recursive"],
    "caching": ["ordereddict", "functools", "lock_protected", "thread_local"],
    "circuit_breaker": ["state_machine", "counter_threshold", "time_based", "sliding_window"],
    "connection_pool": ["queue_blocking", "semaphore", "list_with_lock"],
    "graph": ["adjacency_list", "adjacency_matrix", "edge_set", "hybrid"],
    "heap": ["heapq", "manual_sift", "sorted_list"],
    "interval_merge": ["sort_and_scan", "sweep_line", "reduce"],
    "sorting": ["quicksort", "mergesort", "builtin", "iterative"],
    "search": ["binary_iterative", "binary_recursive", "bisect_module", "linear"],
    "read_write_lock": ["condition_based", "single_lock_counter", "two_locks", "rlock_counter"],
    "request_coalescing": ["dict_of_futures", "dict_of_events", "lock_per_key"],
    "retry": ["fixed_delay", "exponential_backoff", "jitter"],
    "ring_buffer": ["deque", "modulo_index", "list_rotate"],
    "semaphore": ["threading_semaphore", "counter_condition", "queue_tokens"],
    "skip_list": ["linked_levels", "array_levels", "sorted_list_baseline"],
    "string_match": ["kmp", "rabin_karp", "naive", "builtin"],
    "thread_pool": ["futures", "queue_workers", "dynamic_threads"],
    "timer": ["threading_timer", "thread_sleep", "event_flag", "loop_thread"],
    "topological_sort": ["dfs_postorder", "kahns_bfs", "recursive"],
    "trie": ["dict_children", "array_children", "defaultdict", "nested_dict"],
    "priority_queue": ["heapq_module", "sorted_list", "manual_heap"],
}


OSC_PREFIX = {
    "backpressure": "osc_backpressure",
    "barrier": "osc_barrier",
    "bst": "osc_tree",
    "caching": "osc_caching",
    "circuit_breaker": "osc_circuit_breaker",
    "connection_pool": "osc_connection_pool",
    "graph": "osc_graph",
    "heap": "osc_heap",
    "interval_merge": "osc_interval_merge",
    "sorting": "osc_sort",
    "search": "osc_search",
    "priority_queue": "osc_pq",
    "read_write_lock": "osc_rwlock",
    "request_coalescing": "osc_request_coalescing",
    "retry": "osc_retry",
    "ring_buffer": "osc_ring",
    "semaphore": "osc_semaphore",
    "skip_list": "osc_skip",
    "string_match": "osc_string_match",
    "thread_pool": "osc_thread_pool",
    "timer": "osc_timer",
    "topological_sort": "osc_topological_sort",
    "trie": "osc_trie",
}


def generate_v10_oscillation() -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []

    def add(
        *,
        topic: str,
        task_id: str,
        prompt: str,
        signature: str,
        starter_code: str,
        tests: str,
    ) -> None:
        tasks.append(
            _make_task(
                task_id=task_id,
                prompt=prompt,
                signature=signature,
                starter_code=starter_code,
                tests=tests,
                category="bugfix",
                topic=topic,
                tier="oscillation",
                starter_check="fail",
                approaches=OSC_APPROACHES[topic],
            )
        )

    # backpressure (10)
    for idx in range(1, 11):
        capacity = 2 + (idx % 5)
        values = [idx * 10 + 1, idx * 10 + 2, idx * 10 + 3]
        allow_third = capacity >= 3
        task_id = f"v10_{OSC_PREFIX['backpressure']}_{idx:03d}"
        add(
            topic="backpressure",
            task_id=task_id,
            prompt="Fix this bounded queue so it enforces capacity and preserves FIFO order.",
            signature="class BackpressureQueue:",
            starter_code=_dedent(
                """
                class BackpressureQueue:
                    def __init__(self, capacity: int):
                        self.capacity = capacity
                        self.items = []
                        self.count = 0

                    def offer(self, item) -> bool:
                        if len(self.items) > self.capacity:
                            return False
                        self.items.append(item)
                        self.count += 1
                        return True

                    def poll(self):
                        if not self.items:
                            return None
                        self.count -= 1
                        return self.items.pop(0)
                """
            ),
            tests=_dedent(
                f"""
                def test_backpressure_queue_{idx}():
                    q = BackpressureQueue({capacity})
                    assert q.offer({values[0]}) is True
                    assert q.offer({values[1]}) is True
                    assert q.offer({values[2]}) is {allow_third}
                    assert q.poll() == {values[0]}
                """
            ),
        )

    # barrier (10)
    for idx in range(1, 11):
        parties = 2 + (idx % 4)
        task_id = f"v10_{OSC_PREFIX['barrier']}_{idx:03d}"
        add(
            topic="barrier",
            task_id=task_id,
            prompt="Fix this barrier so it releases exactly when the required number of parties arrive.",
            signature="class SimpleBarrier:",
            starter_code=_dedent(
                """
                class SimpleBarrier:
                    def __init__(self, parties: int):
                        self.parties = parties
                        self.count = 0

                    def wait(self) -> bool:
                        self.count += 1
                        if self.count > self.parties:
                            self.count = 0
                            return True
                        return False
                """
            ),
            tests=_dedent(
                f"""
                def test_simple_barrier_{idx}():
                    b = SimpleBarrier({parties})
                    for _ in range({parties} - 1):
                        assert b.wait() is False
                    assert b.wait() is True
                """
            ),
        )

    # bst (10)
    for idx in range(1, 11):
        values = [idx * 3, idx * 3 - 1, idx * 3 + 1]
        missing = idx * 3 + 2
        task_id = f"v10_{OSC_PREFIX['bst']}_{idx:03d}"
        add(
            topic="bst",
            task_id=task_id,
            prompt="Fix BST insertion so searches find inserted values.",
            signature="class BST:",
            starter_code=_dedent(
                """
                class Node:
                    def __init__(self, value: int):
                        self.value = value
                        self.left = None
                        self.right = None

                class BST:
                    def __init__(self):
                        self.root = None

                    def insert(self, value: int):
                        if self.root is None:
                            self.root = Node(value)
                            return
                        node = self.root
                        while True:
                            if value < node.value:
                                if node.right is None:
                                    node.right = Node(value)
                                    return
                                node = node.right
                            elif value > node.value:
                                if node.left is None:
                                    node.left = Node(value)
                                    return
                                node = node.left
                            else:
                                return

                    def contains(self, value: int) -> bool:
                        node = self.root
                        while node:
                            if value == node.value:
                                return True
                            if value < node.value:
                                node = node.left
                            else:
                                node = node.right
                        return False
                """
            ),
            tests=_dedent(
                f"""
                def test_bst_{idx}():
                    tree = BST()
                    for v in {values!r}:
                        tree.insert(v)
                    for v in {values!r}:
                        assert tree.contains(v) is True
                    assert tree.contains({missing}) is False
                """
            ),
        )

    # caching (10)
    for idx in range(1, 11):
        capacity = 2 + (idx % 3)
        key1 = f"k{idx}a"
        key2 = f"k{idx}b"
        key3 = f"k{idx}c"
        task_id = f"v10_{OSC_PREFIX['caching']}_{idx:03d}"
        add(
            topic="caching",
            task_id=task_id,
            prompt="Fix this cache so it evicts the oldest entry when capacity is exceeded (FIFO).",
            signature="class FIFOCache:",
            starter_code=_dedent(
                """
                class FIFOCache:
                    def __init__(self, capacity: int):
                        self.capacity = capacity
                        self.data = {}
                        self.order = []

                    def get(self, key):
                        return self.data.get(key)

                    def set(self, key, value):
                        if key not in self.data:
                            self.order.append(key)
                        self.data[key] = value
                        if len(self.order) > self.capacity:
                            old = self.order.pop()
                            self.data.pop(old, None)
                """
            ),
            tests=_dedent(
                f"""
                def test_fifo_cache_{idx}():
                    cache = FIFOCache({capacity})
                    cache.set({key1!r}, 1)
                    cache.set({key2!r}, 2)
                    cache.set({key3!r}, 3)
                    assert cache.get({key1!r}) is None
                    assert cache.get({key2!r}) == 2
                """
            ),
        )

    # circuit_breaker (10)
    for idx in range(1, 11):
        threshold = 2 + (idx % 3)
        task_id = f"v10_{OSC_PREFIX['circuit_breaker']}_{idx:03d}"
        add(
            topic="circuit_breaker",
            task_id=task_id,
            prompt="Fix this circuit breaker so it opens when failures reach the threshold.",
            signature="class CircuitBreaker:",
            starter_code=_dedent(
                """
                class CircuitBreaker:
                    def __init__(self, threshold: int):
                        self.threshold = threshold
                        self.failures = 0
                        self.state = "CLOSED"

                    def record_failure(self):
                        self.failures += 1
                        if self.failures > self.threshold:
                            self.state = "OPEN"

                    def record_success(self):
                        self.failures = 0
                        self.state = "CLOSED"

                    def allow_request(self) -> bool:
                        return self.state != "OPEN"
                """
            ),
            tests=_dedent(
                f"""
                def test_circuit_breaker_{idx}():
                    cb = CircuitBreaker({threshold})
                    for _ in range({threshold}):
                        cb.record_failure()
                    assert cb.allow_request() is False
                """
            ),
        )

    # connection_pool (10)
    for idx in range(1, 11):
        size = 2 + (idx % 4)
        task_id = f"v10_{OSC_PREFIX['connection_pool']}_{idx:03d}"
        add(
            topic="connection_pool",
            task_id=task_id,
            prompt="Fix the connection pool so released connections are reusable.",
            signature="class ConnectionPool:",
            starter_code=_dedent(
                """
                class ConnectionPool:
                    def __init__(self, max_size: int):
                        self.max_size = max_size
                        self.pool = list(range(max_size))

                    def acquire(self):
                        if not self.pool:
                            return None
                        return self.pool.pop(0)

                    def release(self, conn):
                        if len(self.pool) > self.max_size:
                            self.pool.append(conn)
                """
            ),
            tests=_dedent(
                f"""
                def test_connection_pool_{idx}():
                    pool = ConnectionPool({size})
                    c1 = pool.acquire()
                    assert c1 is not None
                    pool.release(c1)
                    c2 = pool.acquire()
                    assert c2 == c1
                """
            ),
        )

    # graph (10)
    for idx in range(1, 11):
        start = f"A{idx}"
        short = f"B{idx}"
        long1 = f"C{idx}"
        long2 = f"D{idx}"
        end = f"E{idx}"
        graph = {
            start: [short, long1],
            short: [end],
            long1: [long2],
            long2: [end],
            end: [],
        }
        expected = _bfs_shortest_path_len(graph, start, end)
        task_id = f"v10_{OSC_PREFIX['graph']}_{idx:03d}"
        add(
            topic="graph",
            task_id=task_id,
            prompt="Return the shortest path length in an unweighted graph.",
            signature="def shortest_path_length(graph: dict, start, end) -> int:",
            starter_code=_dedent(
                """
                def shortest_path_length(graph: dict, start, end) -> int:
                    stack = [(start, 0)]
                    visited = set()
                    while stack:
                        node, dist = stack.pop()
                        if node == end:
                            return dist
                        if node in visited:
                            continue
                        visited.add(node)
                        for nbr in graph.get(node, []):
                            stack.append((nbr, dist + 1))
                    return -1
                """
            ),
            tests=_dedent(
                f"""
                def test_shortest_path_length_{idx}():
                    graph = {graph!r}
                    assert shortest_path_length(graph, {start!r}, {end!r}) == {expected}
                """
            ),
        )

    # heap (10)
    for idx in range(1, 11):
        values = [idx * 3, idx * 3 + 2, idx * 3 + 1]
        task_id = f"v10_{OSC_PREFIX['heap']}_{idx:03d}"
        add(
            topic="heap",
            task_id=task_id,
            prompt="Fix this min-heap so pop returns the smallest element.",
            signature="class MinHeap:",
            starter_code=_dedent(
                """
                import heapq

                class MinHeap:
                    def __init__(self):
                        self.data = []

                    def push(self, value):
                        heapq.heappush(self.data, value)

                    def pop(self):
                        if not self.data:
                            return None
                        return self.data.pop(0)
                """
            ),
            tests=_dedent(
                f"""
                def test_min_heap_{idx}():
                    h = MinHeap()
                    for v in {values!r}:
                        h.push(v)
                    assert h.pop() == {min(values)}
                """
            ),
        )

    # interval_merge (10)
    for idx in range(1, 11):
        intervals = [(idx + 3, idx + 5), (idx, idx + 1), (idx + 1, idx + 4)]
        expected = _merge_intervals(intervals)
        task_id = f"v10_{OSC_PREFIX['interval_merge']}_{idx:03d}"
        add(
            topic="interval_merge",
            task_id=task_id,
            prompt="Merge overlapping intervals from unsorted input.",
            signature="def merge_intervals(intervals: list) -> list:",
            starter_code=_dedent(
                """
                def merge_intervals(intervals: list) -> list:
                    if not intervals:
                        return []
                    merged = [intervals[0]]
                    for start, end in intervals[1:]:
                        last_start, last_end = merged[-1]
                        if start <= last_end:
                            merged[-1] = (last_start, max(last_end, end))
                        else:
                            merged.append((start, end))
                    return merged
                """
            ),
            tests=_dedent(
                f"""
                def test_merge_intervals_{idx}():
                    assert merge_intervals({intervals!r}) == {expected!r}
                """
            ),
        )

    # read_write_lock (10)
    for idx in range(1, 11):
        task_id = f"v10_{OSC_PREFIX['read_write_lock']}_{idx:03d}"
        add(
            topic="read_write_lock",
            task_id=task_id,
            prompt="Fix this read/write lock so writers are blocked when readers are active.",
            signature="class ReadWriteLock:",
            starter_code=_dedent(
                """
                from threading import Lock

                class ReadWriteLock:
                    def __init__(self):
                        self.readers = 0
                        self.writer = False
                        self.lock = Lock()

                    def acquire_read(self) -> bool:
                        if self.writer:
                            return False
                        self.readers += 1
                        return True

                    def release_read(self) -> None:
                        if self.readers > 0:
                            self.readers -= 1

                    def acquire_write(self) -> bool:
                        if self.writer:
                            return False
                        self.writer = True
                        return True

                    def release_write(self) -> None:
                        self.writer = False
                """
            ),
            tests=_dedent(
                f"""
                def test_read_write_lock_{idx}():
                    rw = ReadWriteLock()
                    assert rw.acquire_read() is True
                    assert rw.acquire_write() is False
                    rw.release_read()
                    assert rw.acquire_write() is True
                """
            ),
        )

    # request_coalescing (10)
    for idx in range(1, 11):
        key = f"key{idx}"
        task_id = f"v10_{OSC_PREFIX['request_coalescing']}_{idx:03d}"
        add(
            topic="request_coalescing",
            task_id=task_id,
            prompt="Fix request coalescing so repeated calls reuse the same in-flight result.",
            signature="class RequestCoalescer:",
            starter_code=_dedent(
                """
                class RequestCoalescer:
                    def __init__(self):
                        self.inflight = {}

                    def get_or_start(self, key, factory):
                        if key in self.inflight:
                            return self.inflight[key]
                        value = factory()
                        return value
                """
            ),
            tests=_dedent(
                f"""
                def test_request_coalescer_{idx}():
                    calls = []
                    def factory():
                        calls.append(1)
                        return "value"
                    c = RequestCoalescer()
                    assert c.get_or_start({key!r}, factory) == "value"
                    assert c.get_or_start({key!r}, factory) == "value"
                    assert len(calls) == 1
                """
            ),
        )

    # retry (10)
    for idx in range(1, 11):
        failures = (idx % 3) + 1
        retries = failures
        task_id = f"v10_{OSC_PREFIX['retry']}_{idx:03d}"
        add(
            topic="retry",
            task_id=task_id,
            prompt="Fix retry so it makes retries + 1 total attempts.",
            signature="def retry_call(fn, retries: int):",
            starter_code=_dedent(
                """
                def retry_call(fn, retries: int):
                    last_err = None
                    for _ in range(retries):
                        try:
                            return fn()
                        except Exception as e:
                            last_err = e
                    if last_err:
                        raise last_err
                """
            ),
            tests=_dedent(
                f"""
                def test_retry_call_{idx}():
                    attempts = {{"count": 0}}
                    def flaky():
                        attempts["count"] += 1
                        if attempts["count"] <= {failures}:
                            raise ValueError("fail")
                        return "ok"
                    assert retry_call(flaky, {retries}) == "ok"
                """
            ),
        )

    # ring_buffer (10)
    for idx in range(1, 11):
        capacity = 3 + (idx % 3)
        task_id = f"v10_{OSC_PREFIX['ring_buffer']}_{idx:03d}"
        add(
            topic="ring_buffer",
            task_id=task_id,
            prompt="Fix this ring buffer so it overwrites the oldest item when full.",
            signature="class RingBuffer:",
            starter_code=_dedent(
                """
                class RingBuffer:
                    def __init__(self, capacity: int):
                        self.capacity = capacity
                        self.data = [None] * capacity
                        self.head = 0
                        self.size = 0

                    def append(self, item) -> None:
                        idx = (self.head + self.size) % self.capacity
                        self.data[idx] = item
                        if self.size < self.capacity:
                            self.size += 1

                    def pop(self):
                        if self.size == 0:
                            return None
                        item = self.data[self.head]
                        self.head = (self.head + 1) % self.capacity
                        self.size -= 1
                        return item
                """
            ),
            tests=_dedent(
                f"""
                def test_ring_buffer_{idx}():
                    rb = RingBuffer({capacity})
                    rb.append(1)
                    rb.append(2)
                    rb.append(3)
                    rb.append(4)
                    assert rb.pop() == 2
                """
            ),
        )

    # semaphore (10)
    for idx in range(1, 11):
        max_permits = 2 + (idx % 4)
        task_id = f"v10_{OSC_PREFIX['semaphore']}_{idx:03d}"
        add(
            topic="semaphore",
            task_id=task_id,
            prompt="Fix semaphore release so permits never exceed the maximum.",
            signature="class SimpleSemaphore:",
            starter_code=_dedent(
                """
                class SimpleSemaphore:
                    def __init__(self, permits: int):
                        self.max_permits = permits
                        self.permits = permits

                    def acquire(self) -> bool:
                        if self.permits <= 0:
                            return False
                        self.permits -= 1
                        return True

                    def release(self) -> None:
                        self.permits += 1
                """
            ),
            tests=_dedent(
                f"""
                def test_simple_semaphore_{idx}():
                    s = SimpleSemaphore({max_permits})
                    for _ in range({max_permits}):
                        assert s.acquire() is True
                    assert s.acquire() is False
                    s.release()
                    s.release()
                    count = 0
                    while s.acquire():
                        count += 1
                    assert count == {max_permits}
                """
            ),
        )

    # skip_list (10)
    for idx in range(1, 11):
        values = [idx, idx + 2, idx + 4]
        search_value = values[1]
        task_id = f"v10_{OSC_PREFIX['skip_list']}_{idx:03d}"
        add(
            topic="skip_list",
            task_id=task_id,
            prompt="Fix this simplified skip list so search checks lower levels.",
            signature="class SkipList:",
            starter_code=_dedent(
                """
                class SkipList:
                    def __init__(self):
                        self.level0 = []
                        self.level1 = []

                    def insert(self, value: int):
                        if value not in self.level0:
                            self.level0.append(value)
                            self.level0.sort()
                            if len(self.level0) % 2 == 0:
                                self.level1.append(value)
                                self.level1.sort()

                    def search(self, value: int) -> bool:
                        return value in self.level1
                """
            ),
            tests=_dedent(
                f"""
                def test_skip_list_{idx}():
                    sl = SkipList()
                    for v in {values!r}:
                        sl.insert(v)
                    assert sl.search({search_value}) is True
                """
            ),
        )

    # string_match (10)
    for idx in range(1, 11):
        text = f"abcde{idx}xyz"
        pattern = "xyz"
        missing = f"nope{idx}"
        task_id = f"v10_{OSC_PREFIX['string_match']}_{idx:03d}"
        add(
            topic="string_match",
            task_id=task_id,
            prompt="Find the first index of pattern in text or -1 if not found.",
            signature="def find_substring(text: str, pattern: str) -> int:",
            starter_code=_dedent(
                """
                def find_substring(text: str, pattern: str) -> int:
                    idx = text.find(pattern)
                    if idx == -1:
                        return 0
                    return idx
                """
            ),
            tests=_dedent(
                f"""
                def test_find_substring_{idx}():
                    assert find_substring({text!r}, {pattern!r}) == {text.find(pattern)}
                    assert find_substring({text!r}, {missing!r}) == -1
                """
            ),
        )

    # thread_pool (10)
    for idx in range(1, 11):
        task_id = f"v10_{OSC_PREFIX['thread_pool']}_{idx:03d}"
        add(
            topic="thread_pool",
            task_id=task_id,
            prompt="Fix this thread pool simulation so tasks run FIFO order.",
            signature="class ThreadPool:",
            starter_code=_dedent(
                """
                class ThreadPool:
                    def __init__(self):
                        self.queue = []

                    def submit(self, fn, *args):
                        self.queue.append((fn, args))

                    def run_all(self):
                        results = []
                        while self.queue:
                            fn, args = self.queue.pop()
                            results.append(fn(*args))
                        return results
                """
            ),
            tests=_dedent(
                f"""
                def test_thread_pool_{idx}():
                    pool = ThreadPool()
                    pool.submit(lambda x: x + 1, 1)
                    pool.submit(lambda x: x + 1, 2)
                    pool.submit(lambda x: x + 1, 3)
                    assert pool.run_all() == [2, 3, 4]
                """
            ),
        )

    # timer (10)
    for idx in range(1, 11):
        interval = 5 + (idx % 3)
        task_id = f"v10_{OSC_PREFIX['timer']}_{idx:03d}"
        add(
            topic="timer",
            task_id=task_id,
            prompt="Fix this timer so it expires when elapsed >= interval.",
            signature="class ManualTimer:",
            starter_code=_dedent(
                """
                class ManualTimer:
                    def __init__(self, interval_ms: int):
                        self.interval_ms = interval_ms
                        self.elapsed = 0

                    def tick(self, delta_ms: int):
                        self.elapsed += delta_ms

                    def expired(self) -> bool:
                        return self.elapsed > self.interval_ms
                """
            ),
            tests=_dedent(
                f"""
                def test_manual_timer_{idx}():
                    t = ManualTimer({interval})
                    t.tick({interval})
                    assert t.expired() is True
                """
            ),
        )

    # topological_sort (10)
    for idx in range(1, 11):
        a = f"A{idx}"
        b = f"B{idx}"
        c = f"C{idx}"
        graph = {a: [b], b: [c]}
        expected = _topological_sort(graph)
        task_id = f"v10_{OSC_PREFIX['topological_sort']}_{idx:03d}"
        add(
            topic="topological_sort",
            task_id=task_id,
            prompt="Return a topological ordering that includes all nodes.",
            signature="def topological_sort(graph: dict) -> list:",
            starter_code=_dedent(
                """
                def topological_sort(graph: dict) -> list:
                    indegree = {node: 0 for node in graph}
                    for node, nbrs in graph.items():
                        for nbr in nbrs:
                            indegree[nbr] = indegree.get(nbr, 0) + 1
                    queue = [n for n in graph if indegree.get(n, 0) == 0]
                    order = []
                    while queue:
                        node = queue.pop(0)
                        order.append(node)
                        for nbr in graph.get(node, []):
                            indegree[nbr] -= 1
                            if indegree[nbr] == 0:
                                queue.append(nbr)
                    return order
                """
            ),
            tests=_dedent(
                f"""
                def test_topological_sort_{idx}():
                    graph = {graph!r}
                    order = topological_sort(graph)
                    assert set(order) == set({expected!r})
                    assert len(order) == len({expected!r})
                """
            ),
        )

    # trie (10)
    for idx in range(1, 11):
        word = f"word{idx}"
        prefix = "wor"
        task_id = f"v10_{OSC_PREFIX['trie']}_{idx:03d}"
        add(
            topic="trie",
            task_id=task_id,
            prompt="Fix Trie search so it only returns True for complete words.",
            signature="class Trie:",
            starter_code=_dedent(
                """
                class TrieNode:
                    def __init__(self):
                        self.children = {}
                        self.is_end = False

                class Trie:
                    def __init__(self):
                        self.root = TrieNode()

                    def insert(self, word: str) -> None:
                        node = self.root
                        for ch in word:
                            node = node.children.setdefault(ch, TrieNode())

                    def search(self, word: str) -> bool:
                        node = self.root
                        for ch in word:
                            if ch not in node.children:
                                return False
                            node = node.children[ch]
                        return True
                """
            ),
            tests=_dedent(
                f"""
                def test_trie_search_{idx}():
                    trie = Trie()
                    trie.insert({word!r})
                    assert trie.search({word!r}) is True
                    assert trie.search({prefix!r}) is False
                """
            ),
        )

    assert len(tasks) == 200, f"Expected 200 oscillation tasks, got {len(tasks)}"
    return tasks


def generate_v11_oscillation() -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []

    def add(
        *,
        topic: str,
        task_id: str,
        prompt: str,
        signature: str,
        starter_code: str,
        tests: str,
    ) -> None:
        tasks.append(
            _make_task(
                task_id=task_id,
                prompt=prompt,
                signature=signature,
                starter_code=starter_code,
                tests=tests,
                category="oscillation",
                topic=topic,
                tier="oscillation",
                starter_check="fail",
                approaches=OSC_APPROACHES[topic],
            )
        )

    # string_match (25)
    for idx in range(1, 26):
        mod = (idx - 1) % 5
        if mod == 0:
            pattern = "aa"
            text = "a" * (5 + (idx % 5))
        elif mod == 1:
            pattern = "aba"
            text = "ab" * (3 + (idx % 3)) + "a"
        elif mod == 2:
            pattern = "abc"
            text = "abc" * (3 + (idx % 2)) + "ab"
        elif mod == 3:
            pattern = "issi"
            text = "mississippi" + ("ssi" if idx % 2 == 0 else "")
        else:
            pattern = "aaab"
            text = "aaab" * (2 + (idx % 3)) + "aaab"
        expected = _find_all_overlapping(text, pattern)
        task_id = f"v11_{OSC_PREFIX['string_match']}_{idx:03d}"
        add(
            topic="string_match",
            task_id=task_id,
            prompt="Fix this function so it returns all start indices of pattern in text, including overlaps.",
            signature="def find_all_indices(text: str, pattern: str) -> list:",
            starter_code=_dedent(
                """
                def find_all_indices(text: str, pattern: str) -> list:
                    indices = []
                    i = 0
                    while i < len(text):
                        j = text.find(pattern, i)
                        if j == -1:
                            break
                        indices.append(j)
                        i = j + len(pattern)
                    return indices
                """
            ),
            tests=_dedent(
                f"""
                def test_find_all_indices_{idx}():
                    assert find_all_indices({text!r}, {pattern!r}) == {expected!r}
                    assert find_all_indices({text!r}, "zzz") == []
                """
            ),
        )

    # interval_merge (25)
    for idx in range(1, 26):
        base = idx * 3
        intervals = [
            (base, base + 2),
            (base + 1, base + 4),
            (base + 6, base + 8),
            (base + 8, base + 9),
        ]
        if idx % 2 == 0:
            intervals.append((base - 3, base - 1))
        if idx % 3 == 0:
            intervals.append((base + 4, base + 5))
        if idx % 2 == 0:
            intervals = list(reversed(intervals))
        expected = _merge_intervals(intervals)
        task_id = f"v11_{OSC_PREFIX['interval_merge']}_{idx:03d}"
        add(
            topic="interval_merge",
            task_id=task_id,
            prompt="Fix merge_intervals so it merges overlapping or touching intervals even if input is unsorted.",
            signature="def merge_intervals(intervals: list[tuple]) -> list[tuple]:",
            starter_code=_dedent(
                """
                def merge_intervals(intervals: list[tuple]) -> list[tuple]:
                    if not intervals:
                        return []
                    merged = [list(intervals[0])]
                    for start, end in intervals[1:]:
                        last = merged[-1]
                        if start < last[1]:
                            last[1] = max(last[1], end)
                        else:
                            merged.append([start, end])
                    return [tuple(x) for x in merged]
                """
            ),
            tests=_dedent(
                f"""
                def test_merge_intervals_{idx}():
                    assert merge_intervals({intervals!r}) == {expected!r}
                    assert merge_intervals([]) == []
                """
            ),
        )

    # sorting (25)
    for idx in range(1, 26):
        pairs = [
            ("alpha", 3 + (idx % 3)),
            ("beta", 1 + (idx % 2)),
            ("gamma", 2 + (idx % 4)),
            ("delta", 1 + ((idx + 1) % 2)),
        ]
        expected = sorted(pairs, key=lambda p: (p[1], p[0]))
        task_id = f"v11_{OSC_PREFIX['sorting']}_{idx:03d}"
        add(
            topic="sorting",
            task_id=task_id,
            prompt="Fix this function so it sorts pairs by score ascending, then by name.",
            signature="def sort_pairs(pairs: list[tuple]) -> list[tuple]:",
            starter_code=_dedent(
                """
                def sort_pairs(pairs: list[tuple]) -> list[tuple]:
                    pairs.sort()
                    return pairs
                """
            ),
            tests=_dedent(
                f"""
                def test_sort_pairs_{idx}():
                    data = {pairs!r}
                    assert sort_pairs(data) == {expected!r}
                """
            ),
        )

    # search (25)
    for idx in range(1, 26):
        nums = list(range(idx, idx + 20, 2))
        target = nums[idx % len(nums)]
        missing = nums[-1] + 1
        expected = nums.index(target)
        task_id = f"v11_{OSC_PREFIX['search']}_{idx:03d}"
        add(
            topic="search",
            task_id=task_id,
            prompt="Fix binary_search to return the index of target, or -1 if missing.",
            signature="def binary_search(nums: list[int], target: int) -> int:",
            starter_code=_dedent(
                """
                def binary_search(nums: list[int], target: int) -> int:
                    low, high = 0, len(nums) - 1
                    while low < high:
                        mid = (low + high) // 2
                        if nums[mid] == target:
                            return mid
                        if nums[mid] < target:
                            low = mid + 1
                        else:
                            high = mid - 1
                    return -1
                """
            ),
            tests=_dedent(
                f"""
                def test_binary_search_{idx}():
                    nums = {nums!r}
                    assert binary_search(nums, {target}) == {expected}
                    assert binary_search(nums, {missing}) == -1
                """
            ),
        )

    # heap (25)
    for idx in range(1, 26):
        values = [
            (idx * 3) % 17,
            (idx * 5) % 19,
            (idx * 7) % 23,
            (idx + 4) % 11,
            (idx * 2) % 13,
        ]
        expected = sorted(values)
        task_id = f"v11_{OSC_PREFIX['heap']}_{idx:03d}"
        add(
            topic="heap",
            task_id=task_id,
            prompt="Fix this MinHeap so pop() returns values in ascending order.",
            signature="class MinHeap:",
            starter_code=_dedent(
                """
                class MinHeap:
                    def __init__(self):
                        self.data = []

                    def push(self, value: int):
                        self.data.append(value)
                        self._sift_up(len(self.data) - 1)

                    def pop(self):
                        if not self.data:
                            return None
                        if len(self.data) == 1:
                            return self.data.pop()
                        root = self.data[0]
                        self.data[0] = self.data.pop()
                        self._sift_down(0)
                        return root

                    def _sift_up(self, idx: int):
                        while idx > 0:
                            parent = (idx - 1) // 2
                            if self.data[parent] < self.data[idx]:
                                self.data[parent], self.data[idx] = self.data[idx], self.data[parent]
                                idx = parent
                            else:
                                break

                    def _sift_down(self, idx: int):
                        n = len(self.data)
                        while True:
                            left = idx * 2 + 1
                            right = idx * 2 + 2
                            smallest = idx
                            if left < n and self.data[left] > self.data[smallest]:
                                smallest = left
                            if right < n and self.data[right] > self.data[smallest]:
                                smallest = right
                            if smallest == idx:
                                break
                            self.data[idx], self.data[smallest] = self.data[smallest], self.data[idx]
                            idx = smallest
                """
            ),
            tests=_dedent(
                f"""
                def test_min_heap_{idx}():
                    h = MinHeap()
                    for v in {values!r}:
                        h.push(v)
                    out = [h.pop() for _ in range({len(values)})]
                    assert out == {expected!r}
                """
            ),
        )

    # priority_queue (25)
    for idx in range(1, 26):
        items = [
            ("alpha", 3 + (idx % 3)),
            ("beta", 1 + (idx % 3)),
            ("gamma", 2 + (idx % 3)),
            ("delta", 4 + (idx % 3)),
        ]
        expected = [v for v, _p in sorted(items, key=lambda x: x[1])]
        task_id = f"v11_{OSC_PREFIX['priority_queue']}_{idx:03d}"
        add(
            topic="priority_queue",
            task_id=task_id,
            prompt="Fix this priority queue so pop() returns the smallest priority first.",
            signature="class PriorityQueue:",
            starter_code=_dedent(
                """
                class PriorityQueue:
                    def __init__(self):
                        self.items = []

                    def push(self, value, priority: int):
                        self.items.append((priority, value))
                        self.items.sort(reverse=True)

                    def pop(self):
                        if not self.items:
                            return None
                        return self.items.pop()[1]
                """
            ),
            tests=_dedent(
                f"""
                def test_priority_queue_{idx}():
                    pq = PriorityQueue()
                    for value, priority in {items!r}:
                        pq.push(value, priority)
                    out = [pq.pop() for _ in range({len(items)})]
                    assert out == {expected!r}
                """
            ),
        )

    # topological_sort (25)
    for idx in range(1, 26):
        nodes = ["A", "B", "C", "D", "E", "F"]
        edges = [
            ("A", "C"),
            ("B", "C"),
            ("C", "D"),
            ("A", "E"),
        ]
        if idx % 2 == 0:
            edges.append(("E", "F"))
        if idx % 3 == 0:
            edges.append(("B", "E"))
        graph = {n: [] for n in nodes}
        for u, v in edges:
            graph[u].append(v)
        task_id = f"v11_{OSC_PREFIX['topological_sort']}_{idx:03d}"
        add(
            topic="topological_sort",
            task_id=task_id,
            prompt="Fix topo_sort so it returns a valid topological ordering for the DAG.",
            signature="def topo_sort(graph: dict) -> list:",
            starter_code=_dedent(
                """
                def topo_sort(graph: dict) -> list:
                    visited = set()
                    order = []

                    def dfs(node):
                        if node in visited:
                            return
                        for nbr in graph.get(node, []):
                            dfs(nbr)
                        order.append(node)
                        visited.add(node)

                    for node in graph:
                        dfs(node)
                    return order
                """
            ),
            tests=_dedent(
                f"""
                def _is_valid_topo(order, graph):
                    if len(order) != len(graph):
                        return False
                    pos = {{n: i for i, n in enumerate(order)}}
                    for u, nbrs in graph.items():
                        for v in nbrs:
                            if pos[u] > pos[v]:
                                return False
                    return True

                def test_topo_sort_{idx}():
                    graph = {graph!r}
                    order = topo_sort(graph)
                    assert _is_valid_topo(order, graph) is True
                """
            ),
        )

    # trie (25)
    for idx in range(1, 26):
        word = f"word{idx}"
        prefix = word[:-1]
        other = f"w{idx}x"
        task_id = f"v11_{OSC_PREFIX['trie']}_{idx:03d}"
        add(
            topic="trie",
            task_id=task_id,
            prompt="Fix Trie search so it only returns True for complete words.",
            signature="class Trie:",
            starter_code=_dedent(
                """
                class TrieNode:
                    def __init__(self):
                        self.children = {}

                class Trie:
                    def __init__(self):
                        self.root = TrieNode()

                    def insert(self, word: str) -> None:
                        node = self.root
                        for ch in word:
                            node = node.children.setdefault(ch, TrieNode())

                    def search(self, word: str) -> bool:
                        node = self.root
                        for ch in word:
                            if ch not in node.children:
                                return False
                            node = node.children[ch]
                        return True
                """
            ),
            tests=_dedent(
                f"""
                def test_trie_search_{idx}():
                    trie = Trie()
                    trie.insert({word!r})
                    assert trie.search({word!r}) is True
                    assert trie.search({prefix!r}) is False
                    assert trie.search({other!r}) is False
                """
            ),
        )

    assert len(tasks) == 200, f"Expected 200 oscillation tasks, got {len(tasks)}"
    return tasks


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    tasks_dir = root / "data" / "tasks"
    v9_path = tasks_dir / "v9_fixedtiers.jsonl"
    v10_path = tasks_dir / "v10_osc_total_200.jsonl"
    v11_path = tasks_dir / "v11_osc_total_200.jsonl"

    v9 = generate_v9_bugfix()
    v10 = generate_v10_oscillation()
    v11 = generate_v11_oscillation()

    ids_v9 = [t["task_id"] for t in v9]
    ids_v10 = [t["task_id"] for t in v10]
    ids_v11 = [t["task_id"] for t in v11]
    assert len(ids_v9) == len(set(ids_v9)), "Duplicate task_id in v9"
    assert len(ids_v10) == len(set(ids_v10)), "Duplicate task_id in v10"
    assert len(ids_v11) == len(set(ids_v11)), "Duplicate task_id in v11"

    _write_jsonl(v9_path, v9)
    _write_jsonl(v10_path, v10)
    _write_jsonl(v11_path, v11)

    print(f"Wrote {len(v9)} tasks to {v9_path}")
    print(f"Wrote {len(v10)} tasks to {v10_path}")
    print(f"Wrote {len(v11)} tasks to {v11_path}")


if __name__ == "__main__":
    main()
