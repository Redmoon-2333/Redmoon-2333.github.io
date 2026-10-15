"""Independent checks for the Day4 algorithms. Author: RedMoon."""

from collections import Counter
from itertools import permutations, product
from math import ceil, floor, log
from pathlib import Path
from random import Random
import json
import time

import day4_algorithms as algorithms


def brute_max_sum(a):
    best = a[0]
    for left in range(len(a)):
        total = 0
        for right in range(left, len(a)):
            total += a[right]
            best = max(best, total)
    return best


def check_majority():
    cases = 0
    for size in range(1, 9):
        for a in product(range(3), repeat=size):
            value, count = Counter(a).most_common(1)[0]
            if count > size // 2:
                assert algorithms.majority_element(a) == value
                cases += 1
    return cases


def check_kadane():
    cases = 0
    for size in range(1, 8):
        for a in product(range(-2, 3), repeat=size):
            best, left, right = algorithms.max_subarray(a)
            assert 0 <= left <= right < size
            assert best == sum(a[left:right + 1]) == brute_max_sum(a)
            cases += 1
    assert algorithms.max_subarray([-5, -2, -8]) == (-2, 1, 1)
    assert algorithms.max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == (6, 3, 6)
    try:
        algorithms.max_subarray([])
    except ValueError:
        pass
    else:
        raise AssertionError("empty input must be rejected")
    return cases + 3


class FixedPivot:
    def __init__(self, index):
        self.index = index

    def randrange(self, start, stop):
        assert start <= self.index < stop
        return self.index


def check_partition():
    cases = 0
    for size in range(1, 8):
        for original in product(range(3), repeat=size):
            for pivot in range(size):
                a = [-99, *original, 99]
                q = algorithms.randomized_partition(a, 1, size, FixedPivot(pivot + 1))
                assert a[0] == -99 and a[-1] == 99
                assert Counter(a[1:-1]) == Counter(original)
                assert a[q] == original[pivot]
                assert all(x <= a[q] for x in a[1:q])
                assert all(x > a[q] for x in a[q + 1:-1])
                cases += 1
    a = [40, 30, 10, 20]
    assert algorithms.randomized_partition(a, 0, 3, FixedPivot(3)) == 1
    assert a == [10, 20, 40, 30]
    return cases + 1


def check_sort_and_select():
    cases = 0
    for size in range(8):
        for a in product(range(3), repeat=size):
            expected = sorted(a)
            assert algorithms.quicksort(list(a), Random(20261015)) == expected
            assert algorithms.heap_sort(a) == expected
            assert algorithms.counting_sort([x + 1 for x in a]) == [x + 1 for x in expected]
            for rank, value in enumerate(expected):
                assert algorithms.quickselect(list(a), rank, Random(rank)) == value
            cases += 3 + size
    return cases


def check_merge():
    rows = [[], [0], [1], [0, 0], [0, 1], [1, 1]]
    cases = 0
    for count in range(5):
        for lists in product(rows, repeat=count):
            expected = sorted(x for row in lists for x in row)
            assert algorithms.merge_sorted_lists(lists) == expected
            cases += 1
    rng = Random(20261015)
    for _ in range(1000):
        lists = [
            sorted(rng.randrange(-5, 6) for _ in range(rng.randrange(20)))
            for _ in range(rng.randrange(13))
        ]
        assert algorithms.merge_sorted_lists(lists) == sorted(
            x for row in lists for x in row
        )
        cases += 1
    return cases


def partial_state(a, k, rng):
    leaves = []

    def split(left, right):
        if right - left + 1 < k:
            if left <= right:
                leaves.append((left, right))
            return
        q = algorithms.randomized_partition(a, left, right, rng)
        split(left, q - 1)
        split(q + 1, right)

    split(0, len(a) - 1)
    return leaves


def check_hybrid():
    cases = 0
    for size in range(8):
        for original in product(range(3), repeat=size):
            for k in range(2, 10):
                a = list(original)
                result = algorithms.hybrid_sort(a, k, Random(k + size))
                assert result is a and result == sorted(original)
                cases += 1

    block_cases = 0
    for size in range(1, 8):
        for original in permutations(range(size)):
            for k in range(2, size + 2):
                a = list(original)
                leaves = partial_state(a, k, Random(size * 17 + k))
                block_of = {}
                for block_id, (left, right) in enumerate(leaves):
                    assert right - left + 1 < k
                    for index in range(left, right + 1):
                        block_of[index] = block_id
                inversions = 0
                for left in range(size):
                    for right in range(left + 1, size):
                        if a[left] > a[right]:
                            inversions += 1
                            assert left in block_of and right in block_of
                            assert block_of[left] == block_of[right]
                assert inversions <= sum(
                    (right - left + 1) * (right - left) // 2
                    for left, right in leaves
                )
                assert 2 * inversions <= size * (k - 1)
                block_cases += 1

    original_partition = algorithms.randomized_partition
    calls = []

    def record_partition(a, left, right, rng):
        calls.append((left, right))
        return original_partition(a, left, right, rng)

    algorithms.randomized_partition = record_partition
    try:
        algorithms.hybrid_sort([3, 2, 1], 4, Random(0))
        assert len(calls) == 0
        algorithms.hybrid_sort([4, 3, 2, 1], 4, Random(0))
        assert len(calls) == 1
    finally:
        algorithms.randomized_partition = original_partition

    a = [3, 1, 2, 4, 6, 5, 7, 9, 8]
    assert sum(a[i] > a[j] for i in range(9) for j in range(i + 1, 9)) == 4
    assert algorithms.hybrid_sort(a, 4, Random(7)) == list(range(1, 10))
    return {"sorting_cases": cases + 4, "block_invariant_cases": block_cases}


def check_formulas():
    comparisons = [0.0, 0.0]
    prefix = 0.0
    harmonic = 1.0
    for size in range(2, 513):
        harmonic += 1 / size
        value = size - 1 + 2 * prefix / size
        comparisons.append(value)
        assert abs(value - (2 * (size + 1) * harmonic - 4 * size)) < 1e-8
        prefix += value
    for size in range(2, 1025):
        ranks = range(floor(size / 4) + 1, ceil(3 * size / 4) + 1)
        assert 2 * len(ranks) >= size
        assert all(max(rank - 1, size - rank) <= 3 * size / 4 for rank in ranks)
        assert sum(max(i, size - 1 - i) for i in range(size)) / size <= 3 * size / 4
    assert abs(log(2 ** 20, 2) - 20) < 1e-10
    return {"comparison_recurrence_sizes": 511, "good_pivot_sizes": 1023}


def main():
    started = time.perf_counter()
    results = {
        "majority": check_majority(),
        "partition": check_partition(),
        "max_subarray": check_kadane(),
        "sorting_and_selection": check_sort_and_select(),
        "k_way_merge": check_merge(),
        "hybrid": check_hybrid(),
        "formulas": check_formulas(),
        "status": "PASS",
    }
    results["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report = Path(__file__).with_name("verification-results.json")
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
