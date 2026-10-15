"""Day4 lecture and homework algorithms. Author: RedMoon.

Generated from Day4-阅读笔记.md; zero-based inclusive endpoints.
"""

def majority_element(a):
    if not a:
        raise ValueError("input must be nonempty")
    candidate, count = None, 0
    for x in a:
        if count == 0:
            candidate = x
        count += 1 if x == candidate else -1
    return candidate


from random import Random


def randomized_partition(a, p, r, rng):
    pivot_index = rng.randrange(p, r + 1)
    a[pivot_index], a[r] = a[r], a[pivot_index]
    x = a[r]
    i = p - 1
    for j in range(p, r):
        if a[j] <= x:
            i += 1
            a[i], a[j] = a[j], a[i]
    a[i + 1], a[r] = a[r], a[i + 1]
    return i + 1


def quicksort(a, rng=None):
    if rng is None:
        rng = Random()

    def sort(p, r):
        if p >= r:
            return
        q = randomized_partition(a, p, r, rng)
        sort(p, q - 1)
        sort(q + 1, r)

    sort(0, len(a) - 1)
    return a


def quickselect(a, rank, rng=None):
    if not 0 <= rank < len(a):
        raise ValueError("rank must be a valid zero-based index")
    if rng is None:
        rng = Random()
    p, r = 0, len(a) - 1
    while p <= r:
        q = randomized_partition(a, p, r, rng)
        if q == rank:
            return a[q]
        if rank < q:
            r = q - 1
        else:
            p = q + 1


import heapq


def heap_sort(a):
    heap = list(a)
    heapq.heapify(heap)
    result = []
    while heap:
        result.append(heapq.heappop(heap))
    return result


def counting_sort(a):
    if not a:
        return []
    upper = max(a)
    counts = [0] * (upper + 1)
    for x in a:
        counts[x] += 1
    result = []
    for x in range(1, upper + 1):
        result.extend([x] * counts[x])
    return result


def max_subarray(a):
    if not a:
        raise ValueError("nonempty subarray requires nonempty input")
    end_sum = best_sum = a[0]
    end_left = best_left = best_right = 0

    for j in range(1, len(a)):
        if end_sum + a[j] < a[j]:
            end_sum = a[j]
            end_left = j
        else:
            end_sum += a[j]
        if end_sum > best_sum:
            best_sum = end_sum
            best_left, best_right = end_left, j

    return best_sum, best_left, best_right


def merge_sorted_lists(lists):
    heap = [
        (row[0], list_id, 0)
        for list_id, row in enumerate(lists)
        if row
    ]
    heapq.heapify(heap)
    result = []
    while heap:
        value, list_id, index = heapq.heappop(heap)
        result.append(value)
        next_index = index + 1
        if next_index < len(lists[list_id]):
            next_value = lists[list_id][next_index]
            heapq.heappush(heap, (next_value, list_id, next_index))
    return result


def hybrid_sort(a, k, rng=None):
    if k < 2:
        raise ValueError("cutoff k must be at least 2")
    if rng is None:
        rng = Random()

    def partial_quicksort(p, r):
        if r - p + 1 < k:
            return
        q = randomized_partition(a, p, r, rng)
        partial_quicksort(p, q - 1)
        partial_quicksort(q + 1, r)

    partial_quicksort(0, len(a) - 1)

    for j in range(1, len(a)):
        key = a[j]
        i = j - 1
        while i >= 0 and a[i] > key:
            a[i + 1] = a[i]
            i -= 1
        a[i + 1] = key
    return a
