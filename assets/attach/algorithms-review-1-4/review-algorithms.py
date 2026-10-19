"""Algorithms accompanying the four-lecture review. Zero-based indices.
Nonempty arrays and cutoff k >= 2 are required where stated in the guide.
"""

def selection_sort(a):
    for end in range(len(a) - 1, 0, -1):
        max_index = 0
        for i in range(1, end + 1):
            if a[i] > a[max_index]:
                max_index = i
        a[max_index], a[end] = a[end], a[max_index]
    return a


def two_sum_sorted_search(values, target):
    a = merge_sort(values)
    left, right = 0, len(a) - 1
    while left < right:
        current = a[left] + a[right]
        if current == target:
            return True
        if current < target:
            left += 1
        else:
            right -= 1
    return False


def merge_sorted(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    while i < len(left):
        result.append(left[i])
        i += 1
    while j < len(right):
        result.append(right[j])
        j += 1
    return result


def merge_sort(a):
    if len(a) <= 1:
        return list(a)
    mid = len(a) // 2
    left = merge_sort(a[:mid])
    right = merge_sort(a[mid:])
    return merge_sorted(left, right)


def max_subarray(a):
    ending = best = a[0]
    start = best_left = best_right = 0
    for j in range(1, len(a)):
        if ending < 0:
            ending = a[j]
            start = j
        else:
            ending += a[j]
        if ending > best:
            best = ending
            best_left, best_right = start, j
    return best, best_left, best_right


import heapq


def merge_k_sorted(lists):
    heap = []
    for table, values in enumerate(lists):
        if values:
            heap.append((values[0], table, 0))
    heapq.heapify(heap)
    result = []
    while heap:
        value, table, index = heapq.heappop(heap)
        result.append(value)
        next_index = index + 1
        if next_index < len(lists[table]):
            heapq.heappush(
                heap,
                (lists[table][next_index], table, next_index),
            )
    return result


def hybrid_sort(a, k, rng=None):
    if rng is None:
        rng = Random()

    def partial(left, right):
        if right - left + 1 < k:
            return
        pivot = randomized_partition(a, left, right, rng)
        partial(left, pivot - 1)
        partial(pivot + 1, right)

    partial(0, len(a) - 1)
    insertion_sort(a)
    return a


def insertion_sort(a):
    for j in range(1, len(a)):
        key = a[j]
        i = j - 1
        while i >= 0 and a[i] > key:
            a[i + 1] = a[i]
            i -= 1
        a[i + 1] = key
    return a


from random import Random


def partition(a, left, right):
    pivot = a[right]
    boundary = left - 1
    for scan in range(left, right):
        if a[scan] <= pivot:
            boundary += 1
            a[boundary], a[scan] = a[scan], a[boundary]
    final = boundary + 1
    a[final], a[right] = a[right], a[final]
    return final


def randomized_partition(a, left, right, rng):
    chosen = rng.randrange(left, right + 1)
    a[chosen], a[right] = a[right], a[chosen]
    return partition(a, left, right)
