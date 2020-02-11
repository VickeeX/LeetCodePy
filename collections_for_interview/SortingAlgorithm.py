# -*- coding: utf-8 -*-

"""
    @File name    :    SortingAlgorithm.py
    @Date         :    2020-02-09 16:45
    @Description  :    Thanks to: https://blog.csdn.net/weixin_41190227/article/details/86600821
                       * Compare Sort:
                            Bubble Sort, Select Sort, Insertion Sort, Shell Sort, Merge Sort, Quick Sort, Heap Sort,
                       * Non-compare Sort:
                            Counting Sort, Bucket Sort, Raidx Sort
    @Author       :    VickeeX
"""


def bubble_sort(arr):
    """
    Time complexity:
        avg: O(n^2); best: O(n); worst: O(n^2)
        note: "best: O(n)" while use swapTag
    Space complexity:
        O(1)
    """
    if len(arr) == 0:
        return arr
    for i in range(len(arr)):
        tag = True
        for j in range(len(arr) - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                tag = False
        if tag:
            break
    return arr


def select_sort(arr, tag):
    """
    Time complexity:
        avg: O(n^2); best: O(n^2); worst: O(n^2)
    Space complexity:
        O(1)
    表现稳定，适合数据规模较小的情况
    """
    if len(arr) == 0:
        return arr

    def min_select_sort(arr):
        for i in range(len(arr)):
            min_idx = i
            for j in range(i + 1, len(arr)):
                if arr[min_idx] > arr[j]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return arr

    def max_select_sort(arr):
        for i in range(len(arr) - 1, -1, -1):
            max_idx = i
            for j in range(i):
                if arr[max_idx] < arr[j]:
                    max_idx = j
            arr[i], arr[max_idx] = arr[max_idx], arr[i]
        return arr

    if tag:
        return min_select_sort(arr)
    else:
        return max_select_sort(arr)


def insertion_sort(arr):
    """
    Time complexity:
        avg: O(n^2); best: O(n); worst: O(n^2)
    Space complexity:
        O(1)
    表现稳定，适合数据规模较小的情况
    """
    if len(arr) == 0:
        return arr
    for i in range(len(arr) - 1):
        tmp, j = arr[i + 1], i
        while j >= 0 and tmp < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = tmp
    return arr


def shell_sort(arr):
    """
    Time complexity:
        avg: O(nlgn); best: O(nlgn); worst: O(nlgn)
    Space complexity:
        O(1)
    缩小增量的插入排序
    """
    if len(arr) == 0:
        return arr
    gap = len(arr) // 2
    while gap > 0:
        for i in range(gap, len(arr)):
            tmp, idx = arr[i], i - gap
            while idx >= 0 and arr[idx] > tmp:
                arr[idx + gap] = arr[idx]
                idx -= gap
            arr[idx + gap] = tmp
        gap //= 2
    return arr


def merge_sort(arr):
    """
    Time complexity:
        avg: O(nlgn); best: O(nlgn); worst: O(nlgn)
    Space complexity:
        O(n)
    """
    if len(arr) < 2:
        return arr

    def merge(left, right):
        tmp, l, r, li, ri = [], len(left), len(right), 0, 0
        for idx in range(l + r):
            if li >= l or (li < l and ri < r and left[li] > right[ri]):
                tmp.append(right[ri])
                ri += 1
            else:
                tmp.append(left[li])
                li += 1
        return tmp

    mid = len(arr) // 2
    left_arr = merge_sort(arr[:mid])
    right_arr = merge_sort(arr[mid:])
    return merge(left_arr, right_arr)


def quick_sort(arr):
    """
    choose a pivot as base
    sort list as two parts: smaller than pivot, bigger than pivot
    recursive sort the left part and right part

    Time complexity:
        avg: O(nlgn); best: O(nlgn); worst: O(n^2)
    Space complexity:
        O(nlgn) or O(lgn) : O(1*lgn) here as O(1) for partition and O(lgn) for recursion times
    """

    def recursion(ar, start, end):
        if len(ar) == 0 or start < 0 or end >= len(ar) or start > end:
            return []
        idx = partition(ar, start, end)
        if idx > start:
            recursion(ar, start, idx - 1)
        if idx < end:
            recursion(ar, idx + 1, end)
        return ar

    def partition(ar, start, end):
        from random import randint
        pivot = randint(start, end)
        idx = start - 1  # the last position of smaller part
        ar[pivot], ar[end] = ar[end], ar[pivot]
        for i in range(start, end + 1):
            if ar[i] <= ar[end]:  # use <= to put the pivot to the middle of partition
                idx += 1
                ar[i], ar[idx] = ar[idx], ar[i]
        # # if use "ar[i] < ar[end]"
        # idx += 1
        # ar[idx], ar[end] = ar[end], ar[idx]
        return idx

    return recursion(arr, 0, len(arr) - 1)


def heap_sort(arr):
    """
    build a standard max heap
    each round: swap the max heap(first element) to expand the sorted part, and adjust the left part

    Time complexity:
        avg: O(nlgn); best: O(nlgn); worst: O(nlgn)
    Space complexity:
        O(lg1): in place
    """
    if len(arr) < 2:
        return arr

    def adjust(ar, i, l):
        # l is the total length of unsorted partition
        idx = i
        for j in (i * 2, i * 2 + 1):  # if left or right child is bigger
            if j < l and ar[j] > ar[idx]:
                idx = j
        if idx != i:  # if any swap, adjust the sub tree to max heap
            ar[idx], ar[i] = ar[i], ar[idx]
            adjust(ar, idx, l)

    # build max heap first by adjusting each sub tree as heap
    for i in range(len(arr) // 2 - 1, -1, -1):
        adjust(arr, i, len(arr))

    # each round: move the heap to sorted right partition, and adjust left unsorted partition
    for i in range(len(arr) - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        adjust(arr, 0, i)
    return arr


def counting_sort(arr):
    """
    find the min and max num of array
    count the occurrences of i, store into the i'th place of tmp array
    fill the original array according to counts
    note: here use bias to reduce the extra space of tmp array

    Time complexity:
        avg: O(n+k); best: O(n+k); worst: O(n+k)
        k = max_num - min_num while use bias
    Space complexity:
        O(k)
    """
    if len(arr) < 2:
        return arr
    min_n = max_n = arr[0]
    for n in arr:  # find the min and max num
        if n < min_n:
            min_n = n
        if n > max_n:
            max_n = n
    bias, counts = 0 - min_n, [0] * (max_n - min_n + 1)  # use bias to reduce extra space
    for n in arr:
        counts[n + bias] += 1
    idx = 0
    for n, count in enumerate(counts):
        if count != 0:
            arr[idx:idx + count] = [n - bias] * count
        idx += count
    return arr


def bucket_sort(arr, bucket_size):
    """
    upper level of counting sort
    put nums to buckets according to nums' value mapping
    use bucket sort recursively or other sort method to sort each bucket
    factually, bucket_size is the gap of values instead size of buckets ——> bucket_size==1, but bucket may be [1,1,1]

    Time complexity:
        avg: O(n+k); best: O(n+k); worst: O(n^2)
        k = max_num - min_num while use bias
    Space complexity:
        O(n+k)
    """
    if len(arr) < 2:
        return arr
    min_n = max_n = arr[0]
    for n in arr:
        if n < min_n:
            min_n = n
        if n > max_n:
            max_n = n

    bucket_count, buckets = (max_n - min_n) // bucket_size + 1, []
    for i in range(bucket_count):
        buckets.append([])
    for n in arr:  # map num to buckets as back buckets have larger numbers
        buckets[(n - min_n) // bucket_size].append(n)

    ans = []
    for bucket in buckets:
        if bucket_size == 1:  # do not use "ans.append(bucket[0])" as the bucket may have repeated nums
            ans += bucket
        else:
            if bucket_count == 1:  # to avoid endless loop in one bucket
                bucket_size -= 1
            tmp = bucket_sort(bucket, bucket_size)
            ans += tmp
    return ans


def raidx_sort(arr):
    """
    sort from low digit to high digit

    Time complexity:
        avg: O(n*k); best: O(n*k); worst: O(n*k)
        k = max_num - min_num while use bias
    Space complexity:
        O(n+k)
    """
    if len(arr) < 2:
        return arr
    max_n = max(arr)
    max_digit, buckets, mod, div = len(str(max_n)), [], 10, 1
    for i in range(10):
        buckets.append([])
    for i in range(max_digit):
        for n in arr:
            buckets[(n % mod) // div].append(n)
        arr = []
        for bucket in buckets:
            for n in bucket:
                arr.append(n)
            bucket.clear()
        mod, div = mod * 10, div * 10
    return arr

# TODO: doubt the time complexity of Bucket Sort and Raidx Sort
