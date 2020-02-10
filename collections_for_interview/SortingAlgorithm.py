# -*- coding: utf-8 -*-

"""
    @File name    :    SortingAlgorithm.py
    @Date         :    2020-02-09 16:45
    @Description  :    sorting algorithms
    @Author       :    VickeeX
"""
"""
比较排序：冒泡排序，归并排序，快速排序，堆排序
非比较排序：计数排序、基数排序、桶排序
"""


def bubbleSort(arr):
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


def selectSort(arr, tag):
    """
    Time complexity:
        avg: O(n^2); best: O(n^2); worst: O(n^2)
    Space complexity:
        O(1)
    表现稳定，适合数据规模较小的情况
    """
    if len(arr) == 0:
        return arr

    def minSelectSort(arr):
        for i in range(len(arr)):
            min_idx = i
            for j in range(i + 1, len(arr)):
                if arr[min_idx] > arr[j]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return arr

    def maxSelectSort(arr):
        for i in range(len(arr) - 1, -1, -1):
            max_idx = i
            for j in range(i):
                if arr[max_idx] < arr[j]:
                    max_idx = j
            arr[i], arr[max_idx] = arr[max_idx], arr[i]
        return arr

    if tag:
        return minSelectSort(arr)
    else:
        return maxSelectSort(arr)


def insertionSort(arr):
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


def shellSort(arr):
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


def mergeSort(arr):
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
    left_arr = mergeSort(arr[:mid])
    right_arr = mergeSort(arr[mid:])
    return merge(left_arr, right_arr)