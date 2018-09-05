# -*- coding: utf-8 -*-

"""
    File name    :    KthLargest
    Date         :    12/08/2018
    Description  :    设计一个找到数据流中第K大元素的类（class）。每次调用 KthLargest.add，返回当前数据流中第K大的元素。
    Author       :    VickeeX
"""

import heapq


class KthLargest:
    def __init__(self, k, nums):
        """
        :type k: int
        :type nums: List[int]
        """
        self.heap = nums
        self.size = len(nums)
        self.k = k
        heapq.heapify(self.heap)
        while k < self.size:
            heapq.heappop(self.heap)
            self.size -= 1

    def add(self, val):
        """
        :type val: int
        :rtype: int
        """
        if self.size < self.k:
            heapq.heappush(self.heap, val)
            self.size += 1
        elif self.heap[0] < val:
            heapq.heapreplace(self.heap, val)
        return self.heap[0]
