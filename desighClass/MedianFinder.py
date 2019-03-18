# -*- coding: utf-8 -*-

"""
    File name    :    MedianFinder
    Date         :    18/03/2019
    Description  :    {TODO}
    Author       :    VickeeX
"""
from heapq import *


class MedianFinder:
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.small, self.large = [], []

    def addNum(self, num: int) -> None:
        if len(self.large) > len(self.small):  # add one to small
            heappush(self.small, -heappushpop(self.large, num))
        else:  # add one to large
            heappush(self.large, -heappushpop(self.small, -num))
            # print(self.small, self.large)

    def findMedian(self) -> float:
        if len(self.small) == len(self.large):
            return (self.large[0] - self.small[0]) / 2
        return self.large[0]

# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
