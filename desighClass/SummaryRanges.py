# -*- coding: utf-8 -*-

"""
    File name    :    SummaryRanges
    Date         :    05/05/2019
    Description  :    {TODO}
    Author       :    VickeeX
"""


class SummaryRanges:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.intervals = []

    def addNum(self, val: int) -> None:
        if not self.intervals:
            self.intervals.append([val, val])
        else:
            pos = 0
            for s, e in self.intervals:
                if val > e + 1:
                    pos += 1
                    continue
                elif val < s - 1:
                    self.intervals = self.intervals[:pos] + [[val, val]] + self.intervals[pos:]
                elif val == s - 1:
                    self.intervals[pos][0] = val
                elif val == e + 1:
                    self.intervals[pos][1] = val
                    if pos + 1 < len(self.intervals) and (
                            val == self.intervals[pos + 1][0] or val + 1 == self.intervals[pos + 1][0]):
                        tmp = self.intervals[pos][0]
                        self.intervals = self.intervals[:pos] + self.intervals[pos + 1:]
                        self.intervals[pos][0] = tmp
                return
            self.intervals += [[val, val]]

    def getIntervals(self) -> list:
        return self.intervals

# Your SummaryRanges object will be instantiated and called as such:
# obj = SummaryRanges()
# obj.addNum(val)
# param_2 = obj.getIntervals()
