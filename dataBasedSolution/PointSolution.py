# -*- coding: utf-8 -*-

"""
    File name    :    PointSolution
    Date         :    28/02/2019
    Description  :    {TODO}
    Author       :    VickeeX
"""

from collections import defaultdict
from decimal import Decimal


class Point(object):
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b


class PointSolution(object):
    def maxPoints(self, points):
        """
        :type points: List[Point]
        :rtype: int
        """
        if len(points) < 2:
            return len(points)
        lines, point_count, ans = defaultdict(set), defaultdict(int), 0
        for i in range(len(points) - 1):
            for j in range(i, len(points)):
                x1, y1, x2, y2 = points[i].x, points[i].y, points[j].x, points[j].y
                if x1 == x2:
                    lines[x1].add((x1, y1))
                    lines[x1].add((x2, y2))
                else:
                    k = (y2 - y1) / (x2 - x1)
                    b = y2 - k * x2
                    lines[(k, b)].add((x1, y1))
                    lines[(k, b)].add((x2, y2))
        for p in points:
            point_count[(p.x, p.y)] += 1
        for l in lines:
            ans = max(ans, sum([point_count[(p[0], p[1])] for p in lines[l]]))
        return ans
