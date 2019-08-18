# -*- coding: utf-8 -*-

"""
    File name    :    PointSolution
    Date         :    28/02/2019
    Description  :    {TODO}
    Author       :    VickeeX
"""

import random
from collections import defaultdict


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


class RandomPointInCircle:
    def __init__(self, radius: float, x_center: float, y_center: float):
        self.xmin, self.xmax = x_center - radius, x_center + radius
        self.ymin, self.ymax = y_center - radius, y_center + radius
        self.x_center, self.y_center = x_center, y_center
        self.radius_pow = radius ** 2

    def randPoint(self) -> list:
        import random
        while True:
            x, y = random.uniform(self.xmin, self.xmax), random.uniform(self.ymin, self.ymax)
            if (x - self.x_center) ** 2 + (y - self.y_center) ** 2 <= self.radius_pow:
                return [x, y]


class RandomPointFromAreas:

    def __init__(self, rects: list):
        self.areas = []
        for i, r in enumerate(rects):
            x1, y1, x2, y2 = r
            self.areas.append((x2 - x1 + 1) * (y2 - y1 + 1) + (self.areas[i - 1] if i != 0 else 0))
        self.rects = rects
        print(self.rects, self.areas)

    def pick(self) -> list:
        selected = random.randint(1, self.areas[-1])  # select a rect by area size
        low, high = 0, len(self.areas) - 1
        while low <= high:
            mid = (low + high) // 2
            if selected <= self.areas[mid]:
                high = mid - 1
            else:
                low = mid + 1

        print(selected)
        x1, y1, x2, y2 = self.rects[low]
        if low != 0:
            selected -= self.areas[low - 1]  # specific point: each rect can be regarded as a rang of points
        selected -= 1  # start from 0,0
        return [x1 + selected % (x2 - x1 + 1), y1 + selected // (x2 - x1 + 1)]
