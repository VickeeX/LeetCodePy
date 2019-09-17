# -*- coding: utf-8 -*-

"""
    File name    :    ArraySolution
    Date         :    13/08/2018
    Description  :    {TODO}
    Author       :    VickeeX
"""

import math, operator, sys, bisect, heapq
from functools import reduce
from heapq import heappush, heappop
from collections import defaultdict, deque, Counter
from itertools import combinations


class Interval:
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e


class ArraySolution:
    def binarySearch(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        low, high = 0, len(nums) - 1
        while low <= high:
            mid = int((low + high) / 2)
            print(low, mid, high)
            if nums[mid] == target:
                return mid
            elif target < nums[mid]:
                high = mid - 1
            else:
                low = mid + 1
        return -1

    def toLowerCase(self, str):
        """
        :type str: str
         :rtype: str
         """
        return str.lower()

    """ 有两种特殊字符,第一种字符可以用一比特0来表示,第二种字符可以用两比特(10 或 11)来表示.
        现给一个由若干比特组成的字符串,给定的字符串总是由0结束:判断最后一个字符是否为一比特0.
    solution:
        若len(bits)>1 且 bits(-2)=0,则必定True; 否则需要记录并判断最后一个0之前的连续1的个数是否为偶数(刚好配对).
    """

    def isOneBitCharacter(self, bits):
        """
        :type bits: List[int]
        :rtype: bool
        """
        if len(bits) < 2 or bits[-2] == 0: return True
        i, begin, count = len(bits) - 2, False, 0
        while i >= 0:
            if bits[i] == 1:
                begin, count = True, count + 1
            if begin and bits[i] == 0:
                break
            i -= 1
        if count % 2 == 0:
            return True
        return False

    """ 给出一个字符串数组words,从中找出最长的一个单词，该单词是由words词典中其他单词逐步添加一个字母组成。
        若其中有多个可行的答案，则返回答案中字典序最小的单词。
    solution:
        遍历排序后的数组,若word[:-1]在集合中,则该单词符合条件,加入集合并与ans进行长度比较
    """

    def longestWord(self, words):
        """
        :type words: List[str]
        :rtype: str
        """
        wset, ans = set(['']), ''
        for word in sorted(words):
            if word[:-1] in wset:
                wset.add(word)
                if len(word) > len(ans):
                    ans = word
        return ans

    """ 给定一个整数类型的数组 nums，返回数组“中心索引”:数组中心索引的左侧所有元素相加的和等于右侧所有元素相加的和。
    solution:
        遍历得到数组和,再次顺序遍历,减去当前位置值,将和分为左/右判断是否相等.
    """

    def pivotIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        sums, curSum = sum(nums), 0
        for i, n in enumerate(nums):
            if curSum == (sums - curSum - n):
                return i
            curSum += n
        return -1

    """ 给定上边界和下边界数字，输出一个列表，列表的元素是边界（含边界）内所有的自除数。
        自除数 是指可以被它包含的每一位数除尽的数。
    """

    def selfDividingNumbers(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: List[int]
        """
        ans = []
        for num in range(left, right + 1):
            i, tmp, tag = 1, num, True
            while tmp > 0:
                i = tmp % 10
                if i == 0 or num % i != 0:
                    tag = False
                    break
                tmp //= 10
            if tag:
                ans.append(num)
        return ans

    """ 给你一个坐标 (sr, sc) 表示图像渲染开始的像素值（行 ，列）和一个新的颜色值 newColor，将与初始点像素值相同且通过上下左右
        相连的像素点进行渲染.
    solution:
        从初始点开始递归,若一个点符合条件,渲染后(先渲染该点,否则重复递归),将其上下左右符合条件的像素点也进行递归渲染.
    """

    def floodFill(self, image, sr, sc, newColor):
        """
        :type image: List[List[int]]
        :type sr: int
        :type sc: int
        :type newColor: int
        :rtype: List[List[int]]
        """
        if image[sr][sc] == newColor:
            return image
        old, relatPos, i = image[sr][sc], [[-1, 0], [0, -1], [0, 1], [1, 0]], 0
        image[sr][sc] = newColor
        while i < 4:
            tx, ty = sr + relatPos[i][0], sc + relatPos[i][1]
            if 0 <= tx < len(image) and 0 <= ty < len(image[0]) and image[tx][ty] == old:
                print(tx, ty, image[tx][ty], old, newColor)
                image = self.floodFill(image, tx, ty, newColor)
            i += 1
        return image

    def nextGreatestLetter(self, letters, target):
        """
        :type letters: List[str]
        :type target: str
        :rtype: str
        """
        if target >= letters[-1]:
            return letters[0]
        for c in letters:
            if target < c:
                return c

    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        dp = [cost[0], cost[1]]
        for x in range(2, len(cost)):
            dp.append(min(dp[x - 1], dp[x - 2]) + cost[x])  # dp[x-1],不加上该台阶cost的最小值;dp[x],加上该台阶cost的最小值
        return min(dp[-1], dp[-2])  # 加上倒数第二个台阶cost的最小值,加上最后一个台阶cost的最小值

    def dominantIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 1:
            return 0
        if nums[0] > nums[1]:
            max1, max2, ind = nums[0], nums[1], 0
        else:
            max1, max2, ind = nums[1], nums[0], 1
        for x in range(2, len(nums)):
            if nums[x] > max1:
                max2, max1, ind = max1, nums[x], x
            elif nums[x] > max2:
                max2 = nums[x]
            print(max1, max2, x)
        if max1 >= 2 * max2:
            return ind
        return -1

    def isToeplitzMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: bool
        """
        length, height, d = len(matrix[0]), len(matrix), defaultdict(set)
        for x in range(height):
            for y in range(length):
                d[x - y].add(matrix[x][y])
                if len(d[x - y]) > 1:
                    return False
        return True

    def largestTriangleArea(self, points):
        """
        :type points: List[List[int]]
        :rtype: float
        """
        # (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        ans, size = 0, len(points)
        triangleAreaCompute = lambda x, y, z: abs(
            (x[0] * (y[1] - z[1]) + y[0] * (z[1] - x[1]) + z[0] * (x[1] - y[1])) * 0.5)
        for i in range(size):
            for j in range(i + 1, size):
                for k in range(j + 1, size):
                    ans = max(ans, triangleAreaCompute(points[i], points[j], points[k]))
        return ans

    def flipAndInvertImage(self, A):
        """
        :type A: List[List[int]]
        :rtype: List[List[int]]
        """
        for i in range(len(A)):
            list.reverse(A[i])
            A[i] = list(map(lambda c: 1 - c, A[i]))
        return A

    """
    solution:
        一维线段若有重叠, max(p1[0],p2[0]) < min(p1[1],p2[1]), 同理,矩形重叠,则对角线在两个维度都重叠
    """

    def isRectangleOverlap(self, rec1, rec2):
        """
        :type rec1: List[int]
        :type rec2: List[int]
        :rtype: bool
        """
        return max(rec1[0], rec2[0]) < min(rec1[2], rec2[2]) and max(rec1[1], rec2[1]) < min(rec1[3], rec2[3])

    def numMagicSquaresInside(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """

        def isMagicSquare(square):
            for i in range(3):
                if any(n < 1 or n > 9 for n in square[i]) \
                        or sum(square[i]) != 15 or square[0][i] + square[1][i] + square[2][i] != 15:
                    return False
            return square[0][0] + square[2][2] == 10 and square[0][2] + square[2][0] == 10

        ans = 0
        for i in range(len(grid) - 2):
            for j in range(len(grid[0]) - 2):
                if isMagicSquare(list(row[j:j + 3] for row in grid[i:i + 3])):
                    ans += 1
        return ans

    def maxDistToClosest(self, seats):
        """
        :type seats: List[int]
        :rtype: int
        """
        count = []
        count.append(0)
        for i in seats:
            if i == 0:
                count[-1] += 1
            else:
                count.append(0)
        ans = (max(count) + 1) // 2
        if seats[0] == 0:
            ans = max(count[0], ans)
        if seats[-1] == 0:
            ans = max(count[-1], ans)
        return ans

    def maxDistToClosest1(self, seats):
        ans, last = 1, 0
        for i, n in enumerate(seats):
            if n == 0:
                if last == 0 or i == len(seats) - 1:
                    ans = max(ans, i - last + 1)
                else:
                    ans = max(ans, (i - last) // 2 + 1)
                if seats[i - 1] == 1:
                    last = i
            else:
                last = i + 1
        return ans

    def peakIndexInMountainArray(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        for i, n in enumerate(A):
            if n > A[i + 1]:
                return i

    def lemonadeChange(self, bills):
        """
        :type bills: List[int]
        :rtype: bool
        """
        a = b = 0
        for money in bills:
            if money == 5:
                a += 1
            elif money == 10:
                if a == 0:
                    return False
                a, b = a - 1, b + 1
            elif money == 20:
                if b > 0 and a > 0:
                    a, b = a - 1, b - 1
                elif a > 2:
                    a -= 3
                else:
                    return False
        return True

    def transpose(self, A):
        """
        :type A: List[List[int]]
        :rtype: List[List[int]]
        """
        return [[row[i] for row in A] for i in range(len(A[0]))]

    def robotSim(self, commands, obstacles):
        """
        :type commands: List[int]
        :type obstacles: List[List[int]]
        :rtype: int
        """
        dx, dy = [0, 1, 0, -1], [1, 0, -1, 0]  # 北,东,南,西
        obs = set(map(tuple, obstacles))
        x = y = di = ans = 0
        for c in commands:
            if c == -2:
                di = (di - 1) % 4  # 3->2, 2->1, 1->0, 0->3
            elif c == -1:
                di = (di + 1) % 4  # 0->1, 1->2, 2->3, 3->0
            else:
                for i in range(c):
                    if (x + dx[di], y + dy[di]) not in obs:
                        x += dx[di]
                        y += dy[di]
                    ans = max(ans, x * x + y * y)
        return ans

    def projectionArea(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        ans = 0
        for row in grid:
            ans += max(row)
            for i in row:
                if i != 0:
                    ans += 1

        gridNew = list(zip(*grid))
        for row in gridNew:
            ans += max(row)
        return ans
        # ans = 0
        # for row in grid:
        #     for i in row:
        #         if i != 0:
        #             ans += 1
        # return ans + sum([max(row) for row in grid]) + sum(
        #         max(row) for row in [[row[i] for row in grid] for i in range(len(grid[0]))])

    """
    solution:
        左右木板向中间靠拢, 若左木板长度小于右木板, 则左木板靠近一步:
           由于左木板更短, 为容器的高, 若右指针左移,底一定变短.
           右木板长短无论变高变短, 新高度都只会小于或等于上一次的高度.
    """

    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        ans = left = 0
        right = len(height) - 1
        while left < right:
            ans = max(ans, min(height[left], height[right]) * (right - left))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return ans

    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        if len(nums) < 3 or nums[0] > 0 or nums[-1] < 0 or (nums[0] >= 0 and nums[0] != nums[-1]):
            return []
        if nums[0] == 0 and nums[0] == nums[-1]:
            return [[0, 0, 0]]
        ans = []
        for i in range(len(nums) - 2):
            if nums[i] > 0:
                break
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            j, k = i + 1, len(nums) - 1
            while j < k:
                sums = nums[i] + nums[j] + nums[k]
                if sums == 0:
                    ans.append([nums[x] for x in [i, j, k]])
                    j += 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
                    k -= 1
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1
                elif sums < 0:
                    j += 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
                else:
                    k -= 1
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1
        return ans

    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        ans = sum(nums[:3])
        nums.sort()
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            j, k = i + 1, len(nums) - 1
            while j < k:
                sums = nums[i] + nums[j] + nums[k]
                if sums == target:
                    return target
                elif abs(sums - target) < abs(ans - target):
                    ans = sums
                if sums > target:
                    k -= 1
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1
                elif sums < target:
                    j += 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
        return ans

    def threeSumTarget(self, nums, target, n1):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        ans, ans0 = [], [n1]
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            j, k = i + 1, len(nums) - 1
            while j < k:
                sums = nums[i] + nums[j] + nums[k]
                if sums == target:
                    ans.append(ans0 + [nums[x] for x in [i, j, k]])
                    j += 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
                    k -= 1
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1
                elif sums < target:
                    j += 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
                else:
                    k -= 1
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1
        return ans

    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        if len(nums) < 4:
            return []
        nums.sort()
        ans = []
        for i in range(len(nums) - 3):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            ans += self.threeSumTarget(nums[i + 1:], target - nums[i], nums[i])
        return ans

    def findNsum(self, nums, target, N, result, results):
        if len(nums) < N or N < 2 or target < nums[0] * N or target > nums[-1] * N:  # early termination
            return
        if N == 2:  # two pointers solve sorted 2-sum problem
            l, r = 0, len(nums) - 1
            while l < r:
                s = nums[l] + nums[r]
                if s == target:
                    results.append(result + [nums[l], nums[r]])
                    l += 1
                    while l < r and nums[l] == nums[l - 1]:
                        l += 1
                elif s < target:
                    l += 1
                else:
                    r -= 1
        else:  # recursively reduce N
            for i in range(len(nums) - N + 1):
                if i == 0 or (i > 0 and nums[i - 1] != nums[i]):
                    self.findNsum(nums[i + 1:], target - nums[i], N - 1, result + [nums[i]], results)

    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        ans = []

        def generate(s='', left=0, right=0):
            if len(s) == 2 * n:
                ans.append(s)
                return
            if left < n:
                generate(s + '(', left + 1, right)
            if right < left:
                generate(s + ')', left, right + 1)

        generate()
        return ans

    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        for x in range(len(nums) - 1, -1, -1):
            if nums[x - 1] < nums[x]:
                break
        if x > 0:
            for y in range(len(nums) - 1, -1, -1):
                if nums[y] > nums[x - 1]:
                    nums[y], nums[x - 1] = nums[x - 1], nums[y]
                    break
        nums[x:] = nums[:x - 1:-1]
        # for i in range((len(nums) - x) // 2):
        #     nums[x + i], nums[len(nums) - i - 1] = nums[len(nums) - i - 1], nums[x + i]

    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if not nums:
            return -1
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            print(left, mid, right)
            if nums[mid] == target:
                return mid
            if nums[left] <= target < nums[mid] or (nums[mid] < nums[right] and not nums[mid] < target <= nums[right]):
                right = mid - 1
            else:
                left = mid + 1
                # if nums[left] < nums[mid]:  # 左半有序
                #     if nums[left] < target < nums[mid]:
                #         right = mid - 1
                #     else:
                #         left = mid + 1
                # else:  # 右半有序
                #     if nums[mid] < target < nums[right]:
                #         left = mid + 1
                #     else:
                #         right = mid - 1
        return -1

    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        if not nums:
            return [-1, -1]
        left, right, start = 0, len(nums) - 1, -1
        while left <= right:
            mid = (right + left) // 2
            if nums[mid] == target:
                start, right = mid, mid - 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        left, right, end = 0, len(nums) - 1, -1
        while left <= right:
            mid = (right + left) // 2
            if nums[mid] == target:
                end, left = mid, mid + 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return [start, end]

    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        ans = []
        for i in candidates:
            if i == target:
                ans.append([i])
            elif i < target and target - i >= i:
                tmp = self.combinationSum(candidates, target - i)
                if tmp:
                    for ta in tmp:
                        ta.sort()  # 保证后一个加入的数字大于等于之前的数字,否则结果重复
                        if ta[0] >= i:
                            ans.append([i] + ta)
        return ans

    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        candidates.sort()
        ans = []
        for i in range(len(candidates)):
            if i > 0 and candidates[i] == candidates[i - 1]:
                continue
            if candidates[i] > target:  # 此处不加, 会超时
                break
            if candidates[i] == target:
                ans.append([candidates[i]])
            else:
                tmp = self.combinationSum2(candidates[i + 1:], target - candidates[i])
                for ta in tmp:
                    ans.append([candidates[i]] + ta)
        return ans

    def combinationSum3(self, k: int, n: int) -> list:
        return [list(c) for c in combinations(range(1, 10), k) if sum(c) == n]

    def combinationSum4(self, nums: list, target: int) -> int:
        # # recursive in low speed
        # if target == 0: return 1
        # ans = 0
        # for n in nums:
        #     if n <= target:
        #         ans += self.combinationSum4(nums, target - n)
        # return ans
        dp = [1] + [-1] * target

        def helper(targ):
            ans = 0
            if dp[targ] != -1:
                return dp[targ]
            for n in nums:
                if n <= targ:
                    ans += helper(targ - n)
            dp[targ] = ans
            return ans

        helper(target)
        return dp[-1]

    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if not nums:
            return []
        if len(nums) == 1:
            return [nums]
        ans, tmp = [], self.permute(nums[1:])
        for i in tmp:
            for j in range(len(i) + 1):
                ans.append(i[:j] + [nums[0]] + i[j:])
        return ans

    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if not nums:
            return []
        if len(nums) == 1:
            return [nums]
        nums.sort()
        ans, tmp = [], self.permuteUnique(nums[1:])
        print(tmp)
        for i in tmp:
            for j in range(len(i)):
                t, tag = i[:j] + [nums[0]] + i[j:], True
                for a in ans:
                    if a == t:
                        tag = False
                        break
                if tag:
                    ans.append(t)
        return ans

    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """

        def spiralLayer(r1, r2, c1, c2):
            for c in range(c1, c2 + 1):
                yield r1, c
            for r in range(r1 + 1, r2 + 1):
                yield r, c2
            if r1 < r2 and c1 < c2:
                for c in range(c2 - 1, c1, -1):
                    yield r2, c
                for r in range(r2, r1, -1):
                    yield r, c1

        if not matrix:
            return []
        ans = []
        r1, r2 = 0, len(matrix) - 1
        c1, c2 = 0, len(matrix[0]) - 1
        while r1 <= r2 and c1 <= c2:
            for r, c in spiralLayer(r1, r2, c1, c2):
                ans.append(matrix[r][c])
            r1, r2 = r1 + 1, r2 - 1
            c1, c2 = c1 + 1, c2 - 1
        return ans

    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """

        def spiralLayer(r1, r2, c1, c2):
            for c in range(c1, c2 + 1):
                yield r1, c
            for r in range(r1 + 1, r2 + 1):
                yield r, c2
            if r1 < r2 and c1 < c2:
                for c in range(c2 - 1, c1, -1):
                    yield r2, c
                for r in range(r2, r1, -1):
                    yield r, c1

        ans = [[1 for i in range(n)] for j in range(n)]
        ta = 1
        r1, r2 = 0, n - 1
        c1, c2 = 0, n - 1
        while r1 <= r2 and c1 <= c2:
            for r, c in spiralLayer(r1, r2, c1, c2):
                ans[r][c] = ta
                ta += 1
            r1, r2 = r1 + 1, r2 - 1
            c1, c2 = c1 + 1, c2 - 1
        return ans

    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if not nums or len(nums) < 2:
            return True
        if nums[0] == 0:
            return False
        farest, cur, far = len(nums) - 1, 0, nums[0]
        while far < farest:
            nex = max([i + nums[i] for i in range(cur + 1, far + 1)])
            cur = far
            far = nex
            if cur == far < farest:
                return False
        return True

    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        intervals.sort(key=lambda x: x.start)

        merged = []
        for interval in intervals:
            # if the list of merged intervals is empty or if the current
            # interval does not overlap with the previous, simply append it.
            if not merged or merged[-1].end < interval.start:
                merged.append(interval)
            else:
                # otherwise, there is overlap, so we merge the current and previous
                # intervals.
                merged[-1].end = max(merged[-1].end, interval.end)

        return merged

    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        nums, ans, k = list(range(1, n + 1)), "", k - 1

        while n > 0:
            n -= 1
            m = math.factorial(n)
            idx = k // m
            ans += str(nums[idx])
            nums.remove(nums[idx])
            k %= m
        return ans

    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        return int(math.factorial(m + n - 2) / math.factorial(m - 1) / math.factorial(n - 1))

        # 动态规划:
        # ans = [[1] * n for i in range(m)]
        # for i in range(1, m):
        #     for j in range(1, n):
        #         ans[i][j] = ans[i - 1][j] + ans[i][j - 1]
        # return ans[m - 1][n - 1]

        # 递归: 超时
        # if m == 1 or n == 1:
        #     return 1
        # return self.uniquePaths(m - 1, n) + self.uniquePaths(m, n - 1)

    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        ans = [[0] * n for i in range(m)]

        for i in range(n):
            if obstacleGrid[0][i] != 1:
                ans[0][i] = 1
            else:
                break
        for i in range(m):
            if obstacleGrid[i][0] != 1:
                ans[i][0] = 1
            else:
                break
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:
                    ans[i][j] = 0
                else:
                    ans[i][j] = ans[i - 1][j] + ans[i][j - 1]
        return ans[m - 1][n - 1]

    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m, n = len(grid), len(grid[0])
        ans = [[0] * n for i in range(m)]
        ans[0][0] = grid[0][0]
        for i in range(1, n):
            ans[0][i] = grid[0][i] + ans[0][i - 1]
        for i in range(1, m):
            ans[i][0] = grid[i][0] + ans[i - 1][0]
        for i in range(1, m):
            for j in range(1, n):
                ans[i][j] = min(ans[i][j - 1], ans[i - 1][j]) + grid[i][j]
        return ans[m - 1][n - 1]

    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        row, column, lh, lw = set(), set(), len(matrix), len(matrix[0])
        for i in range(lh):
            for j in range(lw):
                if not matrix[i][j]:
                    row.add(i)
                    column.add(j)
        for r in row:
            for x in range(lw):
                matrix[r][x] = 0
        for c in column:
            for x in range(lh):
                matrix[x][c] = 0

    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix:
            return False
        rows, cols = len(matrix), len(matrix[0])
        left, right, find = 0, rows * cols - 1, False
        while left <= right:
            # if target < matrix[left // cols][left % cols] or target > matrix[right // cols][right % cols]:
            #     return False
            mid = (left + right) // 2
            mid_v = matrix[mid // cols][mid % cols]
            if target == mid_v:
                return True
            elif target < mid_v:
                right = mid - 1
            else:
                left = mid + 1
        return False

    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        left, right = 0, len(nums) - 1
        for i in nums:
            if i == 0:
                nums[left] = 0
                left += 1
            elif i == 2:
                right -= 1
        for i in range(left, right + 1):
            nums[i] = 1
        for i in range(right + 1, len(nums)):
            nums[i] = 2
        print(nums)

    def sortColorsMove(self, nums):
        i, j, k = -1, -1, -1
        for n in nums:
            if n == 0:
                i, j, k = i + 1, j + 1, k + 1
                nums[k] = 2
                nums[j] = 1
                nums[i] = 0
            elif n == 1:
                j, k = j + 1, k + 1
                nums[k] = 2
                nums[j] = 1
            else:
                k += 1
                nums[k] = 2

    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """

        # return list(combinations(range(1, n + 1), k))
        def helper(nums, kk):
            if kk == 1:
                return [[i] for i in nums]
            ans = []
            for i in range(len(nums)):
                for a in helper(nums[i + 1:], kk - 1):
                    ans.append([nums[i]] + a)
            return ans

        return [tuple(x) for x in helper(list(range(1, n + 1)), k)]

    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        ans = [[]]
        for n in nums:
            p = [sorted(s + [n]) for s in ans]
            ans += p
        return ans
        # for i in range(1, len(nums) + 1):
        #     for a in combinations(nums, i):
        #         ans.append(list(a))
        # return ans

    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        pos, count = 1, 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1]:
                count += 1
            else:
                count = 1
            if count < 3:
                nums[pos] = nums[i]
                pos += 1
        return pos

    def search2(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return True

            while nums[left] == nums[mid] and left < mid:
                left += 1

            if nums[left] <= nums[mid]:
                if nums[left] <= target <= nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] <= target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return False

    def grayCode(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        # return [i ^ (i >> 1) for i in range(2 ** n)]

        if n == 0:
            return [0]
        last, add = self.grayCode(n - 1), 2 ** (n - 1)
        return last + [add + last[i] for i in range(len(last) - 1, -1, -1)]

    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        ans = [[]]
        for n in nums:
            ta = [sorted(s + [n]) for s in ans]
            ans += [a for a in ta if a not in ans]
        return ans

    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        height = len(triangle)
        ans = [[0] * i for i in range(1, height + 1)]
        print(ans)
        ans[0][0] = triangle[0][0]
        for i in range(1, height):
            ans[i][0] = ans[i - 1][0] + triangle[i][0]
            ans[i][-1] = ans[i - 1][-1] + triangle[i][-1]
            for j in range(1, i):
                ans[i][j] = min(ans[i - 1][j - 1], ans[i - 1][j]) + triangle[i][j]
        return min(ans[-1])

    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        size = len(gas)
        i, cur_sum, start, end, red = 0, 0, 0, size, [gas[i] - cost[i] for i in range(size)]
        while i < end < 2 * size:
            if i >= size:
                ind = i % size
            else:
                ind = i
            cur_sum += red[ind]
            if cur_sum < 0:
                cur_sum, start = 0, i + 1
                end = start + size
            i += 1
        if start < size:
            return start
        return -1

    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # a, b = 0, 0
        # for n in nums:
        #     a = (a ^ n) & ~b
        #     b = (b ^ n) & ~a
        # return a
        return (sum(set(nums)) * 3 - sum(nums)) // 2

    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        l1, l2 = len(nums1), len(nums2)
        size = l1 + l2
        p1, p2 = 0, 0  # record the visited position of nums1 and nums2
        n1, n2 = 0, 0

        for i in range(1, (size + 4) // 2):
            n2 = n1
            if p1 >= l1:
                if p2 >= l2:
                    break
                n1 = nums2[p2]
                p2 += 1
            elif p2 >= l2:
                n1 = nums1[p1]
                p1 += 1
            elif nums1[p1] < nums2[p2]:
                n1 = nums1[p1]
                p1 += 1
            else:
                n1 = nums2[p2]
                p2 += 1

        if size % 2 == 0:
            return (n1 + n2) / 2
        else:
            return n1

            # use more space
            # nums1.extend(nums2)
            # nums1.sort()
            # return nums1[len(nums1)//2] if len(nums1) % 2 == 1
            # else float(nums1[(len(nums1)//2)-1]+nums1[len(nums1)//2])/2

    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        self._solveSudoku(board)

    def _solveSudoku(self, board):
        x, y = -1, -1
        for i in range(0, 9):
            for j in range(0, 9):
                if board[i][j] == '.':
                    x, y = i, j
                    break
            if x != -1:
                break
        if x == -1:  # finished filling in
            return True

        candiates = set("123456789")
        for i in range(0, 9):
            if board[x][i] in candiates:
                candiates.remove(board[x][i])  # remove nums appeared in row
            if board[i][y] in candiates:
                candiates.remove(board[i][y])  # remove nums appeared in column
        for i in range(x // 3 * 3, x // 3 * 3 + 3):
            for j in range(y // 3 * 3, y // 3 * 3 + 3):
                if board[i][j] in candiates:
                    candiates.remove(board[i][j])  # remove nums appeared in sub box
        for c in candiates:
            board[x][y] = c
            if self._solveSudoku(board):
                return True
            board[x][y] = '.'  # backtrace, select another candiate number
        return False

    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 1
        size = len(nums)  # 1 <= ans <= size+1
        for i in range(0, size):  # put num in [1,size] to position (num-1)
            cur = nums[i]
            while 0 < cur < size + 1 and cur != nums[cur - 1]:
                tmp = nums[cur - 1]
                nums[cur - 1] = cur
                cur = tmp

        for i in range(0, size):
            if nums[i] != i + 1:  # the first position i not put i+1 be the first missing
                return i + 1
        return size + 1  # 1~size all exists, first missing: size+1

    def trap_brute_force(self, height):  # O(n^2) time and const extra space
        """
        :type height: List[int]
        :rtype: int
        """
        ans, l, r, size = 0, 0, 0, len(height)
        for i in range(0, size):
            if i != 0:
                l = max(l, height[i - 1])
            if i != size - 1:
                r = max(height[i + 1:])
            if min(l, r) > height[i]:
                ans += min(l, r) - height[i]
        return ans

    def trap_dynamic(self, height):  # O(n) time and O(n) extra space
        size = len(height)
        ans, left, right = 0, [0] * size, [0] * size  # store the left_max and right_max
        for i in range(0, size):
            if i == 0:
                left[0] = height[0]
            else:
                left[i] = max(left[i - 1], height[i])
            if i == 0:
                right[-1] = height[-1]
            else:
                right[size - i - 1] = max(right[size - i], height[size - i - 1])
        print(left, right)
        for i in range(0, size):
            m = min(left[i], right[i])
            if m > height[i]:
                ans += m - height[i]
        return ans

    def jump(self, nums):
        ans, i = 0, 0
        while i < len(nums) - 1:
            if nums[i] + i >= len(nums) - 1:
                return ans + 1
            nexts = [x + j for j, x in enumerate(nums[i + 1:i + 1 + nums[i]])]
            i += nexts.index(max(nexts)) + 1  # greedy
            ans += 1
        return ans

    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """

        def dfs(queens, pos_sum, pos_dif):
            cur = len(queens)
            if cur == n:
                ans.append(queens)
                return
            for i in range(n):
                if i not in queens and cur + i not in pos_sum and cur - i not in pos_dif:
                    dfs(queens + [i], pos_sum + [cur + i], pos_dif + [cur - i])

        ans = []
        dfs([], [], [])
        return [['.' * i + 'Q' + '.' * (n - i - 1) for i in s] for s in ans]

    def totalNQueens(self, n):
        """
        :type n: int
        :rtype: int
        """

        def dfs(queens, pos_sum, pos_dif):
            cur = len(queens)
            if cur == n:
                ans.append(queens)
                return
            for i in range(n):
                if i not in queens and cur + i not in pos_sum and cur - i not in pos_dif:
                    dfs(queens + [i], pos_sum + [cur + i], pos_dif + [cur - i])

        ans = []
        dfs([], [], [])
        return len(ans)

    def insert(self, intervals, newInterval):
        """
        :type intervals: List[Interval]
        :type newInterval: Interval
        :rtype: List[Interval]
        """
        ans, used = [], False
        for i in range(0, len(intervals)):
            if newInterval.start > intervals[i].end:  # newInterval starts later
                ans += [intervals[i]]
            elif newInterval.end < intervals[i].start:  # newIntervals ends earlier
                ans, used = ans + [newInterval] + intervals[i:], True
                break
            else:  # encounters, modify newInterval to meet the next
                newInterval.start = min(intervals[i].start, newInterval.start)
                newInterval.end = max(intervals[i].end, newInterval.end)
        if not used:
            ans += [newInterval]
        print([[a.start, a.end] for a in ans])
        return ans

    def fullJustify(self, words, maxWidth):
        """
        :type words: List[str]
        :type maxWidth: int
        :rtype: List[str]
        """
        if not words:
            return [" " * maxWidth]
        ans, cur, num = [], [], 0
        for w in words:
            if num + len(w) + len(cur) > maxWidth:
                for i in range(maxWidth - num):
                    cur[i % (len(cur) - 1 or 1)] += ' '
                ans.append(''.join(cur))
                cur, num = [], 0
            cur, num = cur + [w], num + len(w)
        return ans + [' '.join(cur).ljust(maxWidth)]

    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """

        # to prevent board changing, do not use deepcopy() -- time limit exceeded
        def search_helper(ni, nj, new_board, new_word):
            if len(new_word) == 0:
                return True
            tmp = new_board[ni][nj]
            new_board[ni][nj] = '.'
            for x, y in [(ni - 1, nj), (ni, nj - 1), (ni + 1, nj), (ni, nj + 1)]:
                if 0 <= x < len(new_board) and 0 <= y < len(board[0]) and new_board[x][y] == new_word[0]:
                    if search_helper(x, y, new_board, new_word[1:]):
                        return True
            new_board[ni][nj] = tmp  # restore the board
            return False

        for i, row in enumerate(board, 0):
            for j, char in enumerate(row, 0):
                if char == word[0]:
                    if search_helper(i, j, board, word[1:]):
                        return True
        return False

    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        s, res, heights = [], 0, [0] + heights + [0]
        print(heights)
        for i, height in enumerate(heights):
            if len(s) > 0:
                while height < heights[s[-1]]:
                    top = s.pop()
                    print(i, height, ": ", s[-1], heights[s[-1]], top)
                    res = max(res, heights[top] * (i - s[-1] - 1))
            s.append(i)
            # print(s)
        return res

    def largestRectangleAreaDP(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        if not heights:
            return 0
        ans, dp = max(heights), [h for h in heights]  # dp[i][j]: min num from heights[j] to heights[j+i+1]
        for i in range(1, len(heights)):
            for j in range(0, len(heights) - i):
                dp[j] = min(dp[j], heights[i + j])
                ans = max(ans, dp[j] * (i + 1))
                # print(dp)
        return ans

    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if not matrix or not matrix[0]:
            return 0
        n = len(matrix[0])
        height, ans = [0] * (n + 1), 0
        for row in matrix:
            for i in range(0, n):
                height[i] = height[i] + 1 if row[i] == '1' else 0  # record the height of column
            stack = [-1]
            for i in range(0, n + 1):
                while height[i] < height[stack[-1]]:  # compute the constant width
                    h = height[stack.pop()]
                    w = i - 1 - stack[-1]
                    ans = max(ans, h * w)
                stack.append(i)
        return ans

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices) == 0 or len(prices) == 1:
            return 0
        b1, b2, s1, s2 = prices[0], prices[0], 0, 0
        for price in prices[1:]:
            b1 = min(b1, price)
            s1 = max(s1, price - b1)
            b2 = min(b2, price - s1)  # profit for last transaction is sell1
            s2 = max(s2, price - b2)
        return s2

    def maxProfit309(self, prices: list) -> int:
        # free    : the max profit while being free to buy
        # cooldown: the max profit while cooling down
        # kept    : the max profit while having stock
        free, cooldown, kept = 0, float('-inf'), float('-inf')
        for p in prices:
            free, cooldown, kept = max(free, cooldown), kept + p, max(kept, free - p)
            # print(free, cooldown, kept)
        return max(free, cooldown)

    def findLadders(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: List[List[str]]
        """
        words, dicts, ans = set(wordList), {}, []
        dicts[beginWord] = [[beginWord]]
        while dicts:
            new_dict = defaultdict(list)  # current word path
            for w in dicts:
                if w == endWord:
                    ans.extend(v for v in dicts[w])
                else:
                    for i in range(len(w)):
                        for c in 'abcdefghijklmnopqrstuvwxyz':
                            tw = w[:i] + c + w[i + 1:]
                            if tw in words:
                                new_dict[tw] += [j + [tw] for j in dicts[w]]
            words -= set(new_dict.keys())
            dicts = new_dict

        return ans

    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums, ans = set(nums), 0
        while nums:
            n = nums.pop()
            n1, n2 = n - 1, n + 1
            while n1 in nums:
                nums.remove(n1)
                n1 -= 1
            while n2 in nums:
                nums.remove(n2)
                n2 += 1
            ans = max(ans, n2 - n1 - 1)
        return ans

    def longestConsecutive_(self, nums):
        nums, ans = set(nums), 0
        for n in nums:
            if n - 1 not in nums:
                nb = n + 1
                while nb in nums:
                    nb += 1
                ans = max(ans, nb - n)
        return ans

    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        if not board:
            return
        row, col = len(board), len(board[0])
        border = [(i, j) for i in (0, row - 1) for j in range(col)] \
                 + [(i, j) for i in range(1, row) for j in (0, col - 1)]  # record all the borders
        while border:
            i, j = border.pop()
            if 0 <= i < row and 0 <= j < col and board[i][j] == 'O':
                board[i][j] = '*'
                border += [[i - 1, j], [i, j - 1], [i + 1, j], [i, j + 1]]  # DFS
        board[:] = [['XO'[x == '*'] for x in line] for line in board]

    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """
        l = len(ratings)
        ans = [1] * l
        for i in range(1, l):
            if ratings[i] > ratings[i - 1]:
                ans[i] = ans[i - 1] + 1
        for i in range(l - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                ans[i] = max(ans[i], ans[i + 1] + 1)
        return sum(ans)

    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: Set[str]
        :rtype: List[str]
        """

        def helper(s, memo):  # store traversed str to save time
            if not s:
                return []
            if s in memo:
                return memo[s]
            tmp = []
            for word in wordDict:
                if s.startswith(word):
                    if len(s) == len(word):
                        tmp.append(s)
                    else:
                        tmp += ([word + " " + x for x in helper(s[len(word):], memo)])
            memo[s] = tmp
            return tmp

        return helper(s, {})

    def evalRPN(self, tokens: list) -> int:
        operators = ['+', '-', '*', '/']
        stack = [0]
        for c in tokens:
            if c not in operators:
                stack += [int(c)]
            else:
                n1, n2 = stack.pop(), stack.pop()
                if c == '+':
                    stack += [n1 + n2]
                elif c == '-':
                    stack += [n2 - n1]
                elif c == '*':
                    stack += [n1 * n2]
                else:
                    stack += [int(n2 / n1)]
        return stack[-1]

    def maxProduct(self, nums: list) -> int:
        rev = nums[::-1]
        for i in range(1, len(nums)):
            nums[i] *= nums[i - 1] or 1
            rev[i] *= rev[i - 1] or 1
        return max(nums + rev)

    def maxProduct_(self, nums: list) -> int:
        m1, m2, ans = nums[0], nums[0], nums[0]
        for i in range(1, len(nums)):
            m1, m2 = max(nums[i], m1 * nums[i], m2 * nums[i]), \
                     min(nums[i], m1 * nums[i], m2 * nums[i])
            ans = max(ans, m1, m2)
        return ans

    def findMin(self, nums: list) -> int:
        if nums[0] <= nums[-1]:
            return nums[0]
        l, r = 0, len(nums) - 1
        while l < r:
            mid = l + (r - l) // 2
            if nums[mid] > nums[mid + 1]:
                return nums[mid + 1]
            elif nums[mid] < nums[mid - 1]:
                return nums[mid]
            elif nums[0] < nums[mid]:
                l = mid + 1
            else:
                r = mid - 1
        return -1

    def findMin_2(self, nums: list) -> int:
        l, r = 0, len(nums) - 1
        sigl, sigr = 0, 0
        while r - l > 1:
            if nums[l] > nums[r]:
                l = (l + r) // 2
                sigl, sigr = 1, 0
            elif nums[l] < nums[r]:
                l, r = max(l - (r - l), 0), l
                sigl, sigr = 0, 1
            else:
                if sigl == 0 and sigr == 1:
                    l = (l + r) // 2
                elif sigr | sigl == 0:
                    l += 1
                else:
                    l, r = max(l - (r - l), 0), l
        return min(nums[l], nums[r])

    def maximumGap(self, nums: list) -> int:
        if len(nums) < 2:
            return 0
        nums = sorted(nums)
        return max([nums[i] - nums[i - 1] for i in range(1, len(nums))])

    # Todo: buckets sorting.

    def findPeakElement(self, nums: list) -> int:
        # nums = [-float("inf")] + nums + [-float("inf")]
        # for i in range(1, len(nums) + 1):
        #     if nums[i - 1] < nums[i] and nums[i] > nums[i + 1]:
        #         return i - 1
        l, r = 0, len(nums) - 1
        while l < r:
            mid = l + (r - l) // 2
            if nums[mid] < nums[mid + 1]:  # exists a peak bigger than neighbor, or the peak is last num
                l = mid + 1
            else:
                r = mid
        return r

    def largestNumber(self, nums: list) -> str:
        def cmp(x, y):
            return x + y > y + x

        nums, ans = [str(x) for x in nums], ""
        for i in range(len(nums)):
            tmp = nums[i]
            for j in range(i + 1, len(nums)):
                if not cmp(tmp, nums[j]):
                    print(tmp, nums[j])
                    tmp, nums[j] = nums[j], tmp
            ans += tmp
            if i == 0 and tmp == '0':
                break
        return ans

    def calculateMinimumHP(self, dungeon: list) -> int:
        n1, n2, dp = len(dungeon), len(dungeon[0]), []
        for i in range(n1 + 1):
            dp += [[2 ** 31] * (n2 + 1)]
        dp[n1][n2 - 1] = dp[n1 - 1][n2] = 1
        for row in (dungeon[::-1]):
            n1 -= 1
            for i in range(n2)[::-1]:
                dp[n1][i] = max(min(dp[n1][i + 1], dp[n1 + 1][i]) - row[i], 1)
        return dp[0][0]

    def calculateMinimumHP_(self, dungeon: list) -> int:
        dp = [2 ** 31] * (len(dungeon[0]) - 1) + [1]
        for row in dungeon[::-1]:
            for i in range(len(dungeon[0]))[::-1]:
                # min(dp[i], dp[i+1]) -> maybe out of index
                dp[i] = max(min(dp[i:i + 2]) - row[i], 1)  # min life num in this pos(>=1)
        return dp[0]

    def maxProfit_(self, k: int, prices: list) -> int:
        if len(prices) == 0 or len(prices) == 1 or k == 0:
            return 0
        if k >= len(prices) / 2:
            return sum(i - j for i, j in zip(prices[1:], prices[:-1]) if i - j > 0)
        b, s = [prices[0]] * k, [0] * k
        for p in prices[1:]:
            for i in range(k):
                if i == 0:
                    b[0] = min(b[0], p)
                    s[0] = max(s[0], p - b[0])
                else:
                    b[i] = min(b[i], p - s[i - 1])
                    s[i] = max(s[i], p - b[i])
        return s[-1]

    def rangeBitwiseAnd(self, m: int, n: int) -> int:
        count, tmp, ans = 1, n, n
        while tmp >= m:
            ans &= tmp
            tmp = n - count
            count *= 2
        return ans & m

    def rangeBitwiseAnd_(self, m: int, n: int) -> int:
        while m < n:
            n &= (n - 1)
        return n

    def canFinish(self, numCourses: int, prerequisites: list) -> bool:
        graph, visit = [[] for _ in range(numCourses)], [0] * numCourses
        for i, j in prerequisites:
            graph[i].append(j)

        def dfs(i):
            if visit[i] == 1:  # has been visited
                return False
            visit[i] = 1
            for j in graph[i]:
                if not dfs(j):
                    return False
            visit[i] = 0  # set back to unvisited
            return True

        for i in range(numCourses):
            if not dfs(i):
                return False
        return True

    def minSubArrayLen(self, s: int, nums: list) -> int:
        cur_sum, begin, ans = 0, 0, len(nums) + 1
        for i, n in enumerate(nums):
            cur_sum += n
            while cur_sum >= s:
                ans = min(ans, i - begin + 1)
                cur_sum -= nums[begin]
                begin += 1
        return ans if ans <= len(nums) else 0

    def findOrderDFS(self, numCourses: int, prerequisites: list) -> list:
        pre, post = defaultdict(set), defaultdict(set)
        for i, j in prerequisites:
            pre[i].add(j)
            post[j].add(i)
        stack, ans = [i for i in range(numCourses) if not pre[i]], []  # the node to any position in the order
        while stack:
            node = stack.pop()
            ans.append(node)
            for i in post[node]:
                pre[i].remove(node)
                if not pre[i]:
                    stack.append(i)
            pre.pop(node)
        return ans if not pre else []  # return the order if stack and pre are both empty

    def rob(self, nums: list) -> int:
        if len(nums) == 1:
            return nums[0]

        def helper(i, j):
            s1 = s2 = 0
            for n in nums[i:j]:
                s1, s2 = max(s1, s2 + n), s1
            return s1

        return max(helper(0, -1), helper(1, len(nums)))

    def findKthLargest(self, nums: list, k: int) -> int:
        return sorted(nums)[len(nums) - k]

    def getSkyline(self, buildings: list) -> list:
        events = sorted([(l, -h, r) for l, r, h in buildings] + [(r, 0, 0) for _, r, _ in buildings])
        print(events)
        ans, hp = [[0, 0]], [(0, float('inf'))]
        for l, neg_h, r in events:
            while l >= hp[0][1]:
                heappop(hp)
            if neg_h:
                heappush(hp, (neg_h, r))
            if ans[-1][1] != -hp[0][0]:
                ans.append([l, -hp[0][0]])
        return ans[1:]

    def containsNearbyAlmostDuplicate(self, nums: list, k: int, t: int) -> bool:
        if t < 0:
            return False
        buckets, w = {}, t + 1
        for i, n in enumerate(nums):
            m = n // w
            if m in buckets or (m - 1 in buckets and abs(n - buckets[m - 1]) < w) or (
                    m + 1 in buckets and abs(n - buckets[m + 1]) < w):
                return True
            buckets[m] = n
            if i >= k:
                del buckets[nums[i - k] // w]
        return False

    def maximalSquareBruteForce(self, matrix: list) -> int:
        def isSquare(i, j, l):
            for ni in range(i, i + l):
                for nj in range(j, j + l):
                    if matrix[ni][nj] != "1":
                        return False
            return True

        if not matrix or not matrix[0]:
            return 0
        row, col, ans = len(matrix), len(matrix[0]), 0
        for i in range(row):
            for j in range(col):
                if matrix[i][j] == "1":
                    for l in range(ans + 1, min(row - i, col - j) + 1)[::-1]:
                        if isSquare(i, j, l):
                            ans = l
                            break
        return ans ** 2

    def maximalSquare(self, matrix: list) -> int:
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                matrix[i][j] = int(matrix[i][j])
                if matrix[i][j] and i and j:
                    matrix[i][j] = min(matrix[i - 1][j], matrix[i - 1][j - 1], matrix[i][j - 1]) + 1
        return max([max(row + [0]) for row in matrix] + [0]) ** 2

    def summaryRanges(self, nums: list) -> list:
        ans, start = [], 0
        for i, n in enumerate(nums):
            if (i < len(nums) - 1 and n + 1 != nums[i + 1]) or i == len(nums) - 1:  # not continuous or end to array
                if start == i:  # single number
                    ans.append(str(n))
                else:  # continuous numbers
                    ans.append(str(nums[start]) + "->" + str(n))
                start = i + 1
        return ans

    def majorityElement(self, nums: list) -> list:
        # Boyer Moore majority algorithm
        count1, count2, candidate1, candidate2 = 0, 0, 0, 1
        for n in nums:
            if n == candidate1:
                count1 += 1
            elif n == candidate2:
                count2 += 1
            elif count1 == 0:
                count1, candidate1 = 1, n
            elif count2 == 0:
                count2, candidate2 = 1, n
            else:  # not the candidates
                count1, count2 = count1 - 1, count2 - 1
        return [n for n in (candidate1, candidate2) if nums.count(n) > len(nums) // 3]

    def productExceptSelf(self, nums: list) -> list:
        count = 0
        for n in nums:
            if n == 0:
                count += 1
        if count > 1: return [0] * len(nums)
        product = reduce(lambda x, y: x * y or x or y, nums)
        count = 1 if count == 0 else 0
        return [product if n == 0 else product * count // n for n in nums]

    def maxSlidingWindow(self, nums: list, k: int) -> list:
        pos, ans = deque(), []  # pos to store the bigger nums' position, the num smaller than another traversed num will be throwed.
        for i, n in enumerate(nums):
            while pos and nums[pos[-1]] < n:
                pos.pop()
            pos += [i]
            if pos[0] <= i - k:  # k size window
                pos.popleft()
            if i >= k - 1:
                print([nums[p] for p in pos])
                ans.append(nums[pos[0]])
        return ans

    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix or not matrix[0]:
            return False
        row, col, tmp = len(matrix) - 1, len(matrix[0]), 0
        while row >= 0 and tmp < col:
            if matrix[row][tmp] == target:
                return True
            elif matrix[row][tmp] > target:
                row -= 1
            else:
                tmp += 1
        return False

    def searchMatrix_(self, matrix, target):
        j = -1
        for row in matrix:
            if not row:
                return False
            while j + len(row) > 0 and row[j] > target:
                j -= 1
            if row[j] == target:
                return True
        return False

    def singleNumber(self, nums: list) -> list:
        sets = set()
        for n in nums:
            if n not in sets:
                sets.add(n)
            else:
                sets.remove(n)
        return list(sets)

    def hIndex(self, citations: list) -> int:
        citations.sort()
        n = len(citations)
        for i in range(n):
            if citations[i] >= n - i:
                return n - i
        return 0

    def hIndex_2(self, citations: list) -> int:
        n = len(citations)
        for i in range(n):
            if citations[i] >= n - i:
                return n - i
        return 0

    def findDuplicate(self, nums: list) -> int:
        # return sum(nums) - sum(range(1, len(nums))) # only find num that duplicated once
        for i in range(len(nums)):
            if nums[i] == i or nums[i] == -1:
                continue
            tmp, nums[i] = nums[i], -1
            while True:
                if nums[tmp] == tmp:
                    return tmp
                t = nums[tmp]
                nums[tmp], tmp = tmp, t
                if tmp == -1:
                    break
        return 0

    def gameOfLife(self, board: list) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board or not board[0]:
            return
        row, col = len(board), len(board[0])
        for i in range(row):
            for j in range(col):
                count = 0
                for xi, xj in [[-1, -1], [-1, 0], [-1, 1], [0, 1], [0, -1], [1, -1], [1, 0], [1, 1]]:
                    ti, tj = i + xi, j + xj
                    if 0 <= ti < row and 0 <= tj < col and board[ti][tj] in (1, -2):
                        count += 1
                if board[i][j] == 0 and count == 3:
                    board[i][j] = -1  # to live
                elif board[i][j] == 1 and (count < 2 or count > 3):
                    board[i][j] = -2  # to die
        for i in range(row):
            for j in range(col):
                board[i][j] = 1 if board[i][j] == -1 else 0 if board[i][j] == -2 else board[i][j]
                # print(board)

    def lengthOfLIS(self, nums: list) -> int:
        tails, ans = [0] * len(nums), 0
        for n in nums:
            i, j = 0, ans
            while i != j:
                m = (i + j) // 2
                if tails[m] < n:
                    i = m + 1
                else:
                    j = m
            tails[i] = n  # to recover the one bigger than n but previous is smaller
            ans = max(i + 1, ans)
            # print(tails)
        return ans

    def removeInvalidParentheses_(self, s: str) -> list:
        def isvaild(x):
            cnt = 0
            for c in x:
                if c == '(':
                    cnt += 1
                elif c == ')':
                    cnt -= 1
                    if cnt < 0:
                        return False
            return cnt == 0

        level = {s}
        while True:
            valid = list(filter(isvaild, level))
            if valid:
                return valid
            level = {x[:i] + x[i + 1:] for x in level for i in range(len(x))}

    def removeInvalidParentheses(self, s: str) -> list:
        removed, l, r, results = 0, 0, 0, {s}
        for i, c in enumerate(s):  # remove redundant ")"
            if c == ")" and l == r:
                print(i)
                new_results = set()
                while results:
                    result = results.pop()
                    for j in range(i - removed + 1):
                        if result[j] == ")":
                            new_results.add(result[:j] + result[j + 1:])
                results = new_results
                removed += 1
            elif c == "(":
                l += 1
            elif c == ")":
                r += 1
        l, r, i, ll = 0, 0, len(s), len(s) - removed
        for ii in range(ll - 1, -1, -1):  # remove redundant "("
            i -= 1
            c = s[i]
            if c == "(" and l == r:
                new_results = set()
                while results:
                    result = results.pop()
                    for j in range(ii, ll):
                        if result[j] == "(":
                            new_results.add(result[:j] + result[j + 1:])
                results = new_results
                ll -= 1
            elif c == "(":
                l += 1
            elif c == ")":
                r += 1
        return list(results)

    def maxCoins(self, nums: list) -> int:
        n = len(nums) + 2
        nums, dp = [1] + nums + [1], [[0] * n for _ in range(n)]
        # nums: image nums[-1] = nums[n] = 1
        # dp[i][j]: maxCoins from nums[i] to nums[j]
        for gap in range(2, n):  # the interval
            for i in range(n - gap):  # i is the left, j is the right one
                j = i + gap
                for k in range(i + 1, j):  # k is the one to be burst
                    dp[i][j] = max(dp[i][j], nums[i] * nums[k] * nums[j] + dp[i][k] + dp[k][j])
        return dp[0][n - 1]

    def countSmaller(self, nums: list) -> list:
        def sort(enum):  # from bottom to up
            # l and r are partly sorted and have counted the smaller in the num self's part;
            # compare l and r, count the smaller from the other part.
            mid = len(enum) // 2
            if mid:
                l, r = sort(enum[:mid]), sort(enum[mid:])
                for i in range(len(enum))[::-1]:  # each turn pop the largest one
                    if not r or l and l[-1][1] > r[-1][1]:
                        ans[l[-1][0]] += len(r)  # add the count of smaller from right part
                        enum[i] = l.pop()
                    else:
                        enum[i] = r.pop()
            return enum

        ans = [0] * len(nums)
        sort(list(enumerate(nums)))
        return ans

    def maxProduct318(self, words: list) -> int:
        # chars, size, ans = [set(w) for w in words], [len(w) for w in words], 0
        # for i in range(len(words) - 1):
        #     for j in range(i, len(words)):
        #         if not chars[i].intersection(chars[j]):
        #             ans = max(ans, size[i] * size[j])
        # return ans

        d = {}
        for word in words:
            mask = 0  # 26-bit stores the char-set of word, 1 presents containing the char
            for c in set(word):
                mask |= (1 << (ord(c) - 97))  # "or" to get the final num that presents char-set
                # print(word, bin(mask))
            d[mask] = max(d.get(mask, 0), len(word))  # stores the max-length
        return max([d[i] * d[j] for i in d for j in d if not i & j] or [0])  # if i&j: the two set intersects

    def maxProduct318_(self, words: list) -> int:
        d = {}
        for word in words:
            mask = reduce(operator.or_, [1 << (ord(c) - 97) for c in set(word)] + [0])
            d[mask] = max(d.get(mask, 0), len(word))
        return max([d[i] * d[j] for i in d for j in d if not i & j] + [0])

    def coinChange(self, coins: list, amount: int) -> int:
        # dp
        MAX = sys.maxsize
        dp = [0] + [MAX] * amount
        for i in range(1, amount + 1):
            dp[i] = min([dp[i - c] if i - c >= 0 else MAX for c in coins]) + 1
        # return dp[-1] if dp[-1] < MAX else -1
        return [dp[-1], -1][dp[-1] >= MAX]

    def wiggleSort(self, nums: list) -> None:
        # cheating, it's better to find a median by O(n) algorithm, then wiggle sort
        nums.sort()
        half = len(nums[::2]) - 1
        nums[::2], nums[1::2] = nums[half::-1], nums[:half:-1]

    def longestIncreasingPath(self, matrix: list) -> int:
        # increasing path wont't be repeated, so it's no need to change matrix to record traversed pos.
        # use dp to record the longest increasing path of pos[i][j]
        if not matrix or not matrix[0]:
            return 0
        h, w = len(matrix), len(matrix[0])
        dp = [[0] * w for i in range(h)]

        def helper(i, j):
            if not dp[i][j]:
                v = matrix[i][j]  # read once
                dp[i][j] = 1 + max(helper(i - 1, j) if i > 0 and v < matrix[i - 1][j] else 0,
                                   helper(i, j - 1) if j > 0 and v < matrix[i][j - 1] else 0,
                                   helper(i + 1, j) if i + 1 < h and v < matrix[i + 1][j] else 0,
                                   helper(i, j + 1) if j + 1 < w and v < matrix[i][j + 1] else 0)
                # can be reduced to one-line but condition process would be more
            return dp[i][j]

        return max(helper(x, y) for x in range(h) for y in range(w))

    def increasingTriplet(self, nums: list) -> bool:
        if len(nums) < 3:
            return False
        # n1,n2 records the increasing two nums
        # m records the min nums which may be followed with bigger num.
        n1, n2, m, tag = 0, 0, nums[0], False
        for x in nums[1:]:
            # print(n1, n2, m)
            if tag and x > n2:
                return True
            elif x > m and (not tag or x < n2):
                n1, n2 = m, x
                tag = True
            elif x < m:
                m = x
        return False

    def findItinerary(self, tickets: list) -> list:
        dic = defaultdict(list)
        for fr, to in sorted(tickets)[::-1]:  # pop the smallest first
            dic[fr].append(to)
        ans = []

        def dfs(port):
            # print(port, dic[port])
            while dic[port]:
                dfs(dic[port].pop())
            ans.append(port)

        dfs("JFK")
        return ans[::-1]

    def isValidSerialization(self, preorder: str) -> bool:
        match = 1  # there's n non-null nodes and (n+1) null nodes for each tree
        for p in preorder.split(","):
            if match == 0:  # more null nodes than required
                return False
            if p == "#":
                match -= 1
            else:
                match += 1
        return match == 0

    def countRangeSum(self, nums: list, lower: int, upper: int) -> int:
        pre = [0]
        for n in nums:
            pre.append(pre[-1] + n)

        def sort(low, high):
            mid = (low + high) // 2
            if mid == low:
                return 0
            cnt = sort(low, mid) + sort(mid, high)
            i = j = mid
            for n in pre[low:mid]:
                while i < high and pre[i] - n < lower: i += 1
                while j < high and pre[j] - n <= upper: j += 1
                cnt += j - i
            pre[low:high] = sorted(pre[low:high])
            return cnt

        return sort(0, len(pre))

    def palindromePairs(self, words: list) -> list:
        def isPalindrome(word):
            return word == word[::-1]

        words = {w: i for i, w in enumerate(words)}
        ans = []
        for w, k in words.items():
            n = len(w)
            for i in range(n + 1):
                pre, suf = w[:i], w[i:]
                if isPalindrome(pre):
                    need = suf[::-1]
                    if need != w and need in words:
                        ans.append([words[need], k])
                if i != n and isPalindrome(suf):  # i!= n: prevent repeating
                    need = pre[::-1]
                    if need != w and need in words:
                        ans.append([k, words[need]])
        return ans

    def minPatches(self, nums: list, n: int) -> int:
        # miss: the smallest sum in [0, n]
        # i: the index of pos that has traversed
        miss, ans, i = 1, 0, 0
        while miss <= n:
            if i < len(nums) and nums[i] <= miss:
                miss += nums[i]  # the smallest sum can be reached
                i += 1
            else:
                print(miss)
                miss += miss  # add miss
                ans += 1
        return ans

    def maxEnvelopes(self, envelopes: list) -> int:
        # O(nlgn):
        if not envelopes:
            return 0
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        # envelopes.sort(key=cmp_to_key(lambda x, y: -1 if x[0] < y[0] or (x[0] == y[0] and x[1] > y[1]) else 1))
        # print(envelopes)
        lis = [envelopes[0][1]]  # record h: decreasing sort, first the max h of same width
        for w, h in envelopes[1:]:
            j = bisect.bisect_left(lis, h)
            if j >= len(lis):
                lis.append(h)
            else:
                lis[j] = h
        return len(lis)
        # # Dynamic programming: O(n^2)
        # dp = [1] * len(envelopes)
        # for i in range(len(envelopes) - 1):
        #     for j in range(i + 1, len(envelopes)):
        #         if envelopes[i][0] < envelopes[j][0] and envelopes[i][1] < envelopes[j][1]:
        #             dp[j] = max(dp[j], dp[i] + 1)
        # return max(dp)

    def largestDivisibleSubset(self, nums: list) -> list:
        # dynamic programming but in low speed
        if not nums: return []
        nums.sort()
        # dp: the max-size subset
        # pre: the pre num's index
        dp, pre = [1] * len(nums), [-1] * len(nums)
        for i, n in enumerate(nums):
            for j in range(i):
                if n % nums[j] == 0 and 1 + dp[j] > dp[i]:
                    dp[i], pre[i] = 1 + dp[j], j
        ans, idx = [], dp.index(max(dp))
        while idx != -1:
            ans.append(nums[idx])
            idx = pre[idx]
        return ans[::-1]

    def largestDivisibleSubset_(self, nums: list) -> list:
        if not nums:
            return []
        nums.sort()
        dp = {}

        max_id_ = nums[0]
        max_len = 1

        for idx, n in enumerate(nums):
            max_id = -1
            cur_len = 1

            for b in range(1, int(n ** 0.5) + 1):
                if n % b == 0:
                    if b in dp and cur_len < 1 + dp[b][1]:
                        cur_len = 1 + dp[b][1]
                        max_id = b
                    if n // b in dp and cur_len < 1 + dp[n // b][1]:
                        cur_len = 1 + dp[n // b][1]
                        max_id = n // b
            if max_id == -1:
                dp[n] = [n, 1]
            else:
                dp[n] = [max_id, 1 + dp[max_id][1]]
                if dp[n][1] > max_len:
                    max_id_ = n
                    max_len = dp[n][1]

        cur_id = max_id_
        ans = [cur_id]
        while dp[cur_id][0] != cur_id:
            cur_id = dp[cur_id][0]
            ans = [cur_id] + ans
        return ans

    def kSmallestPairs(self, nums1: list, nums2: list, k: int) -> list:
        heap = []  # smallests heap: push -(n1+n2)
        for x in nums1:
            for y in nums2:
                if len(heap) < k:
                    heappush(heap, (-x - y, [x, y]))
                else:
                    if heap and heap[0][0] < -x - y:
                        heappop(heap)
                        heappush(heap, (-x - y, [x, y]))
                    else:
                        break  # sorted nums: the inner loop will all satisfy (-heap[0][0] > n1 + n2)
        return [heappop(heap)[1] for _ in range(k) if heap]

    def wiggleMaxLength(self, nums: list) -> int:
        dp = [[1, 1] for _ in range(len(nums))]
        for i, n in enumerate(nums):
            for j in range(i):
                if nums[j] < n:
                    dp[i][0] = max(dp[i][0], 1 + dp[j][1])
                elif nums[j] > n:
                    dp[i][1] = max(dp[i][1], 1 + dp[j][0])
        return max([i if i > j else j for i, j in dp]) if dp else 0

    def wiggleMaxLength_(self, nums: list) -> int:
        if len(nums) < 2: return len(nums)
        inc, dec = 1, 1
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                inc = dec + 1
            elif nums[i] < nums[i - 1]:
                dec = inc + 1
        return max(inc, dec)

    def kthSmallest_(self, matrix: list, k: int) -> int:
        if not matrix or not matrix[0]: return 0
        ans = matrix[0][0]
        wid = len(matrix[0])
        hei = len(matrix)
        idx = [0] * wid
        num = [n for n in matrix[0]]
        ma = matrix[-1][-1]
        while k != 0:
            tmp, index = ma, -1
            for i in range(wid):
                if idx[i] < hei and num[i] <= tmp:
                    index = i
                    tmp = num[i]
            ans = tmp
            idx[index] += 1
            if idx[index] < hei:
                num[index] = matrix[idx[index]][index]
            k -= 1

        return ans

    def kthSmallest(self, matrix: list, k: int) -> int:
        lo = matrix[0][0]
        m = len(matrix)
        n = len(matrix[0])
        hi = matrix[m - 1][n - 1]
        while lo < hi:
            mid = (lo + hi) // 2
            count = 0
            j = n - 1
            i = 0
            while j >= 0 and i < n:
                if matrix[i][j] <= mid:  # 小于当前值的个数
                    count += (j + 1)
                    i += 1
                else:
                    j -= 1
            if count < k:  # 若小于当前值的个数小于k 当前值不够大, 区间右移
                lo = mid + 1
            else:
                hi = mid
        return lo

    def maxRotateFunction(self, A: list) -> int:
        l = len(A)
        if l < 2: return 0
        ans = sum([i * A[i] for i in range(1, l)])
        s, last = sum(A), ans
        for i in range(1, l):
            last += s - l * A[l - i]
            ans = max(ans, last)
        return ans

    def integerReplacement(self, n: int) -> int:
        cnt = 0
        while n > 1:
            cnt += 1
            if n % 2 == 0:  # make division
                n //= 2
            elif n % 4 == 1 or n == 3:  # n-1: to be multiple of 4 for consecutive divisions; 3 is the exception
                n -= 1
            else:  # n+1: to be multiple of 4 for consecutive divisions
                n += 1
        return cnt

    def calcEquation(self, equations: list, values: list, queries: list) -> list:
        def add_edge(a, b, v):
            if a not in graph:
                graph[a] = [(b, v)]
            else:
                graph[a].append((b, v))

        def query(a, b):  # BFS
            if a not in graph or b not in graph:
                return -1.0
            stack = deque([(a, 1.0)])
            visited = set()
            while stack:
                n, curv = stack.popleft()
                if n == b:
                    return curv
                visited.add(n)
                for x, v in graph[n]:
                    if x not in visited:
                        stack.append((x, curv * v))
            return -1.0

        graph = {}
        for [a, b], v in zip(equations, values):
            add_edge(a, b, v)
            add_edge(b, a, 1 / v)
        return [query(a, b) for [a, b] in queries]

    def reconstructQueue(self, people: list) -> list:
        # (h,k) sort from higher to shorter h, smaller to bigger k
        people, ans = sorted(people, key=lambda x: (-x[0], x[1])), []
        for p in people:
            ans.insert(p[1], p)  # insert p according to p[1]
        return ans

    def canCross(self, stones: list) -> bool:
        target = stones[-1]
        # set() has fast travesal speed than list
        # failed: records the failed pos and step pair
        # travesal: pos and step pair to travesal
        stones, failed, travesal = set(stones), set(), [(0, 0)]
        while travesal:
            pos, step = travesal.pop()
            for s in (step - 1, step, step + 1):
                p = pos + s
                if s > 0 and p in stones and (p, s) not in failed:
                    if p == target: return True
                    travesal.append((p, s))
            failed.add((pos, step))
        return False

    def trapRainWater(self, heightMap: list) -> int:
        if not heightMap or not heightMap[0]:
            return 0
        h, w = len(heightMap), len(heightMap[0])
        heap, visited = [], [[0] * w for _ in range(h)]

        for i in range(h):
            for j in range(w):
                if i == 0 or j == 0 or i == h - 1 or j == w - 1:
                    heappush(heap, (heightMap[i][j], i, j))
                    visited[i][j] = 1

        ans = 0
        while heap:
            height, i, j = heappop(heap)
            for x, y in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= x < h and 0 <= y < w and not visited[x][y]:
                    ans += max(0, height - heightMap[x][y])
                    heappush(heap, (max(heightMap[x][y], height), x, y))
                    visited[x][y] = 1
        return ans

    def numberOfArithmeticSlices(self, A: list) -> int:
        if len(A) < 3:  # the length of arithmetic slices should be greater than 3
            return 0

        ans, l = 0, len(A) - 1
        last, start, step = A[1], 0, A[1] - A[0]
        for i, n in enumerate(A[2:], 2):
            if n - last == step:
                last = n
                if i == l:  # the last num
                    ans += (i - start) * (i - start - 1) // 2  # the num of sub slices meet the requirement
            else:
                ans += (i - start - 1) * (i - start - 2) // 2
                step = n - last
                last, start = n, i - 1
        return ans

    def canPartition(self, nums: list) -> bool:
        # reachSum = {0}
        # for n in nums:
        #     reachSum.update({v + n for v in reachSum})
        # return sum(nums) / 2 in reachSum
        def helper(rest, idx):
            if rest == 0:
                return True
            if rest < 0 or idx < 0 or nums[idx] > rest:  # if nums[idx]>rest: unreachable
                return False
            return helper(rest - nums[idx], idx - 1) or helper(rest, idx - 1)  # use nums[idx] or not

        s = sum(nums)
        if s % 2 == 1:
            return False
        return helper(s / 2, len(nums) - 1)

    def pacificAtlantic(self, matrix: list) -> list:
        if not matrix or not matrix[0]:
            return []
        h, w = len(matrix), len(matrix[0])

        def helper(reached):  # BFS for single ocean
            q = list(reached)
            while q:
                i, j = q.pop()
                for x, y in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if 0 <= x < h and 0 <= y < w and (x, y) not in reached and matrix[x][y] >= matrix[i][j]:
                        reached.add((x, y))
                        q.append((x, y))
            return reached

        # return the intersection
        return list(helper(set([(0, i) for i in range(w)] + [(i, 0) for i in range(1, h)])) &
                    helper(set([(h - 1, i) for i in range(w)] + [(i, w - 1) for i in range(h - 1)])))

    def countBattleships(self, board: list) -> int:
        if not board or not board[0]: return 0
        ans = 0
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 'X' and (i == 0 or board[i - 1][j] == '.') and (j == 0 or board[i][j - 1] == '.'):
                    ans += 1
        return ans

    def eraseOverlapIntervals(self, intervals: list) -> int:
        intervals.sort()
        count = 0
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0]:
                intervals[i + 1][1] = min(intervals[i][1],
                                          intervals[i + 1][1])  # the first element is unrelated in next step
                count += 1
        return count

    def findRightInterval(self, intervals: list) -> list:
        # l, ans = sorted((e[0], i) for i, e in enumerate(intervals)), []
        # for _, end in intervals:
        #     r = bisect.bisect_left(l, (end,))
        #     ans.append(l[r][1] if r < len(l) else -1)
        # return ans
        l = sorted([(e[0], i) for i, e in enumerate(intervals)]) + [(float('inf'), -1)]
        return [l[bisect.bisect_left(l, (end,))][1] for _, end in intervals]

    def findDuplicates(self, nums: list) -> list:  # without extra space and O(n) time
        ans = []
        for n in nums:
            if nums[abs(n) - 1] < 0:
                ans.append(abs(n))
            else:
                nums[abs(n) - 1] *= -1  # tag that n has been visited
        return ans

    def lexicalOrder(self, n: int) -> list:
        ans = [1]
        while len(ans) < n:
            new = ans[-1] * 10
            while new > n:
                new = new // 10 + 1
                while new % 10 == 0:
                    new //= 10

        return ans

    def findKthNumber(self, n: int, k: int) -> int:
        ans, k = 1, k - 1
        while k > 0:
            count, interval = 0, [ans, ans + 1]
            while interval[0] <= n:  # to compute next interval less than n
                count += min(n + 1, interval[1]) - interval[0]
                interval = [10 * interval[0], 10 * interval[1]]
            if k >= count:
                ans += 1  # skip this interval
                k -= count
            else:
                ans *= 10  # increase one bit for a smaller interval
                k -= 1
        return ans

    def findMinArrowShots(self, points: list) -> int:
        points = sorted(points, key=lambda x: x[1])
        ans, end = 0, -float('inf')
        for x, y in points:  # greedy
            if x > end:
                ans += 1
                end = y
        return ans

    def fourSumCount(self, A: list, B: list, C: list, D: list) -> int:
        # O(n^2), Counter is slower
        # AB = Counter([a + b for a in A for b in B])
        # return sum(AB[-c - d] for c in C for d in D)
        # A, B, C, D = Counter(A), Counter(B), Counter(C), Counter(D)
        aA, bB, cC, dD = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
        for i in range(len(A)):
            aA[A[i]] += 1
            bB[B[i]] += 1
            cC[C[i]] += 1
            dD[D[i]] += 1
        AB = defaultdict(int)
        for a in aA:
            for b in bB:
                AB[a + b] += aA[a] * bB[b]
        ans = 0
        for c in cC:
            for d in dD:
                ans += AB[-c - d] * cC[c] * dD[d]
        return ans

    def find132pattern(self, nums: list) -> bool:
        # # O(n^2) time limit exceeded
        # dic = {}
        # for i, x in enumerate(nums[:-2]):
        #     for j, y in enumerate(nums[i + 1:-1], i+1):
        #         if x < y and ((j, y) not in dic or dic[(j, y)] > x):
        #             dic[(j,y)] = x
        # for (pos, mid), fir in dic.items():
        #     for x in nums[pos + 1:]:
        #         if fir < x < mid:
        #             return True
        # return False

        if len(set(nums)) < 3: return False
        stack, cmin = [[nums[0], nums[0]]], nums[0]
        for i, n in enumerate(nums[1:], 1):
            print(stack)
            if n < cmin:
                stack.append([n, n])  # the min num as first of 132
                cmin = n
            elif n >= stack[0][1]:  # the mid num is better to be larger
                stack = [[cmin, n]]
            elif n == cmin:
                continue
            else:
                while stack and n > stack[-1][0]:
                    if n < stack[-1][1]:
                        return True
                    else:
                        stack.pop()  # current interval is more strict than [cur_min, cur]
                stack.append([cmin, n])  # current least strict interval
        return False

    def minMoves2(self, nums: list) -> int:
        median = sorted(nums)[len(nums) // 2]
        return sum([abs(median - x) for x in nums])

    def circularArrayLoop(self, nums: list) -> bool:
        # l = len(nums)
        # for i in range(l):
        #     tmp, count, dir = i, 0, nums[i] // abs(nums[i])
        #     while tmp >= i and count < l:
        #         tmp = (tmp + nums[tmp]) % l
        #         if nums[tmp] % l == 0 or nums[tmp] * dir < 0:
        #             break
        #         elif tmp == i:
        #             return True
        #         count += 1
        # return False
        l, failed = len(nums), set()
        for i in range(l):
            if i in failed:
                continue
            seen, direc = set(), 1 if nums[i] > 0 else -1
            while nums[i] * direc > 0:
                nex = (i + nums[i]) % l
                if i == nex:  # nums[i] % l == 0
                    break
                if nex in seen:
                    return True
                else:
                    seen.add(nex)
                i = nex
            failed |= seen
        return False

    def findSubstringInWraproundString(self, p: str) -> int:
        if not p:
            return 0
        match, dic = {chr(i): chr(i + 1) if i != 122 else chr(i - 25) for i in range(97, 123)}, defaultdict(int)
        dic[p[0]], count = 1, 1
        for i, c in enumerate(p[1:], 1):
            if match[p[i - 1]] == c:
                count += 1
            else:
                count = 1
            dic[c] = max(dic[c], count)  # the longest substring ended with c
        print(dic)
        return sum(dic.values())
        # if not p:
        #     return 0
        # match = {chr(i): chr(i + 1) if i != 122 else chr(i - 25) for i in range(97, 123)}
        # dic, count, prev = {chr(i): 0 for i in range(97, 123)}, 1, p[0]
        # dic[prev] = 1
        #
        # for i in range(1, len(p)):
        #     c = p[i]
        #     if match[prev] == c:
        #         count += 1
        #     else:
        #         count = 1
        #     if count > dic[c]:
        #         dic[c] = count
        #     prev = c
        # print(dic)
        # return sum(dic.values())

    def PredictTheWinner(self, nums: list) -> bool:
        # def helper(s1, s2, s, e, tag):
        #     if s >= e:
        #         return s1 >= s2
        #     if tag:
        #         return helper(s1 + nums[s], s2, s + 1, e, False) or helper(s1 + nums[e], s2, s, e - 1, False)
        #     else:
        #         return helper(s1, s2 + nums[s], s + 1, e, True) and helper(s1, s2 + nums[e], s, e - 1, True)
        #
        # return helper(0, 0, 0, len(nums) - 1, True)
        if not nums: return True
        n = len(nums)
        if n & 1 == 0: return True
        dp = [0] * n
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if i == j:
                    dp[i] = nums[i]
                else:
                    dp[j] = max(nums[i] - dp[j], nums[j] - dp[j - 1])
        return dp[-1] >= 0

        # n = len(nums)
        # if n == 1 or n % 2 == 0: return True
        # dp = [[0] * n for _ in range(n)]
        # for i in range(n - 1, -1, -1):
        #     for j in range(i + 1, n):
        #         dp[i][j] = max(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1])
        # return dp[0][-1] >= 0

    def medianSlidingWindow(self, nums: list, k: int) -> list:
        # time complexity: O(n·logk)
        window, ans, odd_even = nums[:k - 1], [], k % 2
        window.sort()
        for i, n in enumerate(nums[k - 1:], k - 1):
            bisect.insort(window, n)
            ans.append((window[k // 2 - 1] + window[k // 2]) / 2 if not odd_even else window[k // 2])
            window.pop(bisect.bisect(window, nums[i + 1 - k]) - 1)  # pop the stale num
        return ans

    def findPoisonedDuration(self, timeSeries: list, duration: int) -> int:
        if not timeSeries: return 0
        pre, ans = timeSeries[0], duration
        for x in timeSeries[1:]:
            ans += duration if pre + duration <= x else x - pre
            pre = x
        return ans

    def reversePairs(self, nums: list) -> int:
        arr, ans = [], 0
        # for i, x in enumerate(nums):
        #     pos = bisect.bisect_right(arr, 2 * x)  # insert to pos: nums behind pos are greater then 2*x
        #     ans += i - pos
        #     bisect.insort(arr, x)
        # return ans

        for x in nums:
            ans += len(arr) - bisect.bisect_right(arr, 2 * x)
            idx = bisect.bisect_right(arr, x)
            arr[idx:idx] = [x]  # much faster than '''bisect.insort(arr, x)'''
        return ans

    def findSubsequences(self, nums: list) -> list:
        # 491. Increasing Subsequences
        ans = {()}  # set to eliminate duplicate
        for n in nums:
            ans |= {item + (n,) for item in ans if not item or n >= item[-1]}
        return [item for item in ans if len(item) > 1]

    def findTargetSumWays(self, nums: list, S: int) -> int:
        # 494. Target Sum
        # if not nums: return 0
        # dic = {nums[0]: 1, -nums[0]: 1} if nums[0] != 0 else {0: 2}
        # for n in nums[1:]:  # dp
        #     tmp = {}
        #     for m in dic:
        #         for x in [m + n, m - n]:
        #             tmp[x] = tmp.get(x, 0) + dic.get(m, 0)
        #     dic = tmp
        # return dic.get(S, 0)

        newt = (sum(nums) + S) // 2
        if sum(nums) < S or (sum(nums) + S) % 2 == 1:
            return 0
        dp = [0 for i in range(newt + 1)]
        dp[0] = 1
        for n in nums:
            for i in range(newt, n - 1, -1):
                dp[i] += dp[i - n]
        return dp[newt]

    def findDiagonalOrder(self, matrix: list) -> list:
        # 498. Diagonal Traverse
        if not matrix or not matrix[0]: return []
        m, n, dic, ans = len(matrix), len(matrix[0]), defaultdict(list), []
        for i in range(m):
            for j in range(n):
                dic[i + j].append(matrix[i][j])
        for k in dic:
            ans += (dic[k][::-1 if k % 2 == 0 else 1])
        return ans

    def findMaximizedCapital(self, k: int, W: int, Profits: list, Capital: list) -> int:
        from heapq import heapify, heappop, heappush
        projects = [[-Profits[i], Capital[i]] for i in range(len(Profits))]
        heapify(projects)
        tmp = []
        while k > 0 and projects:  # use heap to reduce the complexity while same projects have been choosed
            while projects:
                p, c = heappop(projects)
                if c <= W:
                    W -= p
                    k -= 1
                    while tmp:
                        heappush(projects, tmp.pop())
                    break
                else:
                    tmp.append([p, c])
        return W

    def nextGreaterElements(self, nums: list) -> list:
        # to improve: start from the max number
        stack, l, ans = [], len(nums), [-1] * len(nums)
        for i in range(l * 2):  # to traverse circularly
            print(stack)
            while stack and nums[stack[-1]] < nums[i % l]:  # the last one must be smallest in stack
                ans[stack[-1]] = nums[i % l]
                stack.pop()
            if i < len(nums):
                stack.append(i)
        return ans

    def findMinMoves(self, machines: list) -> int:
        # 517. Super Washing Machines
        if not machines: return 0
        s, n = sum(machines), len(machines)
        if s % n != 0: return -1
        targ, left, right, ans = s // n, 0, s, 0
        for i, m in enumerate(machines):
            right -= m
            l, r = left - i * targ, right - s + i * targ + targ
            left += m
            m -= targ
            if l * r >= 0:
                # need l+r dresses for max(l,r) times, or put into left and right for m times
                ans = max(l, r, ans) if m < 0 else max(m, ans)
            elif l * m > 0:
                # l+m+r=0, right put/get abs(r) from left+mid
                ans = max(abs(r), ans)
            else:
                # left get/put abs(r) from right+mid
                ans = max(abs(l), ans)
        return ans

    def findMinMoves_(self, machines: list) -> int:
        # 517. Super Washing Machines
        if not machines: return 0
        s, n = sum(machines), len(machines)
        if s % n != 0: return -1
        targ, balance, ans = s // n, 0, 0
        for m in machines:
            balance += m - targ  # current unbalance of left and right -- the put times from global view
            ans = max(abs(balance), ans)
        return max(max(machines) - targ, ans)  # max(machines) - targ -- the put times from single machine's view

    def countArrangement(self, N: int) -> int:
        # 526. Beautiful Arrangement
        # divs = [set() for i in range(N + 1)]
        # for i in range(1, N + 1):
        #     for j in range(1, N + 1):
        #         if i % j == 0 or j % i == 0:
        #             divs[i].add(j)
        visited = {}

        def helper(rest, n):
            if n == 1:
                return 1
            if (rest, n) in visited:
                return visited[(rest, n)]
            ans = sum(helper(rest[:i] + rest[i + 1:], n - 1) for i, x in enumerate(rest) if x % n == 0 or n % x == 0)
            # ans = sum(helper(rest[:i] + rest[i + 1:], n - 1) for i, x in enumerate(rest) if x in divs[n])
            visited[(rest, n)] = ans
            return ans

        return helper(tuple(range(1, N + 1)), N)

    def findMaxLength(self, nums: list) -> int:
        # 525. Contiguous Array
        records, ans, balance = {}, 0, 0
        for i, n in enumerate(nums):
            balance = balance + 1 if n == 0 else balance - 1
            if balance == 0:
                ans = i + 1
            elif balance in records:
                ans = max(ans, i - records[balance])  # the span satisfies requirement
            else:
                records[balance] = i  # records the **first** occurrence of balance
        return ans

    def singleNonDuplicate(self, nums: list) -> int:
        # 540. Single Element in a Sorted Array
        # O(n) time: xor / search sequentially
        # O(lg n) time: binary search
        # L = len(nums)
        #
        # def helper(l, r):
        #     if l <= r:
        #         mid = (l + r) // 2
        #         if (mid == 0 and nums[mid] != nums[mid + 1]) or (mid == L - 1 and nums[mid] != nums[mid - 1]) or (
        #                 nums[mid - 1] != nums[mid] and nums[mid] != nums[mid + 1]):
        #             return nums[mid]  # nums[mid] is the single number
        #         left = helper(l, mid - 1)
        #         if left: return left
        #         right = helper(mid + 1, r)
        #         if right: return right
        #     return 0
        #
        # if L == 1:
        #     return nums[0]
        # return helper(0, L - 1)
        return reduce(lambda x, y: x ^ y, nums)

    def optimalDivision(self, nums: list) -> str:
        # 553. Optimal Division
        # maximum: a/(b/c/d...) = a*c*d*.../b
        return "/".join(map(str, nums)) if len(nums) < 3 else f'{nums[0]}/({"/".join(map(str, nums[1:]))})'

    def subarraySum(self, nums: list, k: int) -> int:
        # 560. Subarray Sum Equals K
        # ans, sums = 0, [0]
        # for n in nums:
        #     sums.append(sums[-1] + n)
        # for i, s in enumerate(sums[1:], 1):
        #     for ps in sums[:i]:
        #         if s - ps == k:
        #             ans += 1
        # return ans
        ans, dic, s = 0, defaultdict(int), 0
        dic[0] = 1
        for n in nums:
            s += n
            ans += dic[s - k]
            dic[s] += 1
        return ans

    def arrayNesting(self, nums: list) -> int:
        # 565. Array Nesting
        dic = {}
        for i in range(len(nums)):
            if nums[i] == -1: continue
            idx, cnt = i, 0
            while nums[idx] != -1:
                nums[idx], idx, pre_idx, cnt = -1, nums[idx], idx, cnt + 1
            dic[i] = cnt + dic.get(pre_idx, 0)
        return max(dic.values())

    def leastBricks(self, wall: list) -> int:
        # 554. Brick Wall
        dic = defaultdict(int)
        for row in wall:
            s = 0
            for n in row[:-1]:
                s += n
                dic[s] += 1
        return len(wall) - max(dic.values()) if dic else len(wall)

    def checkSubarraySum(self, nums: list, k: int) -> bool:
        # 523. Continuous Subarray Sum
        # find two equal mods
        s, mods = 0, {}
        for i, n in enumerate(nums):
            if s not in mods:
                mods[s] = i
            s += n
            if k != 0:
                s %= k
            if s in mods and mods[s] < i:
                return True
        return False

    def triangleNumber(self, nums: list) -> int:
        # 611. Valid Triangle Number
        nums.sort()
        ans, L = 0, len(nums)
        for i in range(L - 2):
            for j in range(i + 1, L - 1):
                ans += bisect.bisect_left(nums, nums[i] + nums[j], lo=j + 1) - j - 1
        return ans

    def minDistance(self, word1: str, word2: str) -> int:
        # 583. Delete Operation for Two Strings
        l1, l2 = len(word1), len(word2)
        dp = [[0] * (l2 + 1) for _ in range(l1 + 1)]
        for i in range(1, l1 + 1):
            for j in range(1, l2 + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
        return l1 + l2 - dp[-1][-1] - dp[-1][-1]  # deleting chars not in common subsequence

    def leastInterval(self, tasks: list, n: int) -> int:
        # 621. Task Scheduler
        # schedule the most frequent tasks for (most-1) idles + (most-1) execution + last span for all frequent tasks
        ts = Counter(tasks)
        most = max(ts.values())
        return max((most - 1) * (n + 1) + list(ts.values()).count(most), len(tasks))

    def updateMatrix(self, matrix: list) -> list:
        # 542. 01 Matrix
        from collections import deque
        w, h = len(matrix[0]), len(matrix)

        def bfs(i, j):
            queue, visited = deque(), set()
            moves = [(-1, 0), (0, -1), (0, 1), (1, 0)]
            queue.append(((i, j), 0))
            while queue:
                cl = len(queue)
                for i in range(cl):  # traverse nodes with same distance
                    pos, dist = queue.popleft()
                    x, y = pos
                    if matrix[x][y] == 0:
                        return dist
                    visited.add(pos)
                    for move in moves:  # to traverse nodes for dist+1
                        nx, ny = x + move[0], y + move[1]
                        if 0 <= nx < h and 0 <= ny < w and (nx, ny) not in visited:
                            queue.append(((nx, ny), dist + 1))
            return -1

        for i in range(h):
            for j in range(w):
                if matrix[i][j] == 1:
                    matrix[i][j] = bfs(i, j)
        return matrix

    def findLongestChain(self, pairs: list) -> int:
        # 646. Maximum Length of Pair Chain
        pairs.sort(key=lambda x: x[1])
        ans, cur = 0, -sys.maxsize
        for x, y in pairs:
            if cur < x:
                ans += 1
                cur = y
        return ans
