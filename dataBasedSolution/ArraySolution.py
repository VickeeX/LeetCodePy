# -*- coding: utf-8 -*-

"""
    File name    :    ArraySolution
    Date         :    13/08/2018
    Description  :    {TODO}
    Author       :    VickeeX
"""

import math
from collections import defaultdict
from itertools import permutations, combinations


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
        return ans+[' '.join(cur).ljust(maxWidth)]

