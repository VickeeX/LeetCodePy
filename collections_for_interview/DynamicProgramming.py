# -*- coding: utf-8 -*-

"""
    @File name    :    DynamicProgramming.py
    @Date         :    2020-01-07 16:51
    @Description  :    {TODO}
    @Author       :    VickeeX
"""


# TODO: longest palindromic substring ✓，longest increasing subsequence, longest continuous string,
#       longest common subsequence, longest common substring


def longest_palindrome(s):
    """
    note: slow dp here
    dp[i][j] represents if s[j:i+j] (str of length i from j) is palindrome
    """
    if not s:
        return ""
    dp, l = [], len(s)
    dp.append([0] * (l + 1))
    dp.append([1] * l)
    for i in range(2, l + 1):
        dp.append([])
        for j in range(l - i + 1):
            dp[i].append(2 + dp[i - 2][j + 1] if s[j] == s[j + i - 1] and (dp[i - 2][j + 1] != 0 or i == 2) else 0)

    # print(dp)
    for i in range(l, 0, -1):
        for j in range(l - i + 1):
            if dp[i][j] != 0:
                return s[j:i + j]


def longest_palindrome_1(s):
    """
    more smart than dp
    add 1 char each time, the longest length add 1 or 2 and  curLongestPalindrome ended with current char
    """
    if len(s) < 2 or s == s[::-1]:
        return s
    l, start = 1, 0
    for end in range(1, len(s)):
        # add two for length, ended at end pos
        if l + 1 <= end and s[end - l - 1:end + 1] == s[end - l - 1:end + 1][::-1]:
            start, l = end - l - 1, l + 2
            continue
        if l <= end and s[end - l:end + 1] == s[end - l:end + 1][::-1]:
            start, l = end - l, l + 1
    return s[start:start + l]


def reversed_generate_LIS(nums, dp):
    longest, idx = 0, -1
    for i, n in enumerate(dp):
        if n > longest:
            idx, longest = i, n

    ans, longest = [-1] * (longest - 1) + [nums[idx]], longest - 1
    for i in range(idx - 1, -1, -1):
        if nums[i] < nums[idx] and dp[i] == longest:
            longest, idx = longest - 1, i
            ans[longest] = nums[i]
    return ans


def longest_increasing_subsequence(nums):
    """
    dp[i] represents longest increasing subsequence ended with nums[i]
    then find the max pos, and generate whole LIS use reverse traversal
    Time complexity: O(n^2) = O(n^2) + O(n)
    Space complexity:  O(n)
    """
    if len(nums) < 2:
        return nums
    l = len(nums)
    dp = [1] * l
    for i in range(1, l):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    return reversed_generate_LIS(nums, dp)


def longest_increasing_subsequence_1(nums):
    """
    records[i] records the current LIS ended with nums[i] to reduce the reverse traversal
    Time complexity: O(n^2)
    Space complexity:
        avg: O(n^2); best: O(n); worst: O(n^2)
    """
    if len(nums) < 2:
        return nums
    l = len(nums)

    records, dp = [[n] for n in nums], [1] * l
    for i in range(1, l):
        idx = i
        for j in range(i - 1, -1, -1):
            if nums[i] > nums[j] and dp[j] + 1 > dp[i]:
                dp[i], idx = dp[j] + 1, j
        if idx != i:
            records[i] = records[idx] + [nums[i]]
    longest, idx = 0, -1
    for i, n in enumerate(dp):
        if n > longest:
            idx, longest = i, n
    return records[idx]


def longest_increasing_subsequence_2(nums):
    """
    use binary search to optimize the searching process
    Time complexity: O(n*lgn)
    Space complexity:  O(n)

    Mark: hard to understand here
    end[b] = c represents increasing subsequence of length b+1 is ended with num c ???
    """
    if len(nums) < 2:
        return nums
    lg = len(nums)
    dp, ends, right = [1] * lg, [nums[0]] + [0] * (lg - 1), 0
    for i in range(1, lg):
        l, r = 0, right
        while l <= r:
            m = (l + r) // 2
            if nums[i] > ends[m]:
                l = m + 1
            else:
                r = m - 1
        right, ends[l], dp[i] = max(right, l), nums[i], l + 1
    return reversed_generate_LIS(nums, dp)


def is_match(s, p):
    """ 10. Regular Expression Matching
    """
    if not p:
        return not s
    dp = [[False] * (len(p) + 1) for _ in range(len(s) + 1)]
    dp[0][0] = True
    for i, c in enumerate(p):
        if c == '*' and dp[0][i - 1]:
            dp[0][i + 1] = True
    for i in range(len(s)):
        for j in range(len(p)):
            if s[i] == p[j] or p[j] == '.':
                dp[i + 1][j + 1] = dp[i][j]
            elif p[j] == '*':
                if s[i] != p[j - 1] and p[j - 1] != '.':  # match zero
                    dp[i + 1][j + 1] = dp[i + 1][j - 1]
                else:
                    dp[i + 1][j + 1] = dp[i + 1][j - 1] or dp[i][j] or dp[i][j + 1]
    return dp[len(s)][len(p)]


def dp_coins(coins, aim):
    dp = []
    for i in range(len(coins)):
        dp.append([0] * (aim + 1))

    # dp[i][j]: ways to use coins[:i] reach j
    for k in range(coins[0], aim + 1, coins[0]):
        dp[0][k] = 1
    for i in range(len(coins)):
        dp[i][0] = 1
        for j in range(1, aim + 1):
            dp[i][j] = dp[i - 1][j] if coins[i] > j else dp[i - 1][j] + dp[i][j - coins[i]]
    return dp[len(coins) - 1][aim]
