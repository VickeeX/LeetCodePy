# -*- coding: utf-8 -*-

"""
    @File name    :    DynamicProgramming.py
    @Date         :    2020-01-07 16:51
    @Description  :    {TODO}
    @Author       :    VickeeX
"""

"""
    longest palindromic substring ✓，longest increasing subsequence ✓,
    longest common subsequence ✓, longest common substring ✓
    minimal editing cost ✓,
    other DP problems
"""


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


def longest_common_subsequence(s1: str, s2: str):
    """
    Original Method:
        dp[i][j] represents the longest common subsequence between s[:i+1] and s[:j+1]
        reversed traversal to get th LCS from dp
    Optimized Method:
        1. reduce dp from O(m*n) to O(n) [ O(m) is ok too ], iter dp in each round.
           need to use extra O(1) space to delay one step write/
        2. use records to avoid reverse traversal for LCS, extra max O(m*n) space to save O(m*n) time.

    Time complexity: O(m*n)
    Space complexity:  O(m*n) could be optimized to min( O(m), O(n) ) as extra O(m*n) time
    """
    if not s1 or not s2:
        return ""
    l1, l2 = len(s1), len(s2)
    idx1, idx2 = s1.index(s2[0]), s2.index(s1[0])
    dp, records = [0] * idx2 + [1] * (l2 - idx2), [""] * idx2 + [str(s1[0])] * (l2 - idx2)
    for i in range(1, l1):
        pre, cur = 0 if i < idx1 else 1, 0
        pre_s, cur_s = "" if i < idx1 else str(s2[0]), ""
        for j in range(1, l2):
            if pre < dp[j]:
                cur, cur_s = dp[j], records[j]
            else:
                cur, cur_s = pre, pre_s

            # cur = max(pre, dp[j])
            if s1[i] == s2[j] and dp[j - 1] + 1 > cur:
                cur, cur_s = dp[j - 1] + 1, records[j - 1] + str(s1[i])
            dp[j - 1], pre = pre, cur  # delay one step writing dp[j-1] as cur use older values of that
            records[j - 1], pre_s = pre_s, cur_s
        dp[-1], records[-1] = cur, cur_s

    return records[-1]


def longest_common_string(s1: str, s2: str):
    """
    dp[i][j] represents the length of current longest common string ended with s1[i],s2[j]
    which means dp[i][j]=0 if  s1[i]!=s2[j]

    Time complexity: O(m*n)
    Space complexity:  O( m*n )
    """
    if not s1 or not s2:
        return ""
    l1, l2 = len(s1), len(s2)
    dp, ml, end = [[0] * l2 for _ in range(l1)], 0, -1  # ml is the max LCS now and ends with pos "end" of s1
    for i, c in enumerate(s1):  # compute the first column
        if s2[0] == c:
            dp[i][0], ml, end = 1, 1, i
    for i, c in enumerate(s2):  # compute the first row
        if s1[0] == c:
            dp[0][i], ml, end = 1, 1, 0
    for i in range(1, l1):
        for j in range(1, l2):
            if s1[i] == s2[j]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > ml:
                    ml, end = dp[i][j], i
    return s1[end - ml + 1:end + 1]


def longest_common_string_1(s1: str, s2: str):
    """
    dp[i][j] only relates with dp[i-1][j-1], the dp matrix could be split into m or n slants
    for each slant:
        dp[0][i], dp[1][i+1], ..., dp[k][i+k]
    or  dp[i][0], dp[i+1][1], ..., dp[i+k][k]
    ——> we could use only one var to record dp values for each slant
    ——> slants are unrelated, so the single var could be reused between them

    Time complexity: O(m*n)
    Space complexity:  O(1)
    """
    if not s1 or not s2:
        return ""
    l1, l2 = len(s1), len(s2)
    dp, ml, end = 0, 0, -1
    for i, j in [(x, 0) for x in range(l1)] + [(0, x) for x in range(l2)]:  # start position for slants
        for k in range(min(l1 - i, l2 - j)):  # or "while i<l1 and j<l2"
            dp = dp + 1 if s1[i + k] == s2[j + k] else 0
            if dp > ml:
                ml, end = dp, i + k
    return s1[end - ml + 1:end + 1]


def minimal_editing_cost(s1, s2, ic, dc, rc):
    """
    :param ic: inserting cost
    :param dc: deleting cost
    :param rc: replacing cost

    dp[i][j] represents the min editing cost from s1[:i] to s2[:j]
    Time complexity: O(m*n)
    Space complexity:  O(m*n)
    """
    l1, l2 = len(s1), len(s2)
    dp = [[ic * j if i == 0 else i * dc if j == 0 else -1 for j in range(l2 + 1)] for i in range(l1 + 1)]

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            dp[i][j] = min(dp[i - 1][j] + dc,
                           dp[i][j - 1] + ic,
                           dp[i - 1][j - 1] if s1[i - 1] == s2[j - 1] else dp[i - 1][j - 1] + rc)
    return dp[-1][-1]


def minimal_editing_cost_1(s1, s2, ic, dc, rc):
    """
    dp[j] in round i: represents the min editing cost from s1[:i+1] to s2[:j]
    Time complexity: O(m*n)
    Space complexity:  O(n), n is the length of s2; MARK: not min( O(m),O(n) )
    """
    l1, l2 = len(s1), len(s2)
    dp = [ic * i for i in range(l2 + 1)]

    for i in range(l1):  # to reduce a little code, here use range(l1), range(l2), note the indexes' change
        pre, cur = (i + 1) * dc, 0
        for j in range(l2):
            cur = min(dp[j + 1] + dc, pre + ic, dp[j] if s1[i] == s2[j] else dp[j] + rc)
            dp[j], pre = pre, cur
        dp[-1] = cur


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
