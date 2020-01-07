# -*- coding: utf-8 -*-

"""
    @File name    :    dynamic_programming.py
    @Date         :    2020-01-07 16:51
    @Description  :    {TODO}
    @Author       :    VickeeX
"""


class DynamicProgrammingSolution():

    def longestPalindrome(self, s):
        """ 5. Longest Palindromic Substring
            slow dp here
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

    def longestPalindrome_1(self, s):
        """ more smart than dp
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

    def isMatch(self, s, p):
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
