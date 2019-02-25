# -*- coding: utf-8 -*-

"""
    File name    :    StringSolution
    Date         :    15/08/2018
    Description  :    {TODO}
    Author       :    VickeeX
"""

from collections import defaultdict, Counter, deque
from functools import reduce, lru_cache
import re, string


class StringSolution:
    def numJewelsInStones(self, J, S):
        """
        :type J: str
        :type S: str
        :rtype: int
        """
        return sum(s in J for s in S)

    def letterCasePermutation(self, S):
        """
        :type S: str
        :rtype: List[str]
        """
        if not S:
            return [""]
        tmp = self.letterCasePermutation(S[1:])
        if S[0].isalpha():
            return [S[0].lower() + s for s in tmp] + [S[0].upper() + s for s in tmp]
        return [S[0] + s for s in tmp]

    def rotateString(self, A, B):
        """
        :type A: str
        :type B: str
        :rtype: bool
        """
        if not A:
            return [""]
        return any(A[i:] + A[:i] == B for i in range(len(A)))

    def uniqueMorseRepresentations(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        code = [".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--",
                "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."]
        return len(set(map(lambda word: ''.join(code[ord(c) - 97] for c in word), words)))

    def numberOfLines(self, widths, S):
        """
        :type widths: List[int]
        :type S: str
        :rtype: List[int]
        """
        lines = used = 0
        for c in S:
            tmp = widths[ord(c) - 97]
            if used + tmp > 100:
                lines += 1
                used = tmp
            else:
                used += tmp
        return [lines + (used > 0), used]

    # ["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]
    def subdomainVisits(self, cpdomains):
        """
        :type cpdomains: List[str]
        :rtype: List[str]
        """
        countmap = defaultdict(int)
        for cpdomain in cpdomains:
            for i in range(len(cpdomain)):
                if cpdomain[i] == '.' or cpdomain[i] == ' ':
                    countmap[cpdomain[i + 1:]] += int(cpdomain.split()[0])
        return [str(v) + ' ' + k for (k, v) in countmap.items()]

    def mostCommonWord(self, paragraph, banned):
        """
        :type paragraph: str
        :type banned: List[str]
        :rtype: str
        """

        # words = re.sub('[\!\?\'\,;.]', '', paragraph).split()
        # count = Counter(words)
        # for ban in banned:
        #     del count[ban]
        words = paragraph.lower()
        for ban in banned:
            words = words.replace(ban, '')
        return Counter(re.sub('[\!\?\'\,;.]', '', words).split()).most_common(1)[0][0]

    def shortestToChar(self, S, C):
        """
        :type S: str
        :type C: str
        :rtype: List[int]
        """
        ans, last = [0x7FFFFFFF] * len(S), -0x7FFFFFFF
        for i in range(len(S)):  # "loveleetcode"
            if S[i] == C:
                last = i
            print(i, last, S[i], ans[i])
            ans[i] = min(ans[i], i - last)
        last = 0x7FFFFFFF
        for i in range(len(S) - 1, -1, -1):
            if S[i] == C:
                last = i
            print(i, last, S[i], ans[i])
            ans[i] = min(ans[i], last - i)
        return ans

    def toGoatLatin(self, S):
        """
        :type S: str
        :rtype: str
        """
        ans = ""
        for count, word in enumerate(S.split()):
            if word[0] not in "aeiouAEIOU":
                word = word[1:] + word[0]
            word = word + "ma" + "a" * (count + 1)
            ans += (" " + word)
        return ans[1:]

    # "abcdddeeeeaabbbcd"
    def largeGroupPositions(self, S):
        """
        :type S: str
        :rtype: List[List[int]]
        """
        start, ans = 0, []
        for idx, c in enumerate(S + "#"):
            if c != S[start]:
                if idx - start > 2:
                    ans.append([start, idx - 1])
                start = idx
        return ans

    def backspaceCompare(self, S, T):
        """
        :type S: str
        :type T: str
        :rtype: bool
        """

        def transform(s):
            ans = ""
            for i in s:
                if i == '#':
                    ans = ans[:-1]
                else:
                    ans += i
            return ans

        return transform(S) == transform(T)

    def buddyStrings(self, A, B):
        """
        :type A: str
        :type B: str
        :rtype: bool
        """
        if len(A) != len(B):
            return False
        fir, sec = [], []
        for i, c in enumerate(A):
            if c != B[i]:
                fir.append(c)
                sec.append(B[i])
        list.reverse(fir)
        if len(fir) == 2 and fir == sec:
            return True
        if A == B and len(set(A)) < len(A):
            return True
        return False

    def uncommonFromSentences(self, A, B):
        """
        :type A: str
        :type B: str
        :rtype: List[str]
        """
        d, ans = defaultdict(int), []
        for word in A.split() + B.split():
            d[word] += 1
        for word in d.keys():
            if d[word] == 1:
                ans.append(word)
        return ans

    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        newS = '$#' + '#'.join(s) + '#&'
        print(newS)
        id = mx = cur = 0
        ans, p = "", [0]
        for i in range(1, len(newS) - 1):
            if i < mx:
                p.append(min(p[2 * id - i], mx - i))
            else:
                p.append(1)
            while newS[i - p[i]] == newS[i + p[i]]:
                p[i] += 1
            if mx < i + p[i]:
                id, mx = i, i + p[i]
            if cur < p[i] - 1:
                cur = p[i] - 1
                print(i, cur)
                # ans = s[i // 2 - cur // 2:i // 2 + cur // 2] # i奇数, cur偶数
                # ans = s[i // 2 - (cur + 1) // 2:i // 2 + cur // 2]  # i偶数, cur奇数
                ans = s[i // 2 - (cur + 1) // 2:i // 2 + cur // 2]
        return ans

    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if numRows == 1 or len(s) <= numRows:
            return s
        ans = ["" for i in range(numRows)]
        line, dir = 0, 1  # dir: 0--down, 1--up; line: line
        for i in s:
            ans[line] += i
            line += dir
            if line == numRows:
                dir, line = -1, line - 2
            elif line == -1:
                dir, line = 1, 1
        return ''.join(ans)

    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if not digits:
            return []
        values, ans = ["", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"], [""]
        for i in digits:
            ans = [pre + "" + x for pre in ans for x in values[int(i)]]
        return ans

    # return str(int(num1) * int(num2))
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        if num1 == "0" or num2 == "0":
            return "0"
        ans = []
        for i in range(len(num1) + len(num2)):
            ans.append("0")
        for i in range(len(num1) - 1, -1, -1):
            for j in range(len(num2) - 1, -1, -1):
                s = (ord(num1[i]) - 48) * (ord(num2[j]) - 48) + (ord(ans[i + j + 1]) - 48) + (ord(ans[i + j]) - 48) * 10
                ans[i + j - 1] = chr(ord(ans[i + j - 1]) + s // 100)
                s %= 100
                ans[i + j], ans[i + j + 1] = chr(s // 10 + 48), chr(s % 10 + 48)

        if ans[0] == '0':
            return "".join(ans[1:])
        return "".join(ans)

    """
    solution:
        异构字符串的排序串都相等, 故将其元组形式作为字典索引.
    """

    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        anaM = defaultdict(list)
        for s in strs:
            anaM[tuple(sorted(s))].append(s)
        return list(anaM.values())

    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        paths, simplified = path.split("/"), []
        for i in paths:
            if not i or i == '.':
                continue
            if i == '..':
                if len(simplified) > 0:
                    simplified.pop()
            else:
                simplified.append(i)
        return '/' + '/'.join(simplified)

    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 2261
        last, lw, way = '', 0, int(s > '')
        for c in s:
            lw, way, last = way, (c > '0') * way + (9 < int(last + c) < 27) * lw, c
        return way

    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """

        def valid(sl):
            if len(sl) > 1 and sl[0] == '0':
                return False
            if 0 <= int(sl) < 256:
                return True
            return False

        size, ans = len(s), []
        for a in range(1, 5):
            for b in range(a + 1, a + 5):
                for c in range(b + 1, b + 5):
                    if 0 < size - c < 5:
                        if valid(s[:a]) and valid(s[a:b]) and valid(s[b:c]) and valid(s[c:]):
                            ans.append(s[:a] + "." + s[a:b] + "." + s[b: c] + "." + s[c:])
        return ans

    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        words, found, toTransformWords = set(wordList), {beginWord: 1}, deque()
        toTransformWords.append(beginWord)
        while toTransformWords:
            word = toTransformWords.popleft()
            for i in range(len(word)):
                for c in string.ascii_lowercase:
                    if c == word[i]:
                        continue
                    else:
                        new_word = word[:i] + str(c) + word[i + 1:]
                        if new_word in words and new_word not in found:
                            found[new_word] = found[word] + 1
                            if new_word == endWord:
                                return found[new_word]
                            toTransformWords.append(new_word)

        return 0

    def ladderLength1(self, beginWord, endWord, wordList):
        dictionary = set(wordList)
        if endWord not in dictionary:
            return 0
        forward, backward, n, r = {beginWord}, {endWord}, len(beginWord), 2
        while forward and backward:
            if len(forward) > len(backward):
                forward, backward = backward, forward

            next = set()
            for word in forward:
                for i, char in enumerate(word):
                    first, second = word[:i], word[i + 1:]
                    for item in string.ascii_lowercase:
                        candidate = first + item + second
                        if candidate in backward:
                            return r

                        if candidate in dictionary:
                            dictionary.discard(candidate)
                            next.add(candidate)
            forward = next
            r += 1
        return 0

    def partition(self, s):

        """
        :type s: str
        :rtype: List[List[str]]
        """
        if not s:
            return []
        res = []
        self.dfs(s, [], res)
        return res

    def dfs(self, s, path, res):
        if not s:
            res.append(path)
            return
        for i in range(len(s)):
            if self.isvalid(s, i):
                self.dfs(s[i + 1:], path + [s[:i + 1]], res)

    def isvalid(self, s, i):
        j = 0
        while j <= i:
            if s[j] != s[i]:
                return False
            j += 1
            i -= 1
        return True

    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        # s is ok when any j in range(i) satisfies: s[:j] is ok and s[j:i] in wordDict
        ok = [True]
        for i in range(1, len(s) + 1):
            ok += [any(ok[j] and s[j:i] in wordDict for j in range(i))]
        return ok[-1]

    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        if not p:
            return not s  # true: both s and p empty
        first_match = bool(s) and p[0] in {s[0], '.'}  # whether first char matches
        if len(p) >= 2 and p[1] == '*':
            return self.isMatch(s, p[2:]) or first_match and self.isMatch(s[1:], p)
        else:
            return first_match and self.isMatch(s[1:], p[1:])

    def findSubstring(self, s, words):
        """
        :type s: str
        :type words: List[str]
        :rtype: List[int]
        """
        if not s or not words:
            return []
        word_len, length = len(words[0]), len(words[0]) * len(words)
        ans, words_dic = [], dict()
        for word in words:  # get wrong while using set: the dup words would be omitted
            if word in words_dic.keys():
                words_dic[word] += 1
            else:
                words_dic[word] = 1
        # print(words_dic)

        for i in range(0, len(s) - length + 1):
            dic, j = words_dic.copy(), i  # use dict.copy()
            while dic:
                if s[j:j + word_len] in dic.keys():
                    dic[s[j:j + word_len]] -= 1
                    if dic[s[j:j + word_len]] == 0:
                        dic.pop(s[j:j + word_len])
                    j += word_len
                else:
                    break
            if not dic:
                ans = ans + [i]
        return ans

    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        ans, stack = 0, []
        stack.append(-1)
        for i in range(0, len(s)):
            if s[i] == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    ans = max(ans, i - stack[-1])  # parentheses: stack.peak to current pos
        return ans

    def isNumber(self, s):
        """
        :type s: str
        :rtype: bool
        """
        s = s.strip()
        try:
            s = float(s)
            return True
        except:
            return False

    def isNumber_DFA(self, s):
        state = [{},
                 {'blank': 1, 'sign': 2, 'digit': 3, '.': 4},
                 {'digit': 3, '.': 4},
                 {'digit': 3, '.': 5, 'e': 6, 'blank': 9},
                 {'digit': 5},
                 {'digit': 5, 'e': 6, 'blank': 9},
                 {'sign': 7, 'digit': 8},
                 {'digit': 8},
                 {'digit': 8, 'blank': 9},
                 {'blank': 9}]
        cur = 1
        for c in s:
            if c >= '0' and c <= '9':
                c = 'digit'
            if c == ' ':
                c = 'blank'
            if c in ['+', '-']:
                c = 'sign'
            if c not in state[cur].keys():
                return False
            cur = state[cur][c]
        return cur in [3, 5, 8, 9]

    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        row, col = len(word1), len(word2)
        # dp[i][j] represents minDistance from word1[:i] to word2[:j]
        dp = [[0] * (col + 1) for _ in range(0, row + 1)]
        dp[0][0] = 0
        for i in range(0, row + 1):
            dp[i][0] = i
        for i in range(0, col + 1):
            dp[0][i] = i
        for i in range(0, row):
            for j in range(0, col):
                if word1[i] == word2[j]:
                    dp[i + 1][j + 1] = dp[i][j]
                else:
                    dp[i + 1][j + 1] = min(dp[i][j], dp[i + 1][j], dp[i][j + 1]) + 1
        return dp[-1][-1]

    def minDistanceSlow(self, word1, word2):
        if not word1 and not word2:
            return 0
        if not word1:
            return len(word2)
        if not word2:
            return len(word1)
        if word1[0] == word2[0]:
            return self.minDistanceSlow(word1[1:], word2[1:])
        insert = self.minDistanceSlow(word1, word2[1:]) + 1
        delete = self.minDistanceSlow(word1[1:], word2) + 1
        replace = self.minDistanceSlow(word1[1:], word2[1:]) + 1
        return min(insert, delete, replace)

    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        if not s or not t:
            return ""
        need, count = Counter(t), len(t)
        start, end, i = 0, 0, 0
        for j, char in enumerate(s, 0):
            count -= need[char] > 0  # match current char
            need[char] -= 1
            if count == 0:  # matches success
                while i < j and need[s[i]] < 0:  # the char is redundant in the matching window
                    need[s[i]] += 1
                    i += 1
                if end == 0 or j - i + 1 < end - start:  # first or shorter window
                    start, end = i, j + 1
                # remove current from window to match next
                need[s[i]] += 1
                count += 1
                i += 1
        return s[start:end]

    def isScramble(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        if len(s1) != len(s2) or sorted(s1) != sorted(s2):
            return False
        if s1 == s2:
            return True
        n = len(s1)
        for i in range(1, n):
            # swap the two part or not
            if (self.isScramble(s1[:i], s2[:i]) and self.isScramble(s1[i:], s2[i:])) or (
                        self.isScramble(s1[:i], s2[n - i:]) and self.isScramble(s1[i:], s2[:n - i])):
                return True
        return False

    def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """

        @lru_cache(maxsize=None)
        def helper(i1, i2):
            i3 = i1 + i2
            if i1 == len(s1):
                return s2[i2:] == s3[i3:]
            if i2 == len(s2):
                return s1[i1:] == s3[i3:]

            if s3[i3] == s1[i1] and helper(i1 + 1, i2):
                return True
            if s3[i3] == s2[i2] and helper(i1, i2 + 1):
                return True
            return False

        return helper(0, 0)

    def numDistinct(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: int
        """
        if len(s) < len(t) or (len(s) == len(t) and s != t):
            return 0
        # s, t = '*' + s, '*' + t
        h, w = len(s) + 1, len(t) + 1
        dp = [[0] * w for _ in range(0, h)]  # dp[i][j]: numDistenct(s[:i], t[:j])
        for i in range(0, h):
            dp[i][0] = 1
        for i in range(1, h):
            for j in range(1, w):
                dp[i][j] = dp[i - 1][j]
                if s[i - 1] == t[j - 1]:
                    dp[i][j] += dp[i - 1][j - 1]
        return dp[-1][-1]

    def minCut(self, s):
        """
        :type s: str
        :rtype: int
        """
        res = [i-1 for i in range(0, len(s) + 1)]
        for i in range(0, len(s)):
            for j in range(i, len(s)):
                if s[i:j] == s[j:i:-1]:
                    res[j + 1] = min(res[j + 1], res[i] + 1)
        return res[-1]

