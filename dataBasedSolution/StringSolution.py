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
        res = [i - 1 for i in range(0, len(s) + 1)]
        for i in range(0, len(s)):
            for j in range(i, len(s)):
                if s[i:j] == s[j:i:-1]:
                    res[j + 1] = min(res[j + 1], res[i] + 1)  # dp
        return res[-1]

    def reverseWords(self, s: str) -> str:
        ans, last, first = "", 0, True
        while last < len(s) and s[last] == ' ':
            last += 1
        for i in range(len(s)):
            if s[i] != ' ' and (i == len(s) - 1 or s[i + 1] == ' '):
                if first:
                    ans, first = s[last:i + 1], False
                else:
                    ans = s[last:i + 1] + ' ' + ans
                last = i + 1
                while last < len(s) and s[last] == ' ':
                    last += 1
        return ans

    def compareVersion(self, version1: str, version2: str) -> int:
        i, j, li, lj = 0, 0, 0, 0
        while i < len(version1) or j < len(version2):
            while i < len(version1) and version1[i] != '.':
                i += 1
            while j < len(version2) and version2[j] != '.':
                j += 1
            v1, v2 = int(version1[li:i] if i <= len(version1) else 0), int(version2[lj:j] if j <= len(version2) else 0)
            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1
            i, j = i + 1, j + 1
            li, lj = i, j
        return 0

    def compareVersion_(self, version1: str, version2: str) -> int:
        n1, n2 = [int(i) for i in version1.split('.')], [int(i) for i in version2.split('.')]
        for i in range(max(len(n1), len(n2))):
            t1, t2 = n1[i] if i < len(n1) else 0, n2[i] if i < len(n2) else 0
            if t1 > t2:
                return 1
            elif t1 < t2:
                return -1
        return 0

    def fractionToDecimal(self, numerator: int, denominator: int) -> str:
        ans = ""
        if numerator / denominator < 0:
            ans, numerator, denominator = "-", abs(numerator), abs(denominator)
        dic, ans, numerator = dict(), ans + str(numerator // denominator), numerator % denominator * 10
        if numerator != 0:
            ans += '.'
        while numerator != 0:
            if numerator in dic:
                ans = ans[:dic[numerator]] + "(" + ans[dic[numerator]:] + ")"
                break
            ans += str(numerator // denominator)
            dic[numerator] = len(ans) - 1
            numerator = numerator % denominator * 10
        return ans

    def fractionToDecimal_(self, numerator: int, denominator: int) -> str:
        ans = ""
        if numerator / denominator < 0:
            ans = "-"
        numerator, denominator = -numerator if numerator < 0 else numerator, -denominator if denominator < 0 else denominator
        ans += str(numerator // denominator)
        numerator %= denominator
        mods = {}
        if numerator > 0:
            ans += "."
            while numerator > 0 and numerator not in mods:
                mods[numerator] = (numerator * 10) // denominator
                numerator = (numerator * 10) % denominator
        for i in mods:
            if i == numerator:
                ans += "("
            ans += str(mods[i])
        if numerator != 0:
            ans += ")"
        return ans

    def convertToTitle(self, n: int) -> str:
        # ans= ""
        # while n > 0:
        #     ans = chr((n - 1) % 26 + 65) + ans
        #     n = (n - 1) // 26
        # return ans
        ans, dic = "", {i: char for i, char in enumerate(string.ascii_uppercase)}
        print(dic.keys())
        while n > 0:
            ans = dic[(n - 1) % 26] + ans
            n = (n - 1) // 26
        return ans

    def titleToNumber(self, s: str) -> int:
        ans = 0
        for i, char in enumerate(s[::-1]):
            ans += (ord(char) - 64) * 26 ** i
        # for i in s:
        #     ans = ans * 26 + (ord(i) - 64)
        return ans

    def findRepeatedDnaSequences(self, s: str) -> list:
        dic = defaultdict(int)
        for i in range(len(s) - 9):
            dic[s[i:i + 10]] += 1
        return [k for k in dic if dic[k] > 1]
        # or use str.contains(sub)

    def shortestPalindromeBruteForce(self, s: str) -> str:
        if len(s) < 2:
            return s
        r = s[::-1]
        for i in range(len(s)):
            if s.startswith(r[i:]):
                print(r, s)
                return r[:i] + s

                # def isPalindrome(t):
                #     for i in range(len(t) // 2):
                #         if t[i] != t[-i - 1]:
                #             return False
                #     return True
                #
                # l = len(s)
                # r1, r2 = 1, [0, 1]
                #
                # for i in range(l - 1):
                #     if isPalindrome(s[:l - i]) and l - i > r1:  # in front
                #         r1, r2 = l - i, [0, l - i]
                #         break
                #         # if isPalindrome(s[i:]) and l - i > r1: # behind
                #         #     r1, r2 = l - i, [i, l]
                #         #     break
                # i, j = r2
                # return s[j:][::-1] + s
                # # return s[j:][::-1] + s if i == 0 else s + s[:i][::-1]

    def calculate_1(self, s: str) -> int:
        stack, ans, num, op = [], 0, 0, 1
        for c in s:
            if c.isdigit():
                num = num * 10 + int(c)  # constant digit as one integer
            elif c in '+-':
                ans += num * op  # calculate the formula before
                op, num = 1 if c == '+' else -1, 0  # using 1 or -1 to present + or -
            elif c == '(':
                stack.append(ans)  # the value
                stack.append(op)  # the symbolic item for formula in next brackets
                ans, op = 0, 1
            elif c == ')':
                ans += num * op
                ans *= stack.pop()  # the symbolic item
                ans += stack.pop()  # the value before
                num = 0
        return ans + num * op

    def calculate_2(self, s: str) -> int:
        stack, ans, num, op_mul, tag = [], 0, 0, False, False
        for c in s + '+0':
            if c.isdigit():
                num = num * 10 + int(c)
            elif c in '-+*/':
                if tag:
                    num = ans * num if op_mul else ans // num
                    ans, tag = 0, False
                if c in '-+':
                    stack.append(num)
                    stack.append(1 if c == '+' else '-1')
                    num = 0
                elif c in '*/':
                    ans, num, tag, op_mul = num, 0, True, True if c == '*' else False
        stack.append(num)
        ans = stack[0]
        for i, c in enumerate(stack[1:]):
            if i % 2 == 1:
                ans = ans + c if stack[i] == 1 else ans - c
        return ans

    def calculate_2_(self, s: str) -> int:
        stack, num, op = [], 0, '+'
        for c in s + '+0':
            if c.isdigit():
                num = num * 10 + int(c)
            elif c in '+-*/':
                if op == '+':
                    stack.append(num)
                elif op == '-':
                    stack.append(-num)
                elif op == '*':
                    stack.append(stack.pop() * num)
                else:
                    stack.append(-(-stack.pop() // num) if stack[-1] < 0 else stack.pop() // num)
                num, op = 0, c
        return sum(stack)

    def diffWaysToCompute(self, input: str) -> list:
        ans = []
        for i, c in enumerate(input):
            if c in '-+*':
                ans += [a + b if c == '+' else a - b if c == '-' else a * b
                        for a in self.diffWaysToCompute(input[:i])
                        for b in self.diffWaysToCompute(input[i + 1:])]
        return ans or [int(input)]

    def diffWaysToCompute_(self, input: str) -> list:
        return [a + b if c == '+' else a - b if c == '-' else a * b
                for i, c in enumerate(input) if c in '-+*'
                for a in self.diffWaysToCompute(input[:i])
                for b in self.diffWaysToCompute(input[i + 1:])] or [int(input)]

    def addOperators(self, num: str, target: int) -> list:
        def helper(pos, res, prev, expr):
            if pos == len(num) and res == target:
                ans.append(expr)
                return
            n = 0
            for i in range(pos, len(num)):
                n = n * 10 + int(num[i])
                if pos == 0:  # first number with no prev
                    helper(i + 1, n, n, str(n))
                else:
                    helper(i + 1, res + n, n, expr + "+" + str(n))  # add
                    helper(i + 1, res - n, -n, expr + "-" + str(n))  # sub: note the third parameter is "-n"
                    helper(i + 1, res - prev + prev * n, prev * n, expr + "*" + str(n))  # mul
                if num[pos] == '0':
                    break

        ans = []
        helper(0, 0, 0, "")
        return ans

        # def helper(pos, cur_total, prev_n, expr):
        #     if pos == len(num) and cur_total == target:
        #         ans.append(expr)
        #     else:
        #         n = 0
        #         for i in range(pos, len(num)):
        #             n = n * 10 + int(num[i])
        #             if pos == 0:
        #                 helper(i + 1, cur_total + n, n, expr + str(n))
        #             else:
        #                 helper(i + 1, cur_total + n, n, expr + '+' + str(n))
        #                 helper(i + 1, cur_total - n, -n, expr + '-' + str(n))
        #                 helper(i + 1, cur_total - prev_n + prev_n * n, prev_n * n, expr + '*' + str(n))
        #             if num[pos] == '0':
        #                 break
        #
        # ans = []
        # helper(0, 0, 0, "")
        # return ans
        # if not num:
        #     return []
        #
        # def helper(s):
        #     if not s:
        #         return []
        #     tmp = [s] if not (s[0] == '0' and len(s) > 1) else []
        #     for i in range(1, len(s)):
        #         tmp += [s[:i] + x + y for x in ('+', '-', '*') for y in helper(s[i:]) if
        #                 not (s[0] == '0' and i > 1)]
        #     return tmp
        #
        # return [x for x in helper(num) if self.calculate_2_(x) == target]

    def getHint(self, secret: str, guess: str) -> str:
        # it's also ok to use one record array and two travesal.
        # or use defaultdict.
        rec0, rec1, bulls = [0] * 10, [0] * 10, 0
        for i in range(len(secret)):
            if secret[i] == guess[i]:
                bulls += 1
            else:
                rec0[int(secret[i])] += 1
                rec1[int(guess[i])] += 1
        # print(rec0, rec1)
        return str(bulls) + "A" + str(sum([min(rec0[i], rec1[i]) for i in range(10)])) + "B"

    def removeDuplicateLetters(self, s: str) -> str:
        for c in sorted(set(s)):
            suffix = s[s.index(c):]
            if set(suffix) == set(s):  # the max char's min pos if suffix contains all chars
                return c + self.removeDuplicateLetters(suffix.replace(c, ''))
                # ```else: continue ```
                # if the current max char's suffix doesn't contains all, then next

        return ''

    def lengthLongestPath(self, input: str) -> int:
        ans, path = 0, {0: 0}  # index each depth once at a time
        for line in input.splitlines():
            name = line.lstrip('\t')
            depth = len(line) - len(name)
            if '.' in name:
                ans = max(ans, path[depth] + len(name))
            else:
                path[depth + 1] = path[depth] + len(name) + 1
        return ans

    def isSubsequence(self, s: str, t: str) -> bool:
        # if not s: return True
        # if not t: return False
        # if len(s) == 1:
        #     return s in t
        # for i, c in enumerate(t):
        #     if c == s[0] and self.isSubsequence(s[1:], t[i + 1:]):
        #         return True
        # return False
        for c in s:
            if c not in t:
                return False
            t = t[t.index(c) + 1:]
        return True

    def decodeString(self, s: str) -> str:
        stack, count, num = [""], [1], ""
        for c in s:
            if c.isdigit():
                num += c
            elif c == '[':
                count.append(int(num))
                stack.append("")
                num = ""
            elif c == ']':
                curs = stack.pop()
                stack[-1] += curs * count.pop()
            else:
                stack[-1] += c
        return stack[-1]
        # while '[' in s:
        # s = re.sub(r'(\d+)\[([a-z]*)\]', lambda m: int(m.group(1)) * m.group(2), s)
        # return s

    def longestSubstring(self, s: str, k: int) -> int:
        # for c in set(s):
        #     if s.count(c) < k:
        #         return max(self.longestSubstring(t, k) for t in s.split(c))
        # return len(s)
        stack = [s]
        ans = 0
        while stack:
            s = stack.pop()
            for c in set(s):
                if s.count(c) < k:
                    stack.extend([t for t in s.split(c)])
                    break
            else:
                ans = max(ans, len(s))  # if each count(c)>=k
        return ans

    def strongPasswordChecker(self, s: str) -> int:
        lower, upper, digit = 1, 1, 1
        for c in s:  # missed time for required characters
            if c.isdigit():
                digit = 0
            elif c.islower():
                lower = 0
            elif c.isupper():
                upper = 0
        required, l = lower + upper + digit, len(s)
        if l < 6: return max(required, 6 - l)

        replaced, onedel, twodel, i = 0, 0, 0, 0
        while i < l:
            cl = 1
            while i + cl < l and s[i] == s[i + cl]:
                cl += 1
            if cl > 2:
                replaced += cl // 3
                if cl % 3 == 0: onedel += 1
                if cl % 3 == 1: twodel += 2
            i += cl
        # last, cl = s[0], 1
        # for i, c in enumerate(s[1:], 1):
        #     if c == last:
        #         cl += 1
        #     if c != last or i == l - 1:
        #         if cl > 2:
        #             replaced += cl // 3
        #             if cl % 3 == 0: onedel += 1
        #             if cl % 3 == 1: twodel += 2
        #         cl, last = 1, c

        print(replaced)

        if l <= 20:
            return max(required, replaced)
        todel = l - 20
        replaced -= min(todel, onedel)  # every del saves one replace
        replaced -= min(max(todel - onedel, 0), twodel) // 2  # every two dels saves one replace
        replaced -= max(todel - onedel - twodel, 0) // 3

        return todel + max(required, replaced)

    def originalDigits(self, s: str) -> str:
        # # {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
        # #  5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
        # countChar, nums = defaultdict(int), [0] * 10
        # for c in s: countChar[c] += 1
        # nums[0] = countChar['z']  # only "zero" contains "z"
        # nums[2] = countChar['w']  # only "two" contains "w"
        # nums[4] = countChar['u']  # only "four" contains "u"
        # nums[6] = countChar['x']  # only "six" contains 'x'
        # nums[8] = countChar['g']  # only "eight" contains 'g'
        # nums[1] = countChar['o'] - nums[0] - nums[2] - nums[4]  # 'o'
        # nums[3] = countChar['h'] - nums[8]  # 'h'
        # nums[5] = countChar['f'] - nums[4]  # 'f'
        # nums[7] = countChar['s'] - nums[6]  # 's'
        # nums[9] = countChar['i'] - nums[5] - nums[6] - nums[8]

        nums = [0] * 10
        nums[0] = s.count('z')  # only "zero" contains "z"
        nums[2] = s.count('w')  # only "two" contains "w"
        nums[4] = s.count('u')  # only "four" contains "u"
        nums[6] = s.count('x')  # only "six" contains 'x'
        nums[8] = s.count('g')  # only "eight" contains 'g'
        nums[1] = s.count('o') - nums[0] - nums[2] - nums[4]  # 'o'
        nums[3] = s.count('h') - nums[8]  # 'h'
        nums[5] = s.count('f') - nums[4]  # 'f'
        nums[7] = s.count('s') - nums[6]  # 's'
        nums[9] = s.count('i') - nums[5] - nums[6] - nums[8]

        ans = ""
        for i, n in enumerate(nums):
            ans += str(i) * n
        return ans

    def characterReplacement(self, s: str, k: int) -> int:  # sliding window
        # ans = left = 0
        # counts = Counter()
        # for right in range(len(s)):
        #     counts[s[right]] += 1
        #     commonc = counts.most_common(1)[0][1]  # the repeated char
        #     if right - left + 1 - commonc > k:  # use "if" not "while", because right grow 1 at each time, consider "AABAC"
        #         counts[s[left]] -= 1
        #         left += 1
        #     ans = max(ans, right - left + 1)
        # return ans

        left, freq, count = 0, 0, defaultdict(int)
        for right in range(len(s)):
            count[s[right]] += 1
            freq = max(freq, count[s[right]])
            if (right - left + 1 - freq) > k:  # after the max sub string, the length of window keeps the max
                count[s[left]] -= 1
                left += 1
        return len(s) - left

    def minMutation(self, start: str, end: str, bank: list) -> int:
        if end not in bank: return -1
        mutations = {"A": ["C", "G", "T"], "C": ["A", "G", "T"], "G": ["A", "C", "T"], "T": ["A", "C", "G"]}
        stack, bank = deque(), set(bank)
        stack.append((start, 0))  # BFS

        while stack:
            mid, step = stack.popleft()
            if mid == end:
                return step
            for i, x in enumerate(mid):
                for y in mutations[x]:
                    nxt = mid[:i] + y + mid[i + 1:]
                    if nxt in bank:
                        bank.remove(nxt)  # to prevent loop
                        stack.append((nxt, step + 1))
        return -1

    def frequencySort(self, s: str) -> str:
        counter = Counter(s)
        arr = sorted([(-counter[k], k) for k in counter.keys()])  # first sort by -counter[k], then character sort
        return ''.join([c * (-n) for n, c in arr])

    def validIPAddress(self, IP: str) -> str:
        def IPV4Part(s):
            try:
                return str(int(s)) == s and 0 <= int(s) <= 255
            except:
                return False

        def IPV6Part(s):
            if len(s) > 4: return False
            try:
                return int(s, 16) >= 0 and s[0] != '-'
            except:
                return False

        if IP.count('.') == 3 and all(IPV4Part(s) for s in IP.split(".")):
            return "IPv4"
        if IP.count(':') == 7 and all(IPV6Part(s) for s in IP.split(":")):
            return "IPv6"
        return "Neither"

    def getMaxRepetitions(self, s1: str, n1: int, s2: str, n2: int) -> int:
        if len(s1) * n1 < len(s2) * n2:
            return 0
        s1_round = s2_round = idx = 0
        record, l2 = {}, len(s2)
        # pos, s2set = defaultdict(list), set(s2)
        # for i, c in enumerate(s1):
        #     if c in s2set:
        #         pos[c].append(i)
        # for c in s2set:
        #     if not pos[c]:
        #         return 0
        while s1_round < n1:
            s1_round += 1
            for c in s1:
                if c == s2[idx]:
                    idx += 1
                    if idx == l2:
                        s2_round += 1
                        idx = 0
            if idx in record:
                s1r, s2r = record[idx]
                c1, c2 = s1_round - s1r, s2_round - s2r
                ans = (n1 - s1r) // c1 * c2  # the count or s2 between last idx pos to current pos
                left_s1 = (n1 - s1r) % c1 + s1r
                for k, v in record.items():  # the left count of s1
                    if v[0] == left_s1:
                        ans += v[1]
                        break
                return ans // n2
            else:
                record[idx] = (s1_round, s2_round)
        return s2_round // n2

    def findMaxForm(self, strs: list, m: int, n: int) -> int:
        # # dp: TLE
        # dic = {s: [s.count("0"), s.count("1")] for s in set(strs)}
        # dp = [[0] * (n + 1) for _ in range(m + 1)]
        #
        # for i,s in enumerate(strs):
        #     z, o = dic[s]
        #     for x in range(m, -1, -1):
        #         for y in range(n, -1, -1):
        #             if x >= z and y >= o:
        #                 dp[x][y] = max(1 + dp[x - z][y - o], dp[x][y])
        # return dp[m][n]

        # recursive solution: TLE
        # c0, c1 = strs[0].count("0"), strs[0].count("1")
        # ans = self.findMaxForm(strs[1:], m, n)
        # if c0 <= m and c1 <= n:
        #     ans = max(ans, 1 + self.findMaxForm(strs[1:], m - c0, n - c1))
        # return ans

        strs.sort()
        dp, l, dic = {}, len(strs), {s: [s.count('0'), s.count('1')] for s in set(strs)}

        def helper(pre, pos, rm, rn):
            if rm < 0 or rn < 0: return None
            if pos == l: return 0
            s = strs[pos]
            if (pos, rm, rn) in dp:
                return dp[(pos, rm, rn)]
            z, o = dic[s]
            # if pre == s:
            #     return helper(pre, pos + 1, rm - z, rn - o)
            ans = helper(s, pos + 1, rm, rn)  # do not format current
            if ans is None: ans = 0
            if z <= rm and o <= rn:
                tmp = helper(s, pos + 1, rm - z, rn - o)
                if tmp is not None and tmp >= ans:
                    ans = tmp + 1
            dp[(pos, rm, rn)] = ans
            return ans

        return helper("", 0, m, n)

    def findMaxForm_(self, strs: list, m: int, n: int) -> int:
        strs.sort()
        dp, l, dic = {}, len(strs), {s: [s.count('0'), s.count('1')] for s in set(strs)}

        def helper(start, rm, rn):
            if rm < 0 or rn < 0:
                return None
            if (start, rm, rn) in dp:
                return dp[(start, rm, rn)]

            ans, tag = 0, False
            for i in range(start, l):
                if i > start and strs[i] == strs[i - 1]:  # searched before when dfs(i-1,...)
                    continue
                z, o = dic[strs[i]]
                r = helper(i + 1, rm - z, rn - o)
                if r is not None and (not tag or r > ans):
                    ans, tag = r, True
            dp[(start, rm, rn)] = ans + 1 if tag else 0
            return dp[(start, rm, rn)]

        return helper(0, m, n)

    def findAllConcatenatedWordsInADict(self, words: list) -> list:
        # note: comprised **entirely** of at least two shorter words in the given array
        words.sort(key=len)
        preWords, ans = set(), []

        def form(word):
            if not word:
                return False
            wl = len(word)
            dp = [True] + [False] * wl
            for i in range(1, wl + 1):  # split the word into segments
                for j in range(i - 1, -1, -1):  # pre and current segment both meet the requirement
                    if dp[j] and word[j:i] in preWords:
                        dp[i] = True
                        break
            return dp[-1]

        for w in words:
            if form(w):
                ans.append(w)
            else:
                preWords.add(w)
        return ans

    def findAllConcatenatedWordsInADict_(self, words: list) -> list:
        words, ans = set(words), []

        def form(word):
            if word in words:
                return True
            for i in range(1, len(word)):
                if word[:i] in words and form(word[i:]):
                    return True
            return False

        for w in words:
            words.remove(w)
            if form(w):
                ans.append(w)
            words.add(w)
        return ans

    def findMinStep(self, board: str, hand: str) -> int:
        dic = Counter(hand)

        def helper(board, dic):
            if not board: return 0
            ans, i = float('inf'), 0
            while i < len(board):
                j = i + 1
                while j < len(board) and board[j] == board[i]:
                    j += 1
                need = 3 - (j - i) if j - i < 3 else 0  # need
                if dic[board[i]] >= need:
                    dic[board[i]] -= need  # eliminate board[i] now or not(while traverse others)
                    tmp = helper(board[:i] + board[j:], dic)
                    if tmp >= 0:
                        ans = min(ans, tmp + need)
                    dic[board[i]] += need
                i = j
            return ans if ans != float('inf') else -1

        return helper(board, dic)

    def findRotateSteps(self, ring: str, key: str) -> int:
        m, n, memo = len(ring), len(key), {}  # memo (status,pos) to reduce time complexity

        def helper(status, pos):
            if status == n:
                return 0
            if (status, pos) in memo:
                return memo[(status, pos)]
            l, r, ls, rs = pos, pos, 0, 0
            while ring[l] != key[status]:  # turn left: find the pos ans steps
                l = (l - 1) % m
                ls += 1
            while ring[r] != key[status]:  # turn right
                r = (r + 1) % m
                rs += 1
            ans = min(ls + helper(status + 1, l), rs + helper(status + 1, r))  # choose min of left and right
            memo[(status, pos)] = ans  # append current status to memo
            return ans

        return helper(0, 0) + n

    def longestPalindromeSubseq(self, s: str) -> int:
        if not s:
            return 0
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if i == j:
                    dp[i][j] = 1
                elif s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        return dp[0][-1]

    def shortestCompletingWord(self, licensePlate: str, words: list) -> str:
        # 748. Shortest Completing Word
        plate = Counter([c.lower() for c in licensePlate if c.isalpha()])
        for word in sorted(words, key=len):
            counter = Counter(word)
            tag = True
            for k, v in plate.items():
                if k not in counter or v > counter[k]:
                    tag = False
                    break
            if tag:
                return word

    def shortestCompletingWord_(self, licensePlate: str, words: list) -> str:
        plate = [c.lower() for c in licensePlate if c.isalpha()]
        for word in sorted(words, key=len()):
            tmp = plate.copy()
            for c in word:
                if c in tmp:
                    # i = tmp.index(c)
                    # tmp = tmp[:i] + tmp[i + 1:]
                    del tmp[tmp.index(c)]
            if not tmp:
                return word

    def findLongestWord(self, s: str, d: list) -> str:
        # 524. Longest Word in Dictionary through Deleting
        for word in sorted(sorted(d), key=len, reverse=True):  # sorted by len first, then lexicographically
            tmp = iter(s)
            if all(c in tmp for c in word):
                return word
        return ""

    def findLongestWord_(self, s: str, d: list) -> str:
        def helper(word):  # faster than iter(...)
            pos = 0
            for c in word:
                j = s.find(c, pos)  # according to arrange
                if j == -1:
                    return False
                pos = j + 1
            return True

        ans = ""
        for word in d:
            if len(word) > len(ans) or (len(word) == len(ans) and word < ans):
                if helper(word):
                    ans = word
        return ans

    def complexNumberMultiply(self, a: str, b: str) -> str:
        # 537. Complex Number Multiplication
        A = [int(x) for x in a.replace('i', '').split('+')]
        B = [int(x) for x in b.replace('i', '').split('+')]
        return str(A[0] * B[0] - A[1] * B[1]) + "+" + str(A[0] * B[1] + A[1] * B[0]) + "i"

    def findMinDifference(self, timePoints: list) -> int:
        # 539. Minimum Time Difference
        def sub(a, b):
            dif = (int(b[:2]) - int(a[:2])) * 60 + int(b[3:]) - int(a[3:])
            if dif > 720:
                dif = (int(a[:2]) + 24 - int(b[:2])) * 60 + int(a[3:]) - int(b[3:])
            return dif

        timePoints.sort()
        ans = 2000
        for i, t in enumerate(timePoints):
            if i == len(timePoints) - 1:
                a, b = timePoints[0], timePoints[i]
            else:
                a, b = timePoints[i], timePoints[i + 1]
            dif = sub(a, b)
            if dif < ans:
                ans = dif
        return ans

    def findLUSlength(self, strs: list) -> int:
        # 522. Longest Uncommon Subsequence II
        def is_sub(s1, s2):  # whether s1 is the sub string of s2
            i = j = 0
            while i < len(s1) and j < len(s2):
                if s1[i] == s2[j]:
                    i, j = i + 1, j + 1
                else:
                    j += 1
            return i == len(s1)

        uncom, common = set(), set()
        for s in strs:
            if s in uncom:
                uncom.remove(s)
                common.add(s)
            elif s not in common:
                uncom.add(s)
        for s in sorted(uncom, key=len, reverse=True):
            for ss in common:
                if len(s) < len(ss) and is_sub(s, ss):  # common
                    break
            else:  # s is uncommon
                return len(s)
        return -1

    def checkInclusion(self, s1: str, s2: str) -> bool:
        # 567. Permutation in String
        # using sliding window
        size, s2 = len(s1), "-" + s2
        c1, c2 = Counter(s1), Counter(s2[:size])
        for i, c in enumerate(s2[size:], size):
            c2[s2[i - size]] -= 1
            c2[c] += 1
            for k, v in c1.items():
                if v != c2[k]:
                    break
            else:
                return True
        return False

    def countSubstrings(self, s: str) -> int:
        # 647. Palindromic Substrings
        size = len(s)
        dp1, dp2 = [1] * size, [1 if c == s[i + 1] else 0 for i, c in enumerate(s[:-1])]
        ans = sum(dp1) + sum(dp2)
        for i in range(2, size):
            if i % 2 == 0:
                dp = dp1
            else:
                dp = dp2
            tmp = []
            for j in range(size - i):
                if dp[j + 1] == 1 and s[j] == s[j + i]:
                    tmp.append(1)
                    ans += 1
                else:
                    tmp.append(0)
            if i % 2 == 0:
                dp1 = tmp
            else:
                dp2 = tmp
        return ans

    def findDuplicate(self, paths: list) -> list:
        # 609. Find Duplicate File in System
        dic = defaultdict(list)
        for p in paths:
            path = p.split(' ')
            pre = path[0]
            for file in path[1:]:
                idx = file.index('(')
                dic[file[idx + 1:-1]].append(pre + "/" + file[:idx])
        return [v for v in dic.values() if len(v) > 1]

    def replaceWords(self, dict: list, sentence: str) -> str:
        # 648. Replace Words
        words = sentence.split(" ")
        dict.sort(key=len)
        for i, w in enumerate(words):
            for r in dict:
                if len(r) > len(w):
                    break
                elif w.startswith(r):
                    words[i] = r
                    break
        return " ".join(words)

    def solveEquation(self, equation: str) -> str:
        # 640. Solve the Equation
        a, b, cur, tag = 0, 0, "", True  # a: coefficient of x, b is the constant term,
        cur = ""
        for i, c in enumerate(equation):
            if c in "+-=" or i == len(equation) - 1:
                if i == len(equation) - 1:
                    cur += c
                if not cur:
                    cur = c
                    continue
                symbol = +1 if tag else -1
                if cur[0] in "+-":
                    symbol *= +1 if cur[0] == '+' else -1
                    cur = cur[1:]
                if cur[-1] == 'x':
                    a += symbol * int(cur[:-1] if len(cur) > 1 else 1)
                else:
                    b += symbol * int(cur)
                cur = c
                if c == '=':
                    tag, cur = False, ""
            else:
                cur += c
        if a == 0 and b != 0:
            return "No solution"
        elif a == 0 and b == 0:
            return "Infinite solutions"
        else:
            return "x=" + str(-b // a) if -b % a == 0 else "x=" + str(-b / a)
