# -*- coding: utf-8 -*-

"""
    File name    :    IntegerSolution
    Date         :    14/08/2018
    Description  :    {TODO}
    Author       :    VickeeX
"""

from math import sqrt
from heapq import heappush, heappop


class IntegerSolution:
    def reverseBits(self, n):
        ans, count = 0, 0
        while count < 32:
            ans = ans * 2 + n % 2
            n, count = n // 2, count + 1
        return ans

    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        ans = 0
        while n > 0:
            if n % 2 == 1:
                ans += 1
            n //= 2
        return ans

    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        start, end = 1, n
        while start <= end:
            mid = (start + end) // 2
            if isBadVersion(mid):
                end = mid - 1
            else:
                start = mid + 1
        return start

    def guessNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        left, right = 1, n
        while left <= right:
            mid = (left + right) // 2
            g = guess(mid)
            print(left, right, mid, g)
            if g == -1:
                right = mid - 1
            elif g == 0:
                return mid
            else:
                left = mid + 1

    def countPrimeSetBits(self, L, R):
        """
        :type L: int
        :type R: int
        :rtype: int
        """
        primeSet, ans = {2, 3, 5, 7, 11, 13, 17, 19}, 0  # max: 10^6 ≈ 且 < 2^20, 最多20位即1的个数最多为20,故列出20以内的素数
        for i in range(L, R + 1):
            if primeSet.__contains__(bin(i).count('1')):
                ans += 1
                print(i, bin(i), bin(i).count('1'), ans)
        return ans

    # 0, 1, 8 remains, 2<->5, 6<->9
    def rotatedDigits(self, N):
        """
        :type N: int
        :rtype: int
        """
        ans = 0
        for x in range(1, N + 1):
            if any(s in str(x) for s in ["2", "5", "6", "9"]):
                if not any(s in str(x) for s in ["3", "4", "7"]):
                    ans += 1
        return ans
        # xset = set(map(int, str(x)))

    def binaryGap(self, N):
        """
        :type N: int
        :rtype: int
        """
        n, last, ans = bin(N)[2:], 0x7fffffff, 0
        for i in range(len(n)):
            if n[i] == '1':
                ans = max(ans, i - last)
                last = i
        return ans

    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        romans = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
        ans = ""
        for i in range(len(values)):
            while num >= values[i]:
                num, ans = num - values[i], ans + romans[i]
        return ans

    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        ONE = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
        TEN = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
        HUN = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]
        THU = ["", "M", "MM", "MMM"]
        return THU[num // 1000 % 10] + HUN[num // 100 % 10] + TEN[num // 10 % 10] + ONE[num % 10];

    """
    solution:
        可使用循环减法, 时间复杂度o(N)
        故将divisor每次扩大2倍进行减法, 时间复杂度o(logN)
    """

    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        if dividend == 0 or abs(dividend) < abs(divisor):
            return 0
        sign = (dividend < 0 and divisor > 0) or (dividend > 0 and divisor < 0)
        result, divd, divs = 0, abs(dividend), abs(divisor)
        while divd >= divs:
            tmpd, tmpa = divs, 1
            while divd >= (tmpd << 1):
                tmpd, tmpa = tmpd << 1, tmpa << 1
            divd, result = divd - tmpd, result + tmpa

        if sign:
            result = -result
        if result > 2147483647:
            return 2147483647
        return result

    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n < 0:
            return 1 / self.myPow(x, -n)
        ans = 1
        while n > 0:
            if n % 2 == 1:
                ans *= x
            x *= x
            n //= 2
        return ans

    def computeArea(self, A: int, B: int, C: int, D: int, E: int, F: int, G: int, H: int) -> int:
        ans, l, r, d, u = (C - A) * (D - B) + (G - E) * (H - F), 0, 0, 0, 0
        if (A <= E <= C or E <= A <= G) and (B <= F <= D or F <= B <= H):
            ans -= (min(C, G) - max(A, E)) * (min(D, H) - max(B, F))
        return ans

    def numberToWords(self, num: int) -> str:
        units = ["Thousand", "Million", "Billion"]
        to19 = "One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve " \
               "Thirteen Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen".split()
        tens = 'Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety'.split()

        def helper(n):
            if n < 20:
                return to19[n - 1:n]
            if n < 100:
                return [tens[n // 10 - 2]] + helper(n % 10)
            if n < 1000:
                return [to19[n // 100 - 1]] + ['Hundred'] + helper(n % 100)
            for i, u in enumerate(units, 2):
                if n < 1000 ** i:
                    return helper(n // 1000 ** (i - 1)) + [u] + helper(n % 1000 ** (i - 1))

        return ' '.join(helper(num)) or "Zero"

    def numSquares(self, n: int) -> int:
        if n < 4:
            return n
        pows, curs, ans = [i * i for i in range(1, int(sqrt(n) + 1)) if i * i <= n], {n}, 0
        while curs:
            tmp, ans = set(), ans + 1
            for x in curs:
                for y in pows:
                    if x == y:  # finish
                        return ans
                    if x < y:  # not enogh to split
                        break
                    tmp.add(x - y)  # x>y, add to next layer
            curs = tmp
        return ans

    def isAdditiveNumber(self, num: str) -> bool:
        for i in range(1, len(num) // 2 + 2):
            for j in range(i + 1, len(num)):
                if (i > 1 and num[0] == '0') or (j - i > 1 and num[i] == '0'):  # no preceeding zeros
                    break
                m = str(int(num[:i]) + int(num[i:j]))
                # print(num[:i], num[i:j], m)
                if num[j:] == m or \
                        (num[j:].startswith(m) and self.isAdditiveNumber(num[i:])):  # finished or next iteration
                    return True
        return False

    def nthSuperUglyNumber(self, n: int, primes: list) -> int:
        # heap, count, ans = [1], 1, 1
        # while count < n:
        #     ans = heappop(heap)
        #     count += 1
        #     for p in primes:
        #         if ans * p not in heap:
        #             heappush(heap, ans * p)
        # return heap[0]
        # ------------
        # last: records the latest ugly_num
        # ans: records the ugly_nums
        # index: index of the last ugly_num the prime results in
        # latest_ones: last ugly_num the prime results in

        size = len(primes)
        last, ans, index, latest_ones = 1, [1], [0] * size, [1] * size
        for i in range(1, n):
            for j in range(0, size):
                if latest_ones[j] == last:
                    latest_ones[j] = ans[index[j]] * primes[j]  # update the one to new ugly_num
                    index[j] += 1  # index of the num which should be ready to multipy the prime next
            last = min(latest_ones)
            ans.append(last)

        return ans[-1]


def guess(num):
    if num < 3:
        return 1
    elif num == 3:
        return 0
    else:
        return -1


def isBadVersion(version):
    if version < 3:
        return False
    return True
