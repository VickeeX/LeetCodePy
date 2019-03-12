# -*- coding: utf-8 -*-

"""
    File name    :    IntegerSolution
    Date         :    14/08/2018
    Description  :    {TODO}
    Author       :    VickeeX
"""


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
