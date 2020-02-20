# -*- coding: utf-8 -*-

"""
    @File name    :    SwordOffer.py
    @Date         :    2020-02-19 19:09
    @Description  :    {TODO}
    @Author       :    VickeeX
"""


def topic_3_repeated_num(nums):
    # [1,4,5,0,2,2]
    for i, n in enumerate(nums):
        tmp = n
        while i != tmp:
            if nums[tmp] == tmp:
                return tmp
            else:
                nums[tmp], tmp = tmp, nums[tmp]


def topic_3_repeated_num_without_nums_modify(nums):
    def find_helper(left, right):
        if right == left:
            return right
        mid, count = (right + left) // 2, 0
        for n in nums:
            if left <= n <= mid:
                count += 1
        if count <= (mid - left + 1):
            return find_helper(mid + 1, right)
        else:
            return find_helper(left, mid)

    return find_helper(1, len(nums) - 1)


def topic_4_two_dimensional_array_search(nums, target):
    if not nums or not nums[0]:
        return False
    row, col = 0, len(nums[0]) - 1
    while 0 <= row < len(nums) and 0 <= col < len(nums[0]):
        if nums[row][col] == target:
            return True
        elif nums[row][col] < target:
            row += 1
        else:
            col -= 1
    return False


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def topic_6_print_list_from_tail_to_head(head):
    # using stack to print, but python list don't pop out to reverse
    if not head:
        return []
    stack, tmp = [head.val], head
    while tmp.next:
        tmp = tmp.next
        stack.append(tmp.val)
    return stack[::-1]


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def topic_7_reconstruct_binary_tree_from_preorder_midorder(pre, mid):
    # recursively construct
    if not pre or len(pre) != len(mid):
        return None
    head = TreeNode(pre[0])
    idx1 = -1
    for i, n in enumerate(mid):
        if n == pre[0]:
            idx1 = i
    head.left = topic_7_reconstruct_binary_tree_from_preorder_midorder(pre[1:idx1 + 1], mid[:idx1])
    head.right = topic_7_reconstruct_binary_tree_from_preorder_midorder(pre[idx1 + 1:], mid[idx1 + 1:])
    return head


def topic_8_next_node_in_midorder(target):
    if not target:
        return target
    tmp = None
    if target.right:
        tmp = target.right
        while tmp.left:
            tmp = tmp.left
    elif target.father:
        tmp = target
        while tmp.father and tmp.father.right == tmp:
            tmp = tmp.father
        tmp = tmp.father
    return tmp  # if not tmp: None


class Topic9QueuebyTwoStacks:
    stack1 = []  # keeps new elements
    stack2 = []  # keeps reversed old elements

    def push(self, node):
        self.stack1.append(node)

    def pop(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()


def topic_10_fibonacci(n):
    if n < 2:
        return n
    a, b = 0, 1
    while n > 1:
        a, b, n = b, a + b, n - 1
        print(b)
    return b


def topic_11_min_number_in_rotate_increasing_array(nums):
    l, r = 0, len(nums) - 1
    mid = 0
    while nums[l] >= nums[r]:
        if l + 1 == r:
            mid = r
            break
        mid = (l + r) // 2
        # if nums[l]==num[m]==numd[r]:
        #     search_order
        if nums[l] <= nums[mid]:
            l = mid
        elif nums[mid] <= nums[r]:
            r = mid

    return nums[mid]


def topic_12_path_exists_in_matrix(matrix, rows, cols, path):
    # backtracking
    def helper(i, j, s):
        if not s:
            return True
        record[i * cols + j] = True
        for ni, nj in [(i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1)]:
            if 0 <= ni < rows and 0 <= nj < cols and not record[ni * cols + nj] and \
                    matrix[ni * cols + nj] == s[0] and helper(ni, nj, s[1:]):
                return True
        record[i * cols + j] = False
        return False

    if not matrix or not path or rows < 1 or cols < 1 or len(path) > rows * cols:
        return False

    record = [False] * len(matrix)
    for i in range(rows):
        for j in range(cols):
            if matrix[i * cols + j] == path[0] and helper(i, j, path[1:]):
                return True
    return False


def topic_13_count_robot_moving(threshold, rows, cols):
    # backtracking
    def sum_digit(num):
        ans = 0
        while num > 0:
            ans += num % 10
            num = num // 10
        return ans

    def limit(i, j):
        return sum_digit(i) + sum_digit(j) <= threshold

    def helper(i, j):
        records[i * cols + j] = 1
        for ni, nj in [(i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1)]:
            if 0 <= ni < rows and 0 <= nj < cols and records[ni * cols + nj] == 0 and limit(ni, nj):
                helper(ni, nj)

    if threshold < 0:
        return 0
    records = [0] * (rows * cols)  # 1: visited 2: limited 0: unvisited
    helper(0, 0)
    return records.count(1)


def topic_14_cut_rope(number):
    if number == 0:
        return 0
    if number < 3:
        return 1
    dp = [0, 1, 2] + [0] * (number - 2)
    for i in range(3, number + 1):
        dp[i] = max([dp[j] * (i - j) for j in range(1, i)])
    print(dp)
    return dp[-1]


def topic_14_cut_rope_greedy(number):
    if number == 0:
        return 0
    if number < 3:
        return 1
    if number == 3:
        return 2
    return 3 ** (number // 3 - 1) * 4 if number % 3 == 1 else 3 ** (number // 3) * (number % 3)


def topic_15_number_0f_1(n):
    return sum([n >> i & 1 for i in range(0, 32)])


def topic_16_power(base, exponent):
    tag = True
    if exponent < 0:
        exponent, tag = -exponent, False
    ans = 1
    while exponent != 0:
        if exponent % 2 == 1:
            ans, exponent = ans * base, exponent - 1
        else:
            base, exponent = base * base, exponent // 2

    return ans if tag else 1 / ans


def topic_17_print_n_bit_number(n):
    def print_helper(ns):
        i = 0
        while i < n and ns[i] == 0:
            i += 1
        from functools import reduce
        print(reduce(lambda a, b: a * 10 + b, ns))

    def helper(number, index):
        if index == n:
            print_helper(number)
            return
        for i in range(10):
            number[index] = i
            helper(number, index + 1)

    number = [0] * n
    helper(number, 0)


def topic_18_delete_node_in_linkedlist(head, node):
    if not head or node == head:
        return None
    if not node.next:
        node = None
    node.val = node.next.val
    node.next = node.next.next
    return head


def topic_18_delete_repeated_node_in_linkedlist(head):
    if not head:
        return head
    newH = ListNode(-1)
    newH.next, t1, t2, tag = head, newH, head, True
    while t2.next:
        if t2.next.val != t2.val:
            if tag:
                t1, t2 = t1.next, t2.next
            else:
                t2 = t2.next
                t1.next = t2
            tag = True
        else:
            tag, t2 = False, t2.next
    if not tag:
        t1.next = None

    return newH.next


def topic_19_match(s, pattern):
    # there's a bug: '.*' should match everything
    if (not s and not pattern) or pattern == '.*':
        return True
    if (s and not pattern) or (not s and len(pattern) == 1):
        return False

    if len(pattern) > 1 and pattern[1] == '*':
        if not s or (s[0] != pattern[0] and pattern[0] != '.'):
            return topic_19_match(s, pattern[2:])
        i = 0
        while i < len(s) - 1:
            if s[i] == s[i + 1]:
                i += 1
            else:
                break
        if any([topic_19_match(s[j:], pattern[2:]) for j in range(i + 2)]):
            return True
        return False
    elif s[0] == pattern[0] or pattern[0] == '.':
        return topic_19_match(s[1:], pattern[1:])
    else:
        return False


def topic_20_is_numberic(s):
    def number(x):
        for i, c in enumerate(x):
            if c == '.':
                if i == len(x) - 1 or x[i + 1] in "-+":
                    return False
                if not x[:i] or x[:i] in ["-", "+"] and x[i + 1] not in "-+":
                    return integer(x[i + 1:])

                return integer(x[:i]) and integer(x[i + 1:])
        return integer(x)

    def integer(x):
        if not x:
            return False
        if x[0] in '+-':
            x = x[1:]
        if x[0] == '0' and len(x) > 1:
            return False
        for c in x:
            if c not in "0123456789":
                return False
        return True

    if not s:
        return False
    for i, c in enumerate(s):
        if c in 'eE':
            return number(s[:i]) and integer(s[i + 1:])
    return number(s)


def topic_20_is_numberic_(s):
    import re
    return re.match(r"^[\+\-]?[0-9]*(\.[0-9]*)?([eE][\+\-]?[0-9]+)?$", s)