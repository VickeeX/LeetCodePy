# -*- coding: utf-8 -*-

"""
    @File name    :    SwordOffer.py
    @Date         :    2020-02-19 19:09
    @Description  :    {TODO}
    @Author       :    VickeeX
"""
import heapq


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


def topic_21_reorder_array_by_parity(array):
    if len(array) < 2:
        return array
    i, j = 0, len(array) - 1
    while i < j:
        if array[i] % 2 == 1:
            i += 1
        else:
            array[i], array[j] = array[j], array[i]
            j -= 1
    return array


def topic_21_reorder_array_by_parity_keep(array):
    if len(array) < 2:
        return array
    i = 0
    while i != len(array):
        if array[i] % 2 == 1:
            tmp = i
            for j in range(i - 1, -1, -1):
                if array[j] % 2 == 0:
                    array[tmp], array[j] = array[j], array[tmp]
                    tmp = j
        i += 1
    return array


def topic_22_find_kth_to_tail(head, k):
    # 1 2 3 4 5
    if not head or k < 1:
        return None
    slow, faster = head, head
    while k > 1 and faster.next:
        faster, k = faster.next, k - 1
    if k > 1:
        return None
    while faster.next:
        slow, faster = slow.next, faster.next
    return slow


def topic_23_entry_node_of_loop(head):
    if not head or not head.next:
        return None
    slow, fast, step = head, head.next, 1
    while slow != fast:
        if not fast.next or not fast.next.next:
            return None
        slow, fast, step = slow.next, fast.next.next, step + 1
    # the linked list has a ring of length step

    slow, fast = head, head
    while step > 0:
        fast = fast.next
        step -= 1
    while slow != fast:
        slow = slow.next
        fast = fast.next
    return slow


def topic_24_revers_linked_list(head):
    # 1—>2—>3—>4—>5—>6
    # t1 t2
    if not head or not head.next:
        return head
    pre, t1, t2, bak = None, head, head.next, None
    while t2.next:
        bak = t2.next
        t1.next = pre
        t2.next = t1
        pre = t1
        t1, t2 = t2, bak
    t2.next = t1
    return t2


def topic_25_merge_sorted_linked_list(h1, h2):
    t1, t2, ans = h1, h2, ListNode(-1)
    tmp = ans
    while t1 and t2:
        if t1.val < t2.val:
            tmp.next, t1 = t1, t1.next
        else:
            tmp.next, t2 = t2, t2.next
        tmp = tmp.next
    while t1:
        tmp.next, t1 = t1, t1.next
        tmp = tmp.next
    while t2:
        tmp.next, t2 = t2, t2.next
        tmp = tmp.next
    return ans.next


def topic_26_has_sub_tree(t1, t2):
    def check(n1, n2):
        if not n2:
            return True
        if not n1 or n1.val != n2.val:
            return False
        if (n2.left and not check(n1.left, n2.left)) or (n2.right and not check(n1.right, n2.right)):
            return False
        return True

    if not t2 or (not t1 and t2):  # bug: None should match None!
        return False
    if check(t1, t2) or (t1.left and topic_26_has_sub_tree(t1.left, t2)) or (
            t1.right and topic_26_has_sub_tree(t1.right, t2)):
        return True
    return False


def topic_27_mirror_of_binary_tree(root):
    if not root:
        return root
    root.left, root.right = topic_27_mirror_of_binary_tree(root.right), topic_27_mirror_of_binary_tree(root.left)
    return root


def topic_28_is_symmetrical_binary_tree(root):
    # # judgement of preorder (and rev_preorder) is incorrect: eg. {5,5,5,5,5,#,5}
    # def preorder(node):
    #     if not node:
    #         return
    #     preorder(node.left)
    #     ans.append(node.val)
    #     preorder(node.right)
    #
    # preorder(root)
    # return ans == ans[::-1]

    if not root:
        return True
    ans = [root]
    while any(ans):
        tmp = []
        for x in ans:
            if not x:
                tmp.append(None)
                tmp.append(None)
            else:
                tmp.append(x.left)
                tmp.append(x.right)
        vals = [n.val if n else '#' for n in tmp]
        if vals != vals[::-1]:
            return False
        ans = tmp
    return True


def topic_29_print_matrix_clockwise(matrix):
    if not matrix or not matrix[0]:
        return []
    h, w = len(matrix), len(matrix[0])
    circle = (min(h, w) + 1) // 2

    def print_circle(ans, c):
        if w - 2 * c == 1:
            ans += [matrix[i][c] for i in range(c, h - c)]
        elif h - 2 * c == 1:
            ans += [matrix[c][j] for j in range(c, w - c)]
        else:
            for j in range(c, w - c):
                ans.append(matrix[c][j])
            for i in range(c + 1, h - c):
                ans.append(matrix[i][w - c - 1])
            for j in range(w - c - 2, c - 1, -1):
                ans.append(matrix[h - c - 1][j])
            for i in range(h - c - 2, c, -1):
                ans.append(matrix[i][c])
        return ans

    ans = []
    for c in range(circle):
        ans = print_circle(ans, c)
    return ans


class Topic30MinStack:
    stack, minstack = [], []

    def push(self, node):
        self.stack.append(node)
        if not self.minstack or node <= self.minstack[-1]:
            self.minstack.append(node)

    def pop(self):
        if self.stack[-1] == self.minstack[-1]:
            self.minstack.pop()
        self.stack.pop()

    def top(self):
        return self.stack[-1] if self.stack else None

    def min(self):
        return self.minstack[-1] if self.minstack else None


def topic_31_is_stack_pop_order(p1, p2):
    stack, i, j = [], 0, 0
    while i < len(p2) and j < len(p1):
        stack.append(p1[j])
        j += 1
        while stack and stack[-1] == p2[i]:
            stack.pop()
            i += 1
    if i == len(p2):
        return True
    return False


def topic_32_print_tree_from_top_to_bottom(root):
    if not root:
        return []
    layer, ans = [root], []
    while layer:
        tmp = []
        for n in layer:
            ans.append(n.val)
            if n.left:
                tmp.append(n.left)
            if n.right:
                tmp.append(n.right)
        layer = tmp
    return ans


def topic_32_print_tree_from_top_to_bottom_reverse_line(root):
    if not root:
        return []
    layer, ans = [root], []
    while layer:
        tmp = []
        ans.append([])
        for n in layer:
            ans[-1].append(n.val)
            if n.left:
                tmp.append(n.left)
            if n.right:
                tmp.append(n.right)
        layer = tmp
    ans = [n if i % 2 == 0 else n[::-1] for i, n in enumerate(ans)]
    return ans


def topic_33_verify_postorder_squence_of_bst(sequence):
    # [7,4,6,5]
    if not sequence:
        return False
    if len(sequence) == 1:
        return True
    idx = -1
    for i in range(len(sequence) - 1):
        if sequence[i] > sequence[-1]:
            idx = i
            break
    if idx == -1:
        return topic_33_verify_postorder_squence_of_bst(sequence[:-1])
    elif idx == 0:
        return min(sequence[:-1]) > sequence[-1] and topic_33_verify_postorder_squence_of_bst(sequence[:-1])
    else:
        return min(sequence[idx:-1]) > sequence[-1] and \
               topic_33_verify_postorder_squence_of_bst(sequence[:idx]) and \
               topic_33_verify_postorder_squence_of_bst(sequence[idx:-1])


def topic_34_find_target_path_in_tree(root, target):
    # mark: path refers to "root to leaf"
    ans = []

    def helper(node, cur, left):
        if not node or node.val > left:
            return
        if node.val == left and not node.left and not node.right:
            ans.append(cur + [node.val])
        elif node.val < left:
            helper(node.left, cur + [node.val], left - node.val)
            helper(node.right, cur + [node.val], left - node.val)

    helper(root, [], target)
    ans.sort(key=lambda x: -len(x))
    return ans


class RandomListNode:
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None


def topic_35_clone_complex_linked_list(head):
    # time complexity: O(n)
    # space complexity: O(n)
    if not head:
        return None
    newh, ntmp, tmp = None, None, head
    randoms, clones = {}, {}

    while tmp:
        if tmp == head:
            newh = RandomListNode(head.label)
            ntmp = newh
            clones[head] = newh
        else:
            ntmp.next = RandomListNode(tmp.label)
            ntmp = ntmp.next
            clones[tmp] = ntmp
        if tmp.random:
            randoms[tmp] = tmp.random
        tmp = tmp.next

    for k, v in randoms.items():
        clones[k].random = clones[v]
    return newh


def topic_35_clone_complex_linked_list_1(head):
    # NOT PASSED !!!
    # time complexity: O(n)
    # space complexity: O(1)
    if not head:
        return head
    tmp = head
    # step 1: replciate node to its next position
    while tmp:
        t = tmp.next
        tmp.next = RandomListNode(tmp.label)
        tmp.next.next = t
        tmp = tmp.next.next

    # step 2: replicate random
    tmp = head
    while tmp:
        if tmp.random:
            tmp.next.random = tmp.random.next
        tmp = tmp.next.next

    # step 3: split new nodes
    newh = head.next
    t1, t2 = head, newh
    t1.next = None
    while t2.next:
        t1.next = t2.next
        t2.next = t2.next.next
        t1 = t1.next
        t2 = t2.next
    return newh


class Topic36ConvertTreeToLinkedList:
    last = None

    def Convert(self, root):
        # should use helper method: to avoid return node in sub-recursion
        if not root:
            return None
        if root.left:
            self.Convert(root.left)
        root.left = self.last
        if self.last:
            self.last.right = root
        self.last = root
        if root.right:
            self.Convert(root.right)

        while root and root.left:
            root = root.left
        return root


def topic_38_permutation(ss):
    if len(ss) == 1:
        return [ss]
    dic = set()
    for i in range(len(ss)):
        for ps in topic_38_permutation(ss[:i] + ss[i + 1:]):
            dic.add(ss[i] + ps)
    return sorted(list(dic))


def topic_38_permutation_1(ss):
    from itertools import permutations
    return sorted(list(set(map(''.join, permutations(ss))))) if ss else []


def topic_39_more_than_half_number(numbers):
    if not numbers:
        return 0
    num, count = numbers[0], 1
    for n in numbers[1:]:
        if n == num:
            count += 1
        else:
            count -= 1
            if count == 0:
                num, count = n, 1
    # check in case the number doesn't exist
    return num if numbers.count(num) > (len(numbers) // 2) else 0


def topic_40_get_least_k_numbers(nums, k):
    import heapq
    if k > len(nums) or k == 0:
        return []
    heap = []
    for n in nums:
        if len(heap) < k:
            heap.append(-n)
        else:
            heap.append(max(-n, heapq.heappop(heap)))
        heapq.heapify(heap)
    return sorted([-n for n in heap])


class Topic41MedianSolution:
    left, right = [], []

    def Insert(self, num):
        if not self.left:
            self.left.append(-num)
            heapq.heapify(self.left)
        else:
            tag = True
            if not self.right or len(self.left) > len(self.right):
                n = -self.left[0]
            else:
                n, tag = self.right[0], False
            n, num = min(n, num), max(n, num)
            if tag:
                self.left[0] = -n
                self.right.append(num)
            else:
                self.left.append(-n)
                self.right[0] = num
            heapq.heapify(self.left)
            heapq.heapify(self.right)

    def GetMedian(self):
        # if Python2,use "xx / 2.0"
        return -self.left[0] if len(self.left) > len(self.right) else (-self.left[0] + self.right[0]) / 2


def topic_42_find_greatest_sum_of_subArray(array):
    tmp = max(array)
    if tmp < 0:
        return tmp
    ans = s = 0
    for i, n in enumerate(array):
        s += n
        if s < 0:
            s = 0
        if s > ans:
            ans = s
    return ans


def topic_42_find_greatest_sum_of_subArray_dp(array):
    if not array:
        return 0
    dp = [array[0]]
    for n in array[1:]:
        dp.append(n if dp[-1] < 0 else dp[-1] + n)
    return max(dp)


def topic_43_number_of_1_between_1_and_n(n):
    ans, tmp, base = 0, n, 1
    while tmp:
        last = tmp % 10
        tmp = tmp // 10
        ans += tmp * base  # ones from "1 to 9" increasing
        if last == 1:
            ans += n % base + 1  # the one in match position
        elif last > 1:
            ans += base
        base *= 10
    return ans


def topic_44_numer_in_sequence(n):
    counts = [1]
    tmp, i = n, 0
    while tmp > 0:
        i += 1
        counts.append((10 ** i - 10 ** (i - 1)) * i)
        tmp -= counts[-1]
    # i is the bit number of n
    s = sum(counts[:-1])
    return [int(k) for k in str((n - s) // i + 10 ** (i - 1))][(n - sum(counts[:-1])) % i]


def topic_45_concat_min_number(numbers):
    # bug: if 0 appears ?
    from functools import cmp_to_key
    if not numbers:
        return ""
    # # python2
    # numbers = sorted(numbers, lambda x, y: int(str(x) + str(y)) - int(str(y) + str(x)))
    numbers = sorted(numbers, key=cmp_to_key(lambda x, y: int(str(x) + str(y)) - int(str(y) + str(x))))
    return int(''.join([str(n) for n in numbers]))


def topic_46_number_to_string(number):
    s = [int(c) for c in str(number)]
    dp = [1]
    for i, c in enumerate(s[1:], 1):
        if s[i - 1] * 10 + c > 25:
            dp.append(dp[i - 1])
        else:
            dp.append(dp[i - 1] + dp[i - 2])
    return dp


def topic_48_non_repeated_string(s):
    ans, cur = "", ""
    for i, c in enumerate(s):
        if c not in cur:
            cur += str(c)
            if len(cur) > len(ans):
                ans = cur
        else:
            cur = cur[cur.index(c) + 1:] + str(c)
    return ans


def topic_49_get_ugly_number(index):
    uglys = [1]
    idx = [0, 0, 0]
    base = [2, 3, 5]

    while index > 1:
        tmps = [uglys[idx[i]] * base[i] for i in range(3)]
        t = min(tmps)
        for i in range(3):
            if tmps[i] == t:
                idx[i] += 1
        uglys.append(t)
        index -= 1
    return uglys[- 1]


def topic_50_first_not_repeatingchar(s):
    records = [0] * 80
    for c in s:
        print(ord(c), ord(c) - 65)
        records[ord(c) - 65] += 1
    for i, c in enumerate(s):
        if records[ord(c) - 65] == 1:
            return i
    return -1


class Topic50FirstNotRepeatingCharStreamSolution:
    # records = [0] * 80
    # s = ""
    #
    # def FirstAppearingOnce(self):
    #     for i, c in enumerate(self.s):
    #         if self.records[ord(c) - 65] == 1:
    #             return c
    #     return '#'
    #
    # def Insert(self, char):
    #     self.records[ord(char) - 65] += 1
    #     self.s += str(char)
    #

    records = [0] * 80
    ans = []

    def FirstAppearingOnce(self):
        return self.ans[0] if self.ans else "#"

    def Insert(self, char):
        self.records[ord(char) - 65] += 1
        if self.records[ord(char) - 65] == 1:
            self.ans.append(char)
        elif char in self.ans:
            self.ans.remove(char)


def topic_51_inverse_pairs(data):
    def merge(i, j, l):
        count = 0
        i1, j1 = i, j
        tmp = []
        while i1 < i + l and j1 < min(j + l, end):
            if data[i1] < data[j1]:
                tmp.append(data[i1])
                i1 += 1
            else:
                tmp.append(data[j1])
                j1 += 1
                count += i + l - i1
        while i1 < i + l:
            tmp.append(data[i1])
            i1 += 1
        while j1 < min(j + l, end):
            tmp.append(data[j1])
            j1 += 1
        data[i:min(j + l, end)] = tmp
        return count

    if len(data) < 2:
        return 0
    end = len(data)
    count = 0
    l = 1
    while l < end:
        begins, i = [], 0
        while i < end:
            begins.append(i)
            i += l
        for i in range(1, len(begins), 2):
            count += merge(begins[i - 1], begins[i], l)
        l *= 2
    return count % 1000000007


def topic_51_inverse_pairs_recursive(self, data):
    if len(data) == 0:
        return 0

    def mergeSort(data, begin, end):
        if begin == end - 1:
            return 0
        mid = int((begin + end) / 2)
        left_count = mergeSort(data, begin, mid)
        right_count = mergeSort(data, mid, end)
        merge_count = merge(data, begin, mid, end)
        return left_count + right_count + merge_count

    def merge(data, begin, mid, end):
        i = begin
        j = mid
        count = 0
        temp = []
        while i < mid and j < end:
            if data[i] <= data[j]:
                temp.append(data[i])
                i += 1
            else:
                temp.append(data[j])
                j += 1
                count += mid - i
        while i < mid:
            temp.append(data[i])
            i += 1
        while j < end:
            temp.append(data[j])
            j += 1
        data[begin: end] = temp
        del temp
        return count

    begin = 0
    end = len(data)
    ans = mergeSort(data, begin, end)
    return ans % 1000000007


def topic_52_find_first_common_node(h1, h2):
    # s1 = []
    # tmp = h1
    # while tmp:
    #     s1.append(tmp)
    #     tmp = tmp.next
    # tmp = h2
    # while tmp:
    #     if tmp in s1:
    #         return tmp
    #     tmp = tmp.next
    # return None
    from collections import deque
    # reduce time complexity from O(m*n) to O(m+n)
    def travesal(head):
        tmp, nodes = head, deque()
        while tmp:
            nodes.append(tmp)
            tmp = tmp.next
        return nodes

    s1, s2, ans = travesal(h1), travesal(h2), None
    while s1 and s2:
        if s1[-1] == s2[-1]:
            ans = s1.pop()
            s2.pop()
        else:
            break
    return ans


def topic_53_num_frequency_in_sorted_array(data, k):
    # # trick: return data.count(k)
    # the following method reduce time complexity from O(n) to O(lgn)
    def find_first(l, r):
        if l > r:
            return -1
        mid = (l + r) // 2
        if data[mid] == k:
            if mid == 0 or data[mid - 1] != k:
                return mid
            else:
                r = mid - 1
        elif data[mid] < k:
            l = mid + 1
        else:
            r = mid - 1
        return find_first(l, r)

    def find_last(l, r):
        if l > r:
            return -1
        mid = (l + r) // 2
        if data[mid] == k:
            if mid == len(data) - 1 or data[mid + 1] != k:
                return mid
            else:
                l = mid + 1
        elif data[mid] < k:
            l = mid + 1
        else:
            r = mid - 1
        return find_last(l, r)

    a, b = find_first(0, len(data) - 1), find_last(0, len(data) - 1)
    if a != -1 and b != -1:
        return b - a + 1
    return 0


def topic_53_find_missing_number(numbers):
    if not numbers:
        return -1
    l, r = 0, len(numbers) - 1
    while l <= r:
        mid = (l + r) // 2
        if numbers[mid] != mid:
            if mid == 0 or numbers[mid - 1] == mid - 1:
                return mid
            r = mid - 1
        else:
            l = mid + 1
    if l == len(numbers):
        return l
    return -1


def topic_53_find_right_number(numbers):
    if not numbers:
        return -1
    l, r = 0, len(numbers) - 1
    while l <= r:
        mid = (l + r) // 2
        if numbers[mid] == mid:
            return mid
        elif numbers[mid] < mid:
            l = mid + 1
        else:
            r = mid - 1
    return -1


def topic_54_kth_node_in_tree(root, k):
    if not root or k == 0:
        return None

    def helper(node, n):
        ans = None
        if node.left:
            ans, n = helper(node.left, n)
        if n == 1:
            return node.val, -1
        n -= 1
        if not ans and node.right:
            ans, n = helper(node.right, n)
        return ans, n

    return helper(root, k)[0]


def topic_55_depth_of_tree(root):
    def dfs(node, h):
        if not node:
            return h
        h += 1
        return max(dfs(node.left, h), dfs(node.right, h))

    return dfs(root, 0)


def topic_55_balance_tree(root):
    def dfs(node, h):
        if not node:
            return h, True
        h += 1
        h1, f1 = dfs(node.left, h)
        h2, f2 = dfs(node.right, h)
        return max(h1, h2), f1 and f2 and -1 <= h1 - h2 <= 1

    return dfs(root, 0)[1]


def topic_56_find_single_nums(array):
    # TODO: unclear for bit operation
    from functools import reduce
    if len(array) < 2: return array
    xor = reduce(lambda x, y: x ^ y, array)
    if xor != 1:
        xor = xor & (~xor + 1)
    lst1, lst2 = [], []
    for i in array:
        if i & xor:
            lst1.append(i)
        else:
            lst2.append(i)
    return [reduce(lambda x, y: x ^ y, lst1), reduce(lambda x, y: x ^ y, lst2)]


def topic_56_single_nums(array):
    records = [0] * 32
    for n in array:
        for i, c in enumerate(bin(n)[:1:-1]):
            records[i] += 1 if c == '1' else 0
    return int("0b" + ''.join([str(n % 3) for n in records][::-1]), 2)


def topic_57_find_numbers_with_sum(array, sum):
    i, j = 0, len(array) - 1
    while i < j:
        s = array[i] + array[j]
        if s == sum:
            return array[i], array[j]
        elif s < sum:
            i += 1
        else:
            j -= 1
    return []
