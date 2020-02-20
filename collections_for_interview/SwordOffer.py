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
