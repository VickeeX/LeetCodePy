# -*- coding: utf-8 -*-

"""
    File name    :    SumNumArray
    Date         :    29/03/2019
    Description  :    {TODO}
    Author       :    VickeeX
"""


# # a tricky solution
# class NumArray:
# def __init__(self, nums: list):
#     self.update = nums.__setitem__()
#     self.sumRange = lambda i, j: sum(nums[i:j + 1])


class SegmentNode:
    def __init__(self, start, end):
        self.start, self.end, self.sum = start, end, 0  # the start/end/sum of the interval
        self.left, self.right = None, None  # left/right interval


class NumArray:
    def __init__(self, nums: list):
        def buildTree(l, r):
            if l > r:  # irregular parameters
                return None
            if l == r:  # leaf node
                n = SegmentNode(l, r)
                n.sum = nums[l]
                return n
            mid, root = (l + r) // 2, SegmentNode(l, r)
            root.left, root.right = buildTree(l, mid), buildTree(mid + 1, r)  # recursively build the tree
            root.sum = root.left.sum + root.right.sum
            return root

        self.root = buildTree(0, len(nums) - 1)

    def update(self, i: int, val: int) -> None:
        def updateTree(root, i, val):
            if root.start == root.end:  # the leaf node to update
                root.sum = val
                return val
            mid = (root.start + root.end) // 2  # then recursively update the tree
            if i <= mid:
                updateTree(root.left, i, val)
            else:
                updateTree(root.right, i, val)
            root.sum = root.left.sum + root.right.sum
            return root.sum

        updateTree(self.root, i, val)

    def sumRange(self, i: int, j: int) -> int:
        def findNode(root, x, y):
            if root.start == x and root.end == y:
                return root.sum
            mid = (root.start + root.end) // 2
            if j <= mid:
                return findNode(root.left, x, y)
            if i > mid:
                return findNode(root.right, x, y)
            return findNode(root.left, x, mid) + findNode(root.right, mid + 1, y)

        return findNode(self.root, i, j)

# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# obj.update(i,val)
# param_2 = obj.sumRange(i,j)
