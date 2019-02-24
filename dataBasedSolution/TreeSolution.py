# -*- coding: utf-8 -*-

"""
    File name    :    TreeSolution
    Date         :    14/08/2018
    Description  :    {TODO}
    Author       :    VickeeX
"""


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Node(object):
    def __init__(self, val, children):
        self.val = val
        self.children = children


class QuadNode:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight


class Employee:
    def __init__(self, id, importance, subordinates):
        # It's the unique id of each node.
        # unique id of this employee
        self.id = id
        # the importance value of this employee
        self.importance = importance
        # the id of direct subordinates
        self.subordinates = subordinates


class TreeSolution:
    def getImportance(self, employees, id):
        """
        :type employees: Employee
        :type id: int
        :rtype: int
        """
        dmap = {emp.id: emp for emp in employees}

        def imp(i):
            e = dmap[i]
            return e.importance + sum(imp(j) for j in e.subordinates)

        return imp(id)

    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if (root.val - p.val) * (root.val - q.val) <= 0:  # 左右子树各一个节点,或者其中一个节点为root
            return root
        elif p.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        else:
            return self.lowestCommonAncestor(root.right, p, q)

    def construct(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: QuadNode
        """

        def constructAsPoint(g, i, j, l):
            initv, same = g[i][j], True
            for x in range(i, i + l):
                for y in range(j, j + l):
                    if g[x][y] != initv:
                        same = False
                        break
                if not same:
                    break
            if same:
                return QuadNode(initv, True, None, None, None, None)
            newL = l // 2
            return QuadNode(0, False, constructAsPoint(g, i, j, newL), constructAsPoint(g, i, j + newL, newL),
                            constructAsPoint(g, i + newL, j, newL), constructAsPoint(g, i + newL, j + newL, newL))

        return constructAsPoint(grid, 0, 0, len(grid))

    def nTreeMaxDepth(self, root):
        """
        :type root: Node
        :rtype: int
        """
        if not root:
            return 0
        if not root.children:
            return 1
        return 1 + max(list(map(self.nTreeMaxDepth, root.children)))

    def nTreeLevelOrder(self, root):
        """
        :type root: Node
        :rtype: List[List[int]]
        """
        if not root:
            return []
        curNodes, ans = [], []
        curNodes.append(root)
        while curNodes:
            nextNodes, level = [], []
            for node in curNodes:
                level.append(node.val)
                for child in node.children:
                    nextNodes.append(child)
            curNodes = nextNodes
            ans.append(level)
        return ans

    def nTreePreorder(self, root):
        """
        :type root: Node
        :rtype: List[int]
        """
        if not root:
            return []
        ans = [root.val]
        if root.children:
            for child in root.children:
                ans += self.nTreePreorder(child)
        return ans

    def nTreePostorder(self, root):
        """
        :type root: Node
        :rtype: List[int]
        """
        if not root:
            return []
        ans = []
        if root.children:
            for child in root.children:
                ans += self.nTreePostorder(child)
        return ans + [root.val]

    def minDiffInBST(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        maxL, minR, ans = root.left, root.right, 0x7FFFFFFF
        if maxL:
            while maxL.right:
                maxL = maxL.right
            ans = min(ans, root.val - maxL.val, self.minDiffInBST(root.left))
        if minR:
            while minR.left:
                minR = minR.left
            ans = min(ans, minR.val - root.val, self.minDiffInBST(root.right))
        return ans

    def leafSimilar(self, root1, root2):
        """
        :type root1: TreeNode
        :type root2: TreeNode
        :rtype: bool
        """

        def ergodicLeaf(root):
            if not root:
                return []
            if not root.left and not root.right:
                return [root.val]
            return ergodicLeaf(root.left) + ergodicLeaf(root.right)

        r1 = ergodicLeaf(root1)
        r2 = ergodicLeaf(root2)
        if len(r1) != len(r2):
            return False
        for i in range(len(r1)):
            if r1[i] != r2[i]:
                return False
        return True
        # return ergodicLeaf(root1) == ergodicLeaf(root2)

    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)

    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        if n == 0:
            return []
        self.ht = {}
        return self.generateBSTHelper(1, n)

    def generateBSTHelper(self, i, j):
        if i > j:
            return [None]
        if (i, j) in self.ht:
            return self.ht[(i, j)]

        tmp = []
        for k in range(i, j + 1):
            l, r = self.generateBSTHelper(i, k - 1), self.generateBSTHelper(k + 1, j)
            for tl in l:
                for tr in r:
                    root = TreeNode(k)
                    root.left, root.right = tl, tr
                    tmp.append(root)

        self.ht[(i, j)] = tmp
        return tmp

    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 递归超时,故通过数组存储减少计算次数
        ans = [1, 1, 2]
        for i in range(3, n + 1):
            ans.append(sum([ans[x - 1] * ans[i - x] for x in range(1, i + 1)]))
        return ans[n]

    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        self.ans = []
        self.levelTraversalWithDirection(0, [root])
        return self.ans

    def levelTraversalWithDirection(self, direc, nodes):
        if not nodes:
            return
        ta, next_level = [], []
        for n in nodes:
            ta.append(n.val)
            if n.left:
                next_level.append(n.left)
            if n.right:
                next_level.append(n.right)
        if direc == 1:
            ta.reverse()
        direc = 1 - direc
        self.ans.append(ta)
        self.levelTraversalWithDirection(direc, next_level)

    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if not preorder:
            return None
        root = TreeNode(preorder[0])
        for i in range(len(inorder)):
            if inorder[i] == preorder[0]:
                root.left = self.buildTree(preorder[1:i + 1], inorder[:i])
                root.right = self.buildTree(preorder[i + 1:], inorder[i + 1:])
        return root

    def buildTree1(self, preorder, inorder):
        if not preorder:
            return None

        root = TreeNode(preorder[0])
        stack = [root]
        i, j = 1, 0

        while i < len(preorder):
            tmp = None
            node = TreeNode(preorder[i])
            while stack and stack[-1].val == inorder[j]:
                tmp = stack.pop()
                j += 1
            if tmp:
                tmp.right = node
            else:
                stack[-1].left = node

            stack.append(node)
            i += 1
        return root

    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        if not inorder:
            return None
        root = TreeNode(postorder[-1])
        for i in range(len(inorder)):
            if inorder[i] == postorder[-1]:
                root.left = self.buildTree(inorder[:i], postorder[:i])
                root.right = self.buildTree(inorder[i + 1:], postorder[i:-1])
        return root

    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        if not head:
            return head
        self.nodes, cur = [], head
        while cur:
            self.nodes.append(cur.val)
            cur = cur.next
        return self.toBST(0, len(self.nodes))

    def toBST(self, start, end):
        if start == end:
            return None
        mid = (start + end) // 2
        th = TreeNode(self.nodes[mid])
        th.left, th.right = self.toBST(start, mid), self.toBST(mid + 1, end)
        return th

    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        self.ans = []

        def pathTraversal(node, s, path):
            if not node:
                return
            path.append(node.val)
            if not node.left and not node.right and node.val == s:
                self.ans.append(path)
                return
            if node.left:
                pathTraversal(node.left, s - node.val, path.copy())
            if node.right:
                pathTraversal(node.right, s - node.val, path.copy())

        pathTraversal(root, sum, [])
        return self.ans

    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        nodes = []

        def flattenHelper(root):
            if root:
                nodes.append(root)
                flattenHelper(root.left)
                flattenHelper(root.right)

        flattenHelper(root)
        if root:
            root.left, cur = None, root
        for n in nodes[1:]:
            cur.right = n
            n.left = None
            cur = cur.right


            # Definition for binary tree with next pointer.
            #  class TreeLinkNode:
            #     def __init__(self, x):
            #         self.val = x
            #         self.left = None
            #         self.right = None
            #         self.next = None

    # @param root, a tree link node
    # @return nothing
    def connect(self, root):
        if not root:
            return root
        first, cur = root.left, root
        while first:
            while cur:
                cur.left.next = cur.right
                if cur.next:
                    cur.right.next = cur.next.left
                cur = cur.next
            cur = first
            first = first.left

    def connect(self, root):
        first, pre, head = root, None, None
        while first:
            if first.left:
                if pre:
                    pre.next = first.left
                pre = first.left
            if first.right:
                if pre:
                    pre.next = first.right
                pre = first.right
            if not head:
                head = first.left or first.right
            if first.next:
                first = first.next
            else:
                first = head
                head = None
                pre = None

    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        self.tree_sum = 0

        def sumHelper(s, node):
            tmp = s * 10 + node.val
            if not node.left and not node.right:
                self.tree_sum += tmp
                return
            if node.left:
                sumHelper(tmp, node.left)
            if node.right:
                sumHelper(tmp, node.right)

        sumHelper(0, root)
        return self.tree_sum

    def preorderTraversalIteral(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        ans, stack = [], [root]
        while stack:
            tmp = stack.pop()
            ans.append(tmp.val)
            if tmp.right:
                stack.append(tmp.right)
            if tmp.left:
                stack.append(tmp.left)
        return ans

    def recoverTree(self, root):
        """
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        """
        self.s, nums = [], []

        def mid_order(tmp):
            if not tmp:
                return
            if tmp.left:
                mid_order(tmp.left)
            self.s += [tmp.val]
            if tmp.right:
                mid_order(tmp.right)
            return

        def swap(tmp, n1, n2):
            if not tmp:
                return
            if tmp.val == n1:
                tmp.val = n2
            elif tmp.val == n2:
                tmp.val = n1
            swap(tmp.left, n1, n2)
            swap(tmp.right, n1, n2)

        mid_order(root)
        for i, n in enumerate(sorted(self.s)):  # find the nums(not in sorted) to swap
            if n != self.s[i]:
                nums += [n]
        swap(root, nums[0], nums[1])

    def recoverTreeStack(self, root):
        swap = {0: None, 1: None}
        prev = TreeNode(float('-inf'))
        stack = [root]
        while stack:
            print([node.val for node in stack ])
            if stack[-1].left:
                stack.append(stack[-1].left)
            else:
                while stack:
                    node = stack.pop()
                    if not swap[0] and prev.val >= node.val:
                        swap[0] = prev
                    if swap[0] and prev.val >= node.val:
                        swap[1] = node
                    prev = node
                    if node.right:
                        stack.append(node.right)
                        break
        swap[1].val, swap[0].val = swap[0].val, swap[1].val
