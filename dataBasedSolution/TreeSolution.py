# -*- coding: utf-8 -*-

"""
    File name    :    TreeSolution
    Date         :    14/08/2018
    Description  :    {TODO}
    Author       :    VickeeX
"""

from collections import defaultdict


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
            print([node.val for node in stack])
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

    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        self.ans = -float('inf')  # cache current max_sum path (includes left+node+path)

        def dfs(node):  # return the max of left/right path
            if not node:
                return -float('inf')
            lv, rv = dfs(node.left), dfs(node.right)
            tmp_max = max(node.val, node.val + lv, node.val + rv)
            self.ans = max(self.ans, tmp_max, lv + node.val + rv)
            return tmp_max

        dfs(root)
        return self.ans

    def postorderTraversal(self, root):  # 145. Binary Tree Postorder Traversal
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) + [root.val]

    def rightSideView(self, root: TreeNode) -> list:
        self.ans = []

        def helper(layer):
            new_layer, self.ans = [], self.ans + [layer[-1].val]
            for node in layer:
                if node.left:
                    new_layer += [node.left]
                if node.right:
                    new_layer += [node.right]
            if new_layer:
                helper(new_layer)

        if root:
            helper([root])
        return self.ans

    def rightSideView_(self, root: TreeNode) -> list:
        if not root:
            return []
        stack, depth, ans = [[root, 1]], 0, []
        while stack:
            node, d = stack.pop()
            if d > depth:
                ans.append(node.val)
                depth = d
            if node.left:
                stack.append([node.left, d + 1])
            if node.right:
                stack.append([node.right, d + 1])
        return ans

    def numIslands(self, grid: list) -> int:
        ans = 0
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] == '1':
                    ans += 1
                    s = [[row, col]]
                    while s:
                        i, j = s.pop()
                        for ni, nj in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                            if -1 < ni < len(grid) and -1 < nj < len(grid[0]) and grid[ni][nj] == '1':
                                s.append([ni, nj])
                                grid[ni][nj] = '0'
        return ans

    def countNodes_(self, root: TreeNode) -> int:
        if not root:
            return 0
        ans = 1
        if root.left:
            ans += self.countNodes(root.left)
        if root.left:
            ans += self.countNodes(root.right)
        return ans

    def countNodes(self, root: TreeNode) -> int:
        def getHeight(node, is_left):
            h = 0
            while node:
                h += 1
                if is_left:
                    node = node.left
                else:
                    node = node.right
            return h

        if not root:
            return 0
        lh, rh = getHeight(root, True), getHeight(root, False)
        if lh == rh:
            return 2 ** lh - 1
        return 1 + self.countNodes(root.left) + self.countNodes(root.right)

    def kthSmallest_(self, root: TreeNode, k: int) -> int:
        def helper(node, count):
            if node.left:
                count, val = helper(node.left, count)
                if count == k:
                    return k, val
            count += 1
            if count == k:
                return k, node.val
            if node.right:
                count, val = helper(node.right, count)
                return count, val
            return count, node.val

        _, ans = helper(root, 0)
        return ans

    def kthSmallest(self, root: TreeNode, k: int) -> int:
        stack, tmp = [], root
        while True:
            while tmp:
                stack.append(tmp)
                tmp = tmp.left
            tmp = stack.pop()
            k -= 1
            if k == 0:
                return tmp.val
            tmp = tmp.right

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root in (None, p, q): return root
        # if root.left in (p, q) and root.right in (p, q): return root
        l, r = (self.lowestCommonAncestor(node, p, q) for node in (root.left, root.right))
        return root if l and r else l or r

    def rob(self, root: TreeNode) -> int:
        # use records to memory traced path
        records = {}

        def helper(node):
            if not node:
                return 0
            if node in records.keys():
                return records[node]
            ans = 0
            if node.left:
                ans += helper(node.left.left) + helper(node.left.right)
            if node.right:
                ans += helper(node.right.left) + helper(node.right.right)
            ans = max(node.val + ans, helper(node.left) + helper(node.right))
            records[node] = ans
            return ans

        return helper(root)

    def rob_(self, root: TreeNode) -> int:
        def helper(node):
            if not node:
                return 0, 0
            l0, l1 = helper(node.left)
            r0, r1 = helper(node.right)
            # the greedy max without node, greedy max with node
            return max(l0, l1) + max(r0, r1), node.val + l0 + r0

        return max(helper(root))

    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if not root: return root
        if root.val > key:
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        else:  # the operation node
            if not root.right:
                return root.left
            if not root.left:
                return root.right
            tmp, tmpv = root.right, root.right.val  # replace root.val with minimum val in root's right sub tree
            while tmp.left:
                tmp = tmp.left
                tmpv = tmp.val
            root.val = tmpv
            root.right = self.deleteNode(root.right, root.val)
        return root

    def findFrequentTreeSum(self, root: TreeNode) -> list:
        from collections import defaultdict
        self.sums = defaultdict(int)

        def dfs(node: TreeNode):
            if not node:
                return 0
            sum = dfs(node.left) + dfs(node.right) + node.val
            self.sums[sum] += 1
            return sum

        dfs(root)
        common = max([v for v in self.sums.values()])
        return [k for k in self.sums if self.sums[k] == common]

    def findBottomLeftValue(self, root: TreeNode) -> int:
        layer = [root]
        while layer:
            tmp = []
            for node in layer:
                if node.left: tmp.append(node.left)
                if node.right: tmp.append(node.right)
            if not tmp:
                break
            layer = tmp
        return layer[0].val

    def largestValues(self, root: TreeNode) -> list:
        if not root: return []
        layer, ans = [root], [root.val]
        while layer:
            tmp = []
            for node in layer:
                if node.left: tmp.append(node.left)
                if node.right: tmp.append(node.right)
            if not tmp:
                break
            layer = tmp
            ans.append(max([node.val for node in tmp]))
        return ans

    def addOneRow(self, root: TreeNode, v: int, d: int) -> TreeNode:
        # 623. Add One Row to Tree
        if d == 1:
            new = TreeNode(v)
            new.left = root
            return new
        layer = [root]
        while d > 1:
            d -= 1
            if d == 1:
                for node in layer:
                    l, r = node.left, node.right
                    node.left, node.right = TreeNode(v), TreeNode(v)
                    node.left.left, node.right.right = l, r
            else:
                tmp = []
                for node in layer:
                    if node.left: tmp.append(node.left)
                    if node.right: tmp.append(node.right)
                layer = tmp
        return root

    def constructMaximumBinaryTree(self, nums: list) -> TreeNode:
        # 654. Maximum Binary Tree
        if not nums:
            return None
        num = max(nums)
        idx, root = nums.index(num), TreeNode(num)
        root.left, root.right = self.constructMaximumBinaryTree(nums[:idx]), self.constructMaximumBinaryTree(
            nums[idx + 1:])
        return root

    def printTree(self, root: TreeNode) -> list:
        # 655. Print Binary Tree
        def findDepth(r: TreeNode):
            if not r:
                return 0
            return 1 + max(findDepth(r.left), findDepth(r.right))

        depth = findDepth(root)
        width, layer, d = 2 ** depth - 1, [root], 1
        ans = [[""] * width for _ in range(depth)]
        while d <= depth:
            tmp = []
            pos = 2 ** (depth - d) - 1
            for node in layer:
                if not node or not node.left:
                    tmp.append(None)
                else:
                    tmp.append(node.left)
                if not node or not node.right:
                    tmp.append(None)
                else:
                    tmp.append(node.right)
                if node:
                    ans[d - 1][pos] = str(node.val)
                pos += 2 ** (depth - d + 1)
            layer, d = tmp, d + 1
        return ans

    def findDuplicateSubtrees(self, root: TreeNode) -> list:
        # 652. Find Duplicate Subtrees
        dic, ans = defaultdict(int), []

        def helper(node):
            if not node:
                return '*'
            s = helper(node.left) + '/' + helper(node.right) + '/' + str(node.val)
            if dic[s] == 1:  # add once
                ans.append(node)
            dic[s] += 1
            return s

        return ans