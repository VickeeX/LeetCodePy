# -*- coding: utf-8 -*-

"""
    File name    :    LinkedListSolution
    Date         :    14/08/2018
    Description  :    {TODO}
    Author       :    VickeeX
"""


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class RandomListNode(object):
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None


class LinkedListSolution(object):
    def printLinked(self, node):
        cur = node
        while cur:
            print(cur.val)
            cur = cur.next

    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next

    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head: return False
        slow = fast = head
        while fast.next and fast.next.next:
            slow, fast = slow.next, fast.next.next
            if slow == fast:
                return True
        return False

    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        if not headA or not headB:
            return None
        pa, pb, enda, endb, c = headA, headB, headA, headB, 0
        while True:
            if c >= 2 and enda != endb:
                return None
            if pa == pb:
                return pa
            if pa.next:
                pa = pa.next
            else:
                enda, c = pa, c + 1
                pa = headB
            if pb.next:
                pb = pb.next
            else:
                endb, c = pb, c + 1
                pb = headA

    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        slow, fast, last, tag = head, head.next, head, False
        while slow and fast:
            if tag:
                last.next = fast
            else:
                head, tag = fast, True
            last = slow
            slow.next = fast.next
            fast.next = slow
            slow = slow.next
            if slow:
                fast = slow.next
        return head

    def printLinkedList(self, head):
        while head:
            print(head.val)
            head = head.next

    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if not head:
            return head
        size, tail = 1, head
        while tail.next:
            size += 1
            tail = tail.next
        k = size - k % size
        cur = head
        while k > 1:
            cur = cur.next
            k -= 1
        tail.next = head
        head = cur.next
        cur.next = None
        return head

    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        nhead = ListNode(0)
        nhead.next = head
        pre, cur = nhead, head

        while cur:
            while cur.next and cur.val == cur.next.val:
                cur = cur.next
            if pre.next == cur:
                pre = cur
            else:
                pre.next = cur.next
            cur = cur.next
        return nhead.next

    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        less, greater, cur = ListNode(0), ListNode(0), head
        cl, cg = less, greater
        while cur:
            if cur.val < x:
                cl.next = cur
                cl = cl.next
            else:
                cg.next = cur
                cg = cg.next
            cur = cur.next
        cl.next = greater.next
        cg.next = None
        return less.next

    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        nh = ListNode(0)
        nh.next = head
        pre, end = nh, nh
        count = 0
        while count < n:
            if count + 1 < m:
                pre = pre.next
            end = end.next
            count += 1
        start = pre.next
        pre.next = end
        while start != end:
            t1, t2 = end.next, start.next
            end.next = start
            start.next = t1
            start = t2
        return nh.next

    def copyRandomList(self, head):
        """
        :type head: RandomListNode
        :rtype: RandomListNode
        """
        # 遍历每个节点,将节点和对应的copy分别作为字典的键和值
        # 第一趟建立节点且赋值label
        # 第二趟赋值next和random
        m, cur = {}, head
        while cur:
            m[cur] = RandomListNode(cur.label)
            cur = cur.next
        cur = head
        while cur:
            m[cur].next = m[cur.next] if cur.next else None
            m[cur].random = m[cur.random] if cur.random else None
            cur = cur.next
        return m[head] if head else None

    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # x: steps from head to ans node(cycle start), y: steps from ans node to meet(slow and fast meets in this node)
        # when meets:
        #     slow goes (x+y) steps, fast goes 2*(x+y) steps
        #     fast goes one more cycle, a cycle consists of y and steps from meet to head
        #     2*(x+y) - (x+y) == x+y means steps from meet to head equals x : which explains the second while in code
        if not head:
            return None
        slow, fast, meet = head, head, None
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
            if slow == fast:
                meet = slow
                break
        while head and meet:
            if head == meet:
                return meet
            head, meet = head.next, meet.next

    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: void Do not return anything, modify head in-place instead.
        """
        # o(n) space and o(n) time
        if not head or not head.next:
            return
        mid, cur = head, head.next
        while cur and cur.next:
            mid = mid.next
            cur = cur.next.next
        cur, halfs = mid.next, []

        while cur:
            halfs = [cur.val] + halfs
            cur = cur.next
        mid.next = None

        cur = head
        for i in halfs:
            tmp = ListNode(i)
            tmp.next = cur.next
            cur.next = tmp
            cur = cur.next.next

    def reorderList1(self, head):
        # o(n) time with no extra space
        if not head:
            return None

        # 1st: find the mid
        fast, slow = head.next, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next

        # 2nd: reverse the second half
        run = slow.next
        slow.next, l1 = None, None
        while run:
            runN = run.next
            run.next = l1
            l1 = run
            run = runN

        # 3rd: insert one by one
        l0 = head
        while l1:
            l0n = l0.next
            l1n = l1.next

            l0.next = l1
            l1.next = l0n

            l0 = l0n
            l1 = l1n

    def insertionSortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        nh, cur = ListNode(0), head.next
        nh.next = ListNode(head.val)
        while cur:
            tmp = ListNode(cur.val)
            ncur = nh
            while ncur.next and ncur.next.val < tmp.val:
                ncur = ncur.next
            tmp.next = ncur.next
            ncur.next = tmp
            cur = cur.next
        return nh.next

    # def sortList(self, head):
    #     """
    #     :type head: ListNode
    #     :rtype: ListNode
    #     """
    #     def findMid(node):
    #         mid, fast = head, head.next
    #         while fast and fast.next:
    #             mid = mid.next
    #             fast = fast.next.next
    #         return mid
    #
    #     def merge(h1,h2):
    #

    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        if not lists:
            return []
        empty_num, k, ans = 0, len(lists), []
        while True:
            first, pos, num, empty_num = True, -1, 0, 0
            for i in range(0, k):
                if not lists[i]:  # list is empty
                    empty_num += 1
                elif first or lists[i].val < num:  # first non-empty list or smaller number
                    num, pos, first = lists[i].val, i, False
            if lists[pos]:
                lists[pos] = lists[pos].next
            if empty_num < k:
                ans = ans + [num]
            else:
                return ans

    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        t, i = head, 0
        while t:
            t = t.next
            i += 1
        if i < k:  # the last group contained less than k elements does no change
            return head

        ans, tmp = head, head
        for i in range(0, k):
            tmp = tmp.next
        ans.next, head = self.reverseKGroup(tmp, k), head.next  # recursion
        for i in range(1, k):  # reverse the first group (k elements)
            tmp = head.next
            head.next = ans
            ans = head
            head = tmp
        return ans
