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

        # map = {}
        # iterNode = head
        # while iterNode:
        #     map[iterNode] = RandomListNode(iterNode.label)
        #     iterNode = iterNode.next
        #
        # iterNode = head
        # while iterNode:
        #     map[iterNode].next = map[iterNode.next] if iterNode.next else None
        #     map[iterNode].random = map[iterNode.random] if iterNode.random else None
        #     iterNode = iterNode.next
        # return map[head] if head else None
