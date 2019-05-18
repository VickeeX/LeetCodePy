# -*- coding: utf-8 -*-

"""
    File name    :    RandomLinkedList
    Date         :    18/05/2019
    Description  :    {TODO}
    Author       :    VickeeX
"""
import random


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def __init__(self, head: ListNode):
        """
        @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node.
        """
        self.list = head
        self.len = 0
        tmp = head
        while tmp:
            self.len += 1
            tmp = tmp.next

    def getRandom(self) -> int:
        """
        Returns a random node's value.
        """
        choice, tmp = random.randint(0, self.len - 1), self.list
        while choice > 0:
            tmp = tmp.next
            choice -= 1
        return tmp.val




        # Your Solution object will be instantiated and called as such:
        # obj = Solution(head)
        # param_1 = obj.getRandom()
