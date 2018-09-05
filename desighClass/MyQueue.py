# -*- coding: utf-8 -*-

"""
    File name    :    MyQueue
    Date         :    12/08/2018
    Description  :    使用栈的合法操作实现队列.
    Author       :    VickeeX
"""
from queue import deque


class MyQueue:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.s1 = deque()
        self.s2 = deque()

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: void
        """
        self.s1.appendleft(x)

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        while len(self.s1) > 1:
            self.s2.appendleft(self.s1.popleft())
        ans = self.s1.popleft()
        while len(self.s2) > 0:
            self.s1.appendleft(self.s2.popleft())
        return ans

    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        while len(self.s1) > 1:
            self.s2.appendleft(self.s1.popleft())
        ans = self.s1.popleft()
        self.s2.appendleft(ans)
        while len(self.s2) > 0:
            self.s1.appendleft(self.s2.popleft())
        return ans

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        return len(self.s1) == 0
