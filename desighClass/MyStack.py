# -*- coding: utf-8 -*-

"""
    File name    :    MyStack
    Date         :    12/08/2018
    Description  :    使用队列的合法操作实现栈.
    Author       :    VickeeX
"""
from queue import Queue


class MyStack:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.q1 = Queue()
        self.q2 = Queue()

    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: void
        """
        self.q1.put(x)

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        while self.q1.qsize() > 1:
            self.q2.put(self.q1.get())
        tmp = self.q2
        self.q2 = self.q1
        self.q1 = tmp
        return self.q2.get()

    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        while self.q1.qsize() > 1:
            self.q2.put(self.q1.get())
        ans = self.q1.get()
        self.q2.put(ans)
        tmp = self.q1
        self.q1 = self.q2
        self.q2 = tmp
        return ans

    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        return self.q1.qsize() == 0
