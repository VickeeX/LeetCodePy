# -*- coding: utf-8 -*-

"""
    File name    :    MinStack
    Date         :    12/08/2018
    Description  :    设计一个支持 push，pop，top 操作，并能在常数时间内检索到最小元素的栈。
    Author       :    VickeeX
"""


class MinStack:
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.minStack = []

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.stack.append(x)
        if self.minStack == [] or x <= self.minStack[-1]:
            self.minStack.append(x)

    def pop(self):

        """
        :rtype: void
        """
        if self.stack:
            if self.stack[-1] == self.minStack[-1]:
                self.minStack.pop()
            self.stack.pop()

    def top(self):
        """
        :rtype: int
        """
        if self.stack:
            return self.stack[-1]
        return None

    def getMin(self):
        """
        :rtype: int
        """
        if self.minStack:
            return self.minStack[-1]
        return None
