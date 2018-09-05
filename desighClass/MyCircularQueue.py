# -*- coding: utf-8 -*-

"""
    File name    :    MyCircularQueue
    Date         :    13/08/2018
    Description  :    设计循环队列,其操作表现基于 FIFO（先进先出）原则并且队尾被连接在队首之后以形成一个循环。亦称为“环形缓冲器”。
    Author       :    VickeeX
"""


class MyCircularQueue:
    def __init__(self, k):
        """
        Initialize your data structure here. Set the size of the queue to be k.
        :type k: int
        """
        self.queue = []
        self.size = k
        self.eleNum = 0
        self.front = 0
        self.rear = 0

    def enQueue(self, value):
        """
        Insert an element into the circular queue. Return true if the operation is successful.
        :type value: int
        :rtype: bool
        """
        if len(self.queue) < self.size:
            self.queue.append(value)
            self.rear = len(self.queue) - 1
            self.eleNum += 1
        elif len(self.queue) == self.size:
            if self.eleNum == self.size:
                return False
            else:
                self.rear = (self.rear + 1) % self.size
                self.queue[self.rear] = value
                self.eleNum += 1
        return True

    def deQueue(self):
        """
        Delete an element from the circular queue. Return true if the operation is successful.
        :rtype: bool
        """
        if self.eleNum == 0:
            return False
        else:
            self.front = (self.front + 1) % self.size
            self.eleNum -= 1
            return True

    def Front(self):
        """
        Get the front item from the queue.
        :rtype: int
        """
        if self.eleNum == 0:
            return -1
        else:
            return self.queue[self.front]

    def Rear(self):
        """
        Get the last item from the queue.
        :rtype: int
        """
        if self.eleNum == 0:
            return -1
        else:
            return self.queue[self.rear]

    def isEmpty(self):
        """
        Checks whether the circular queue is empty or not.
        :rtype: bool
        """
        return self.eleNum == 0

    def isFull(self):
        """
        Checks whether the circular queue is full or not.
        :rtype: bool
        """
        return self.eleNum == self.size
