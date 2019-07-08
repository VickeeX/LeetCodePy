# -*- coding: utf-8 -*-

"""
    File name    :    AllOneDict
    Date         :    08/07/2019
    Description  :    {TODO}
    Author       :    VickeeX
"""

from collections import defaultdict


class Node:
    def __init__(self):
        self.keys = set()
        self.prev = self.nxt = None

    def remove_key(self, key):
        self.keys.remove(key)

    def get_any(self):
        if self.keys:
            res = self.keys.pop()
            self.keys.add(res)
            return res
        return ""


class DoubleLinked():  # to store Node, each Node stores keys in matching numbers
    def __init__(self):
        self.head, self.tail = Node(), Node()
        self.head.nxt, self.tail.prev = self.tail, self.head

    def insert_after(self, x):
        node, tmp = Node(), x.nxt
        x.nxt, node.prev = node, x
        node.nxt, tmp.prev = tmp, node
        return node

    def insert_before(self, x):
        return self.insert_after(x.prev)

    def remove(self, node):
        prenode = node.prev
        prenode.nxt, node.nxt.prev = node.nxt, prenode


class AllOne:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.list = DoubleLinked()
        self.counter = defaultdict(int)
        self.nodes = {0: self.list.head}

    def rm_key_node(self, pre, key):
        node = self.nodes[pre]
        node.remove_key(key)
        if len(node.keys) == 0:
            self.list.remove(node)
            self.nodes.pop(pre)

    def inc(self, key: str) -> None:
        """
        Inserts a new key <Key> with value 1. Or increments an existing key by 1.
        """
        self.counter[key] += 1
        cur, pre = self.counter[key], self.counter[key] - 1
        if cur not in self.nodes:
            self.nodes[cur] = self.list.insert_after(self.nodes[pre])
        self.nodes[cur].keys.add(key)
        # print(self.nodes[cur].keys)
        if pre > 0:
            self.rm_key_node(pre, key)
        print(self.counter)

    def dec(self, key: str) -> None:
        """
        Decrements an existing key by 1. If Key's value is 1, remove it from the data structure.
        """
        if key in self.counter:
            self.counter[key] -= 1
            cur, pre = self.counter[key], self.counter[key] + 1
            if cur != 0:
                if cur not in self.nodes:
                    self.nodes[cur] = self.list.insert_before(self.nodes[pre])
                self.nodes[cur].keys.add(key)
            self.rm_key_node(pre, key)

    def getMaxKey(self) -> str:
        """
        Returns one of the keys with maximal value.
        """
        return self.list.tail.prev.get_any()

    def getMinKey(self) -> str:
        """
        Returns one of the keys with Minimal value.
        """
        return self.list.head.nxt.get_any()



        # Your AllOne object will be instantiated and called as such:
        # obj = AllOne()
        # obj.inc(key)
        # obj.dec(key)
        # param_3 = obj.getMaxKey()
        # param_4 = obj.getMinKey()
