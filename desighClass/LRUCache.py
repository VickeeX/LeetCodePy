# -*- coding: utf-8 -*-

"""
    File name    :    LRUCache
    Date         :    26/02/2019
    Description  :    {TODO}
    Author       :    VickeeX
"""
from collections import OrderedDict


class LRUCache(object):
    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.cache = OrderedDict()
        self.c = capacity

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        val = -1
        if key in self.cache:
            val = self.cache[key]
            self.cache = self.cache.pop(key)
        return val

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        if key in self.cache:
            del self.cache[key]
        self.cache[key] = value
        if len(self.cache) > self.c:
            self.cache.popitem(last=False)



            # Your LRUCache object will be instantiated and called as such:
            # obj = LRUCache(capacity)
            # param_1 = obj.get(key)
            # obj.put(key,value)
