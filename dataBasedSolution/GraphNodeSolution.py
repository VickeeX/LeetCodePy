# -*- coding: utf-8 -*-

"""
    File name    :    GraphNodeSolution
    Date         :    26/02/2019
    Description  :    {TODO}
    Author       :    VickeeX
"""

from collections import defaultdict


class Node(object):
    def __init__(self, val, neighbors):
        self.val = val
        self.neighbors = neighbors


class GraphNodeSolution(object):
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        res = defaultdict(lambda: Node(0, []))  # store the nodes.deepcopy
        nodes, seen = [node], set()  # nodes: un-traversed nodes, seen: traversed nodes
        while nodes:
            n = nodes.pop()
            res[n].val, res[n].neighbors = n.val, [res[x] for x in n.neighbors]
            nodes.extend(x for x in n.neighbors if x not in seen)
            seen.add(n)
        return res[node]

    def findMinHeightTrees(self, n: int, edges: list) -> list:
        if n == 1:
            return [0]
        adjacent = [set() for _ in range(n)]  # adjacent[i] is the set of i's neighbors
        for i, j in edges:
            adjacent[i].add(j)
            adjacent[j].add(i)

        leaves = [i for i in range(n) if len(adjacent[i]) == 1]
        while n > 2:  # at most two rooted trees can reach the min height
            n -= len(leaves)  # the left nums
            new = []
            for i in leaves:  # leaves cannot reach the min height in priority
                j = adjacent[i].pop()
                adjacent[j].remove(i)
                if len(adjacent[j]) == 1:
                    new.append(j)
                leaves = new
        return leaves


class Node:
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child


class NodeSolution:
    def flatten(self, head: 'Node') -> 'Node':
        if not head: return None
        dummy, stack = Node(0, None, None, None), [head]
        pre = dummy
        while stack:
            tmp = stack.pop()
            pre.next = tmp
            tmp.prev = pre

            if tmp.next: # the sort is not matter
                stack.append(tmp.next)
                tmp.next = None
            if tmp.child:
                stack.append(tmp.child)
                tmp.child = None
            pre = tmp
        dummy.next.prev = None
        return dummy.next
