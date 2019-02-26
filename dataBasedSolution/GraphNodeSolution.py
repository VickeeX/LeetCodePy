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
            nodes.extend(x for x in n.neighbors if x not in seen )
            seen.add(n)
        return res[node]
