# -*- coding: utf-8 -*-

"""
    File name    :    Codec
    Date         :    19/03/2019
    Description  :    {TODO}
    Author       :    VickeeX
"""


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return "null"
        ans = str(root.val)
        # if root.left:
        ans += "," + self.serialize(root.left)
        # if root.right:
        ans += "," + self.serialize(root.right)
        return ans
        # ans, layer = "", [root]
        # while layer:
        #     tmp, tag = [], False
        #     for n in layer:
        #         if n:
        #             tag = True
        #         if n and n.left:
        #             tmp.append(n.left)
        #         else:
        #             tmp.append(None)
        #         if n and n.right:
        #             tmp.append(n.right)
        #         else:
        #             tmp.append(None)
        #     if not tag:
        #         break
        #     ans += "," + ",".join([str(n.val) if n else "null" for n in layer])
        #     layer = tmp
        # return "[" + ans[1:] + "]"

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        nodes = data.split(",")
        for i, n in enumerate(nodes):
            nodes[i] = n if n == "null" else int(n)
        root, _ = self.buildTree(nodes, -1)
        return root

    def buildTree(self, data, pos):
        pos += 1
        if pos >= len(data) or data[pos] == "null":
            return None, pos
        root = TreeNode(data[pos])
        root.left, pos = self.buildTree(data, pos)
        root.right, pos = self.buildTree(data, pos)
        return root, pos

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))
