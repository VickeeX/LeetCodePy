from collections import defaultdict, OrderedDict


class Node:
    def __init__(self, key, value, count):
        self.key, self.value, self.count = key, value, count


class LFUCache:
    # keycache: {key: Node}
    # cntcache: {count: OrderedDict{Node sorted by time}}  sorted by frequency then recently used time

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.keycache = {}
        self.cntcache = defaultdict(OrderedDict)
        self.count, self.mincnt = 0, 0

    def get(self, key: int) -> int:
        if key not in self.keycache:
            return -1
        node = self.keycache[key]
        del self.cntcache[node.count][key]

        if not self.cntcache[node.count]:
            del self.cntcache[node.count]
            if node.count == self.mincnt:
                self.mincnt += 1
        node.count += 1
        self.cntcache[node.count][key] = node
        return node.value

    def put(self, key: int, value: int) -> None:
        if not self.capacity:
            return
        if key in self.keycache:
            self.keycache[key].value = value
            self.get(key)
            return
        if self.count == self.capacity:
            k, _ = self.cntcache[self.mincnt].popitem(last=False)  # the head is least recently
            del self.keycache[k]
        else:
            self.count += 1

        self.keycache[key] = self.cntcache[1][key] = Node(key, value, 1)
        self.mincnt = 1
