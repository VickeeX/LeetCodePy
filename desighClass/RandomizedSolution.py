# -*- coding: utf-8 -*-

"""
    File name    :    RandomizedSet
    Date         :    17/05/2019
    Description  :    {TODO}
    Author       :    VickeeX
"""
import random, collections


class RandomizedSet:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.nums, self.pos = [], {}

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val not in self.pos:
            self.nums.append(val)
            self.pos[val] = len(self.nums) - 1
            return True
        return False

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val in self.pos:
            idx, last = self.pos[val], self.nums[-1]
            self.nums[idx], self.pos[last] = last, idx
            self.nums.pop()
            del self.pos[val]
            return True
        return False

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        return self.nums[random.randint(0, len(self.nums) - 1)]


class RandomizedCollection(object):

    def __init__(self):
        self.vals, self.idxs = [], collections.defaultdict(set)

    def insert(self, val):
        self.vals.append(val)
        self.idxs[val].add(len(self.vals) - 1)
        return len(self.idxs[val]) == 1

    def remove(self, val):
        if self.idxs[val]:
            out, ins = self.idxs[val].pop(), self.vals[-1]
            self.vals[out] = ins
            if self.idxs[ins]:
                self.idxs[ins].add(out)
                self.idxs[ins].discard(len(self.vals) - 1)
            self.vals.pop()
            return True
        return False

    def getRandom(self):
        return random.choice(self.vals)


class RandoPickIndexSolution:

    #     def __init__(self, nums: List[int]):
    #         self.dict = defaultdict(list)
    #         for i,n in enumerate(nums):
    #             self.dict[n].append(i)

    #     def pick(self, target: int) -> int:
    #         idx = random.randint(0, len(self.dict[target])-1)
    #         return self.dict[target][idx]

    def __init__(self, nums: list):
        self.nums = nums

    def pick(self, target: int) -> int:
        return random.choice([i for i, n in enumerate(self.nums) if n == target])


class RandomFlipMatrixSolution:
    # 519. Random Flip Matrix
    def __init__(self, n_rows: int, n_cols: int):
        self.rows, self.cows = n_rows, n_cols
        self.last = n_rows * n_cols
        self.rec = {}
        # keep the matching of num to pos index

    def flip(self) -> list:
        choice = random.randint(0, self.last - 1)
        index = self.rec.get(choice, choice)  # if choice is the last and modified before
        self.last -= 1
        self.rec[choice] = self.rec.get(self.last, self.last)  # modify choice's matching index
        return list(divmod(index, self.cows))

    def reset(self) -> None:
        self.last = self.rows * self.cows
        self.rec = {}


class RandomPickWithWeightSolution:
    # 528. Random Pick with Weight
    def __init__(self, w: list):
        self.sum = 0
        self.w = []
        for n in w:
            self.sum += n
            self.w.append(self.sum)

    def pickIndex(self) -> int:
        import bisect
        choice = random.randint(1, self.sum)
        return bisect.bisect_left(self.w, choice)
