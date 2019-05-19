# -*- coding: utf-8 -*-

"""
    File name    :    ShuffleArray
    Date         :    18/05/2019
    Description  :    384. Shuffle an Array
    Author       :    VickeeX
"""
import random


class Solution:
    def __init__(self, nums: list):
        # # trick
        # self.reset = lambda: nums
        # self.shuffle = lambda: random.sample(nums, len(nums))
        self.nums = nums

    def reset(self) -> list:
        """
        Resets the array to its original configuration and return it.
        """
        return self.nums

    def shuffle(self) -> list:
        """
        Returns a random shuffling of the array.
        """
        # # random comparision: reduce the swap times for element
        # return sorted(self.nums, key=(lambda x: random.random()))
        count, l = 0, len(self.nums) - 1
        tmp = [i for i in self.nums]
        while count < l:
            choice = random.randint(count, l)
            tmp[count], tmp[choice] = tmp[choice], tmp[count]
            count += 1
        return tmp


        # Your Solution object will be instantiated and called as such:
        # obj = Solution(nums)
        # param_1 = obj.reset()
        # param_2 = obj.shuffle()
