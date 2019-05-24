import random
from collections import defaultdict

class Solution:

#     def __init__(self, nums: List[int]):
#         self.dict = defaultdict(list)
#         for i,n in enumerate(nums):
#             self.dict[n].append(i)

#     def pick(self, target: int) -> int:
#         idx = random.randint(0, len(self.dict[target])-1)
#         return self.dict[target][idx]
        

    def __init__(self, nums: List[int]):
        self.nums = nums


    def pick(self, target: int) -> int:
        return random.choice([i for i,n in enumerate(self.nums) if n==target])
        

# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.pick(target)
