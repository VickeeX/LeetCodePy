# -*- coding: utf-8 -*-

"""
    File name    :    NumMatrix
    Date         :    27/03/2019
    Description  :    {TODO}
    Author       :    VickeeX
"""


class NumMatrix:
    def __init__(self, matrix: list):
        self.sums = [[0] + [sum(matrix[i][:j + 1]) for j in range(len(matrix[i]))] for i in range(len(matrix))]
        # print(self.sums)

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return sum([self.sums[i][col2 + 1] - self.sums[i][col1] for i in range(row1, row2 + 1)])

        # Your NumMatrix object will be instantiated and called as such:
        # obj = NumMatrix(matrix)
        # param_1 = obj.sumRegion(row1,col1,row2,col2)
