"""
Set Matrix Zeroes

Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in place.
"""
"""
store states of each row in the first of that row, and store states of each column in the first of that column.
Because the state of row0 and the state of column0 would occupy the same cell, I let it be the state of row0, and use another variable "col0" for column0.

In the first phase, use matrix elements to set states in a top-down way.
In the second phase, use states to set matrix elements in a bottom-up way.
"""
def setZeroes(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: void Do not return anything, modify matrix in-place instead.
    """
    rows, cols = len(matrix), len(matrix[0])
    col0 = 1

    for i in range(rows):
        if matrix[i][0] == 0:  # col0 = 0 if one value in column 0 = 0
            col0 = 0
        for j in range(1, cols):
            if matrix[i][j] == 0:
                matrix[i][0] = matrix[0][j] = 0
    for i in range(rows - 1, -1, -1):
        for j in range(cols - 1, 0, -1):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
        if col0 == 0:
            matrix[i][0] = 0

"""
Rotate Image

You are given an n x n 2D matrix representing an image.

Rotate the image by 90 degrees (clockwise).
"""
"""
clockwise rotate
first reverse up to down, then swap the symmetry
1 2 3     7 8 9     7 4 1
4 5 6  => 4 5 6  => 8 5 2
7 8 9     1 2 3     9 6 3
"""
def rotate(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: void Do not return anything, modify matrix in-place instead.
    """
    # reverse up to down
    for i in range(len(matrix) / 2):
        matrix[i], matrix[len(matrix) - i - 1] = matrix[len(matrix) - i - 1], matrix[i]
    # swap the symmetry
    for i in range(len(matrix)):
        for j in range(i):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
