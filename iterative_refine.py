

"""
Trapping Rain Water II
https://leetcode.com/problems/trapping-rain-water-ii/

Given an m x n matrix of positive integers representing the height of each unit cell in a 2D elevation map,
compute the volume of water it is able to trap after raining.

Note:
Both m and n are less than 110. The height of each unit cell is greater than 0 and is less than 20,000.

Example:

Given the following 3x6 height map:
[
  [1,4,3,1,3,2],
  [3,2,1,3,2,4],
  [2,3,3,2,3,1]
]

Return 4.
"""
"""
Container With Most Water
https://leetcode.com/problems/container-with-most-water/

Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai).
n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0).
Find two lines, which together with x-axis forms a container, such that the container contains the most water.

Note: You may not slant the container and n is at least 2.
"""
def maxArea(self, height):
    """
    :type height: List[int]
    :rtype: int
    """
    """
    Use v[low, high] indicates the volume of container with low and high. suppose height[low] < height[high],
    then we move low to low+1, that means we ingored v[low, high-1],v[low, high-2],etc, if this is safe,
    then the algorithm is right, and it's obvious that v[low, high-1],high[low, high-2]......
    can't be larger than v[low, high] since its width can't be larger than high-low, and its height is limited by height[low]
    """

    l = len(height)
    low, high = 0, l - 1 # the idea is to move the index with lower height
    max_area = 0
    while low < high:
        max_area = max(max_area, (high - low) * min(height[low], height[high]))
        if height[low] < height[high]:
            low += 1
        else:
            high -= 1
    return max_area



"""
Trapping Rain Water
https://leetcode.com/problems/trapping-rain-water/

Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.

For example,
Given [0,1,0,2,1,0,1,3,2,1,2,1], return 6.
"""
def trap(self, height):
    """
    :type height: List[int]
    :rtype: int
    """
    """
     x   x x
    xx xxx x
    xxxxxxxx
    xxxxxxxx
    34233424
    return 4

    34  0
    for each value from index 1 to n-2
        find max_left, max_right (index 2 -> [4(1),4(5)])
            min_lr = min(max_left, max_right)
            area = 4-2 + 4-3 + 4-3
            max_area = max(area, max_area)
        jump to index of max_right + 1

    O(n) time
    O(n) space --> can be reduced to O(1)
    """
    l = len(height)
    if l <= 2:
        return 0

    # pre-compute max_left and max_right array
    # max_left[i] the maximum values on the left of index i, including i
    # max_right[i] the maximum values on the right of index i, including i
    max_left, max_right = [0] * l, [0] * l
    max_left[0], max_right[-1] = height[0], height[-1]
    for i in range(1, l):
        max_left[i] = max(max_left[i - 1], height[i])
    for j in range(l - 2, -1, -1):
        max_right[j] = max(max_right[j + 1], height[j])

    area = 0
    i = 0
    while i <= l - 2:
        i += 1
        curr = height[i]
        diff = min(max_left[i], max_right[i]) - curr
        if diff <= 0:
            continue

        # we know for sure that diff > 0
        area += diff

    return area

"""
Permutations
https://leetcode.com/problems/permutations/

Given a collection of distinct numbers, return all possible permutations.

For example,
[1,2,3] have the following permutations:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
"""
def permute(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """

    """
    P[1,2,3]
        P[2,3] + [1]
        P[1,3] + [2]
        P[1,2] + [3]

    P[2,3]
        P[2] + P[3]
        P[3] + P[2]
    ...
    P[1] = [1]

    """
    # basecase
    # if len(nums) == 1:
    #     return [[nums[0]]]
    # res = []
    # for n in nums:
    #     without_n = [x for x in nums if x != n]
    #     res_temp = self.permute(without_n)
    #     for e in res_temp:
    #         res.append(e + [n])
    # return res

    perms = [[]]
    for n in nums:
        new_perms = []
        for perm in perms:
            for i in range(len(perm) + 1):
                print (perm[:i] + [n] + perm[i:])
                new_perms.append(perm[:i] + [n] + perm[i:])  ###insert n
        print (new_perms)
        perms = new_perms
    return perms


"""
Power Set
http://www.geeksforgeeks.org/power-set/

Power Set Power set P(S) of a set S is the set of all subsets of S. For example S = {a, b, c} then P(s) = {{}, {a}, {b}, {c}, {a,b}, {a, c}, {b, c}, {a, b, c}}.

If S has n elements in it then P(s) will have 2^n elements

Time Complexity: O(n2^n)

This algorithm would have a problem when len(s) is very large
    --> can use recursive call to solve the problem
"""
def subsets(s):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    # Using bit manimupation
    # res_size = 2**len(s)
    # res = []
    # for i in range(res_size):
    #     curr = []
    #     for j in range(len(s)):
    #         if i & (1 << j):
    #             curr.append(s[j])
    #     res.append(curr)
    # return res

    # recursive call
    # if not s:
    #     return [[]]
    # subsets = self.subsets(s[:-1])
    # return subsets + [sub + [s[-1]] for sub in subsets]

    # iterative
    res = [[]]
    for i in range(len(s)):
        res = res + [sub + [s[i]] for sub in res]
    return res


"""
Matrix Spiral Print

Given a 2D array (matrix) named M, print all items of M in a spiral order, clockwise.
For example:

M  =  1   2   3   4   5
       6   7   8   9  10
      11  12  13  14  15
      16  17  18  19  20

The clockwise spiral print is:  1 2 3 4 5 10 15 20 19 18 17 16 11 6 7 8 9 14 13 12

Let M be a matrix of m rows and n columns.
The spiral print can be implemented by repeating a print of 4 edges, in a converging manner:

Print the uppermost row from left to right
Print the rightmost column from top to bottom
Print the lowermost row from right to left
Print the leftmost column from bottom to top
To direct the spiral order and figure what is the next row/column to print we maintain 4 indices:

topRow - index of the the upper most row to be printed, starting from 0 and incrementing
btmRow - index of the the lowermost row to be printed, stating from m-1 and decrementing
leftCol - index of the leftmost column to be printed, starting from 0 and incrementing
rightCol - index of the the rightmost row to be printed, starting from n-1 and decrementing

Runtime Complexity: iterating over nm cells and printing them takes O(nm).

Space Complexity: using a constant number of indices (4), therefore: O(1).
"""

def print_spiral_matrix(M):
    top, bottom, left, right = 0, len(M) - 1, 0, len(M[0]) - 1

    while top <= bottom and left <= right:
        # print next top row
        for i in range(left, right + 1):
            print (M[top][i])
        top += 1

        # print next right hand side col
        for i in range(top, bottom + 1):
            print (M[i][right])
        right -= 1

        # print next bottom row
        if top <= bottom:
            for i in range(right, left-1, -1):
                print (M[bottom][i])
            bottom -= 1

        # print next left hand side col
        if left <= right:
            for i in range(bottom, top-1, -1):
                print (M[i][left])
            left += 1

# M = [[1,   2,   3,   4,   5], [6,   7,   8,   9,  10], [11,  12,  13,  14,  15], [16,  17, 18,  19,  20]]
# print print_spiral_matrix(M)


"""
Count Negative Numbers in a Column-Wise and Row-Wise Sorted Matrix
http://www.geeksforgeeks.org/count-negative-numbers-in-a-column-wise-row-wise-sorted-matrix/

Find the number of negative numbers in a column-wise / row-wise sorted matrix M[][]. Suppose M has n rows and m columns.

Example:

Input:  M =  [-3, -2, -1,  1]
             [-2,  2,  3,  4]
             [4,   5,  7,  8]
Output : 4
We have 4 negative numbers in this matrix

Optimal Solution

We start from the top right corner and find the position of the last negative number in the first row.
Using this information, we find the position of the last negative number in the second row.
We keep repeating this process until we either run out of negative numbers or we get to the last row.
"""

def countNegative(M, n, m):
    count = 0  # initialize result

    # Start with top right corner
    i = 0   # current row
    j = m - 1   # current col

    # Follow the path shown using arrows above
    while j >= 0 and i < n:
        if M[i][j] < 0:
            # j is the index of the last negative number
            # in this row.  So there must be ( j+1 ) negative numbers in this row.
            count += (j + 1)

            i += 1
        else:
            # move to the left and see if we can find a negative number there
            j -= 1
    return count

