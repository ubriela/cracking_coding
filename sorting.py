"""
Top K Frequent Elements
https://leetcode.com/problems/top-k-frequent-elements
Given a non-empty array of integers, return the k most frequent elements.

For example,
Given [1,1,1,2,2,3] and k = 2, return [1,2].

Note:
You may assume k is always valid, 1 ≤ k ≤ number of unique elements.
Your algorithm's time complexity must be better than O(n log n), where n is the array's size.
"""
from collections import Counter
def topKFrequent(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: List[int]
    """
    c = Counter(nums)
    return [i[0] for i in Counter(nums).most_common(k)]
"""
Merge Intervals
https://leetcode.com/problems/merge-intervals/
Given a collection of intervals, merge all overlapping intervals.

For example,
Given [1,3],[2,6],[8,10],[15,18],
return [1,6],[8,10],[15,18].
"""
# Definition for an interval.
class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e

def merge(intervals):
    """
    :type intervals: List[Interval]
    :rtype: List[Interval]
    """
    out = []
    for i in sorted(intervals, key=lambda i: i.start):
        if out and i.start <= out[-1].end:
            out[-1].end = max(out[-1].end, i.end)
        else:
            out += i,
    return out


"""
3Sum
https://leetcode.com/problems/3sum/

Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

Note: The solution set must not contain duplicate triplets.

For example, given array S = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]
Show Company Tags
Show Tags
Show Similar Problems
"""

def threeSum(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    if not nums or len(nums) < 3:
        return []

    # O(N^2)
    # # find twoSum in neg that sum up to a value in pos
    # checked = [False] * len(nums)
    # res = set()
    # for j in range(len(nums)):
    #     p = nums[j]
    #     checked[j] = True
    #     d = {}
    #     # temp = nums[:j] + nums[j+1:]
    #     temp = [nums[k] for k in range(len(nums)) if not checked[k]]
    #     if len(temp) < 2:
    #         continue
    #     for i in range(len(temp)):
    #         v = temp[i]
    #         if v in d:
    #             s = tuple(sorted([v, temp[d[v]], p]))
    #             if s not in res:
    #                 res.add(s)
    #         else:
    #             d[-p - v] = i
    # return list(res)

    # O(N^2)
    res = []
    nums.sort()
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        l, r = i + 1, len(nums) - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0:
                l += 1
            elif s > 0:
                r -= 1
            else:
                res.append((nums[i], nums[l], nums[r]))
                while l < r and nums[l] == nums[l + 1]:
                    l += 1
                while l < r and nums[r] == nums[r - 1]:
                    r -= 1
                l += 1;
                r -= 1
    return res
"""
Group Anagrams

This is the code in python that I did after the interview:
Time complexity is O(NlogN) and space complexity is O(N) where N is the size of input string.

The assumption:
Strings are small --> first sorts take O(N)
As lens of strings are small (say 1--20), we can use faster sort algorithm such as bucket sort or radix sort, which take O(N)

So I think optimized code can be O(N) for both space and time complexity.
"""
from collections import defaultdict
def groupAnagrams(strs):
    """
    :type strs: List[str]
    :rtype: List[List[str]]
    """
    d = defaultdict(list)

    # add tuple of original string and sorted string
    for s in strs:
        d[''.join(sorted(s))].append(s)  # first sort

    # sort by length of string in reversed order
    # second sort
    sorted_l = sorted(d.items(), key=lambda x: len(x[1]), reverse=True)

    return [s[1] for s in sorted_l]


# l = ['abc','test','vac', 'bac', 'london', 'cba', 'cav', 'lon', 'pst']
# print (groupAnagrams(l))
# [['abc', 'bac', 'cba'], ['vac', 'cav'], ['test'], ['london'], ['lon'], ['pst']]


"""
Search a 2D Matrix II
https://leetcode.com/problems/search-a-2d-matrix-ii/sqrt(

Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

Integers in each row are sorted in ascending from left to right.
Integers in each column are sorted in ascending from top to bottom.
For example,

Consider the following matrix:

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
Given target = 5, return true.
"""
def searchMatrix(matrix, target):
    """
    :type matrix: List[List[int]]
    :type target: int
    :rtype: bool
    """
    """
      [1,   4,  7, 11, 15],
      [2,   5,  8, 12, 19],
      [3,   6,  9, 16, 22],
      [10, 13, 14, 17, 24],
      [18, 21, 23, 26, 30]

    Well, the idea is to search from the top-right element and then reduce the range for further searching by comparisons between target and the current element.
    """
    if not matrix or not matrix[0]:
        return False
    m, n = len(matrix), len(matrix[0])
    r, c = 0, n - 1
    while r < m and c >= 0:
        if matrix[r][c] == target:
            return True
        if matrix[r][c] > target:  # curr > target -> on left
            c -= 1
        else:  # curr < target -> below
            r += 1
    return False


"""
Obtain a sorted array from sorted matrix
use queue (BFS) with a heap

O(n^2logn)
"""
def sortedMatrix2SortedArr(matrix):
    """
      [1,   4,  7, 11, 15],
      [2,   5,  8, 12, 19],
      [3,   6,  9, 16, 22],
      [10, 13, 14, 17, 24],
      [18, 21, 23, 26, 30]

      q = 1         h = 1
      q = 2,4,      h = 1,2,4
      q = 4,3,5     h = 1,2,3,4,5
      q = 3,5,7     h = 1,2,3,4,5,7

      pick the smallest val to expand
      1
      2,4
      3,4,5
      4,5,6,10
      5,6,7,10
      6,7,8,10
    """
    if not matrix or not matrix[0]:
        return []
    row = len(matrix)
    col = len(matrix[0])
    res = []
    h = [(matrix[0][0], 0, 0)]  # (val, row, col)
    visited = set([(0, 0)])  # (row, col)
    while h:
        val, i, j = heapq.heappop(h)  # smallest
        res.append(val)

        if i + 1 < row and (i + 1, j) not in visited:
            visited.add((i + 1, j))
            heapq.heappush(h, (matrix[i + 1][j], i + 1, j))
        if j + 1 < col and (i, j + 1) not in visited:
            visited.add((i, j + 1))
            heapq.heappush(h, (matrix[i][j + 1], i, j + 1))

    return res

"""
Kth Smallest Element in a Sorted Matrix
https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/
Given a n x n matrix where each of the rows and columns are sorted in ascending order, find the kth smallest element in the matrix.

Note that it is the kth smallest element in the sorted order, not the kth distinct element.

Example:

matrix = [
   [ 1,  5,  9],
   [10, 11, 13],
   [12, 13, 15]
],
k = 8,

return 13.
Note:
You may assume k is always valid, 1 <= k <= n^2.

if k << n then using a visited set would be helpful
"""
def kthSmallest(matrix, k):
    """
    :type matrix: List[List[int]]
    :type k: int
    :rtype: int
    """

    """
    Since the matrix is sorted, we do not need to put all the elements in heap at one time. We can simply pop and put for k times.
    By observation, if we look at the matrix diagonally, we can tell that if we do not pop matrix[i][j],
    we do not need to put on matrix[i][j + 1] and matrix[i + 1][j] since they are bigger.

    e.g., given the matrix below:
    1 2 4
    3 5 7
    6 8 9
    We put 1 first, then pop 1 and put 2 and 3, then pop 2 and put 4 and 5, then pop 3 and put 6
    """
    if not matrix or not matrix[0]:
        return None
    row = len(matrix)
    col = len(matrix[0])

    h = []
    heapq.heappush(h, (matrix[0][0], 0, 0))  # (val, row, col)
    val = 0
    while k > 0:
        val, i, j = heapq.heappop(h)  # smallest

        if i + 1 < row:  # and (i+1,j) not in visited:
            heapq.heappush(h, (matrix[i + 1][j], i + 1, j))
        if i == 0 and j + 1 < col:  # and (i,j+1) not in visited:
            heapq.heappush(h, (matrix[i][j + 1], i, j + 1))
        k -= 1
    return val

"""
The "Busiest Time in The Mall" Problem
https://www.pramp.com/join/zAgPBLllgOTyYKyJELb1

Given input which is a vector of log entries of some online system each entry is something like
(user_name, login_time, logout_time). Come up with an algorithm with outputs number of users logged in
the system at each time slot in the input. Output should contain only the time slot which are in the input.

e.g.,
input:
[1,3,4,6,8,10]
[
("Jane",1,4),
("Jin",3,6),
("Jun",8,10)
]

output:
[(1,1), (3, 2), (4, 1), (6, 0), (8,1), (10,0)]

Assumptions:
    input timestamp are sorted
    input log entries are not sorted

Baseline
convert the set of login and logout time to a set of tuples, where the second part of tuple indicates login/logout
    (1,0),(4,1),(3,0),(6,1),(8,0),(10,1)
sort log entries based on time
    (1,0),(3,0),(4,1),(6,1),(8,0),(10,1)
for each time in input: # [1,3,4,6,8,10]
    count number of login and logout
    people = #login - #logout

log_time = [1,3,4,5,6,8,10,11]
times = [(1,0),(3,0),(4,1),(6,1),(8,0),(10,1)]
output = [(1, 1), (3, 2), (4, 1), (5, 1), (6, 0), (8, 1), (10, 0), (11, 0)]
Test:
init i=0 ti=1
t=1
    1 <= 1: False
    1 == 1 and 0 == 0: True
        login = 1
        i = 1
    res = (1,1)
t=3
    3 <= 3: True
    First branch
        i = 2
        login = 2
    res = (1,1),(3,2)
t=4
    First branch
        logout = 1
        i = 3
    res = (1,1),(3,2),(4,1)
t=5
    False
    res = (1,1),(3,2),(4,1), (5,1)
t=6
    True
        logout = 2
        i = 4
    res = (1,1),(3,2),(4,1),(6,0),(5,1)
t=8
    True
        login = 3
        i = 5
    res = (1,1),(3,2),(4,1),(5,1),(6,0),(8,1)
t=10
    True
        logout = 3
        i = 6
    res = (1,1),(3,2),(4,1),(5,1),(6,0),(8,1),(10,0)
t=11
    False
    res = (1,1),(3,2),(4,1),(5,1),(6,0),(8,1),(10,0),(11,0)
"""

# O(nlogn + len(log_time)), n = 2*len(user_log)
def countOnlineUsers(user_log, log_time):
    times = []
    for u in user_log:
        times.append((u[1],0)) # login
        times.append((u[2],1)) # logout

    times.sort(key=lambda t : t[0])

    res = []
    i = 0   # index on times
    login, logout = 0, 0
    for t in log_time:
        while i < len(times) and times[i][0] <= t:
            # update counts
            if times[i][1] == 0:
                login += 1
            else:
                logout += 1
            i += 1
        res.append((t,login-logout))
    return res

# print count_online_users([
# ("Foo",0,11),
# ("Jane",1,4),
# ("Jin",3,6),
# ("Jun",8,10)
# ],
# [1,3,4,5,6,8,10,11]
# )

"""
Generate a random numerator given a list of ranges
"""
import bisect
def randomGen(ranges):
    # sort ranges by start point in increasing order
    ranges = sorted(ranges, key=lambda x : x[0])

    # merge all ranges
    mergedRanges = []
    curr = ranges[0]
    for range in ranges[1:]:
        if range[0] <= curr[1]:
            # update curr
            curr = (curr[0], range[1])
        else:
            mergedRanges.append(curr)
            curr = range
    mergedRanges.append(curr)

    # sizes of ranges
    sizes = [range[1] - range[0] + 1 for range in mergedRanges]

    # commulative sizes
    cumSizes = [sizes[0]]
    for size in sizes[1:]:
        cumSizes.append(cumSizes[-1] + size)

    # randIndex = random value between 1,cumSizes[-1]
    randIndex = random.randint(1,cumSizes[-1])
    index = bisect.bisect_left(cumSizes, randIndex)

    return random.randint(mergedRanges[index][0], mergedRanges[index][1])

# print randomGen([(1,2), (2,3), (4,5)])
# l = []
# for i in range(10000):
#     l.append(randomGen([(1,2), (2,3), (4,5)]))
# from collections import Counter
# print (Counter(l))

"""
Two Sum II - Input array is sorted
https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
Given an array of integers that is already sorted in ascending order,
find two numbers such that they add up to a specific target number.

The function twoSum should return indices of the two numbers such that they add up to the target,
where index1 must be less than index2. Please note that your returned answers
(both index1 and index2) are not zero-based.

You may assume that each input would have exactly one solution and you may not use the same element twice.

Input: numbers={2, 7, 11, 15}, target=9
Output: index1=1, index2=2

Show Company Tags
Show Tags
Show Similar Problems

"""
def twoSumSorted(numbers, target):
    """
    :type numbers: List[int]
    :type target: int
    :rtype: List[int]
    """
    if len(numbers) < 2:
        return False
    start, end = 0, len(numbers) - 1
    while start < end:
        sum = numbers[start] + numbers[end]
        if sum == target:
            return [start + 1, end + 1]
        elif sum < target:
            start += 1
        else:
            end -= 1
    return False

"""
Search in Rotated Sorted Array
https://leetcode.com/problems/search-in-rotated-sorted-array/

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.
(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
You are given a target value to search. If found in the array return its index, otherwise return -1.

You may assume no duplicate exists in the array.

O(logn):
+ A rotated array can be classified as two sub-array that is sorted
+ Look at the middle element (7). Compare it with the left most (4) and right most element.
+ The left most element (4) is less than (7)
+ --->All elements in the bottom half must be in strictly increasing order
+ Otherwise
"""
def searchRotatedSortedArr(arr, val):
    start, end = 0, len(arr) - 1
    while start <= end:
        mid = int((end + start) / 2)
        if arr[mid] == val:
            return mid

        # the left half is sorted
        if arr[start] <= arr[mid]:
            if arr[start] <= val < arr[mid]:
                end = mid - 1
            else:
                start = mid + 1
        # the right half is sorted
        elif arr[mid] < val <= arr[end]:
            start = mid + 1
        else:
            end = mid - 1
    return -1

# print (search_rotated_sorted_arr([5, 6, 7, 1, 2, 3, 4], 4))


"""
Union and Intersection of two sorted arrays
http://www.geeksforgeeks.org/union-and-intersection-of-two-sorted-arrays-2/
Given two sorted arrays, find their union and intersection.

For example, if the input arrays are:
arr1[] = {1, 3, 4, 5, 7}
arr2[] = {2, 3, 5, 6}
Then your program should print Union as {1, 2, 3, 4, 5, 6, 7} and Intersection as {3, 5}.

Assuming the numbers are distict.

O(m+n): when the size of the two set are similar

O(min(mLogn, nLogm)): when the ratio of difference in sizes are large.
This solution works better than the above approach when ratio the more than logarithmic order).
"""
# O(m+n)
def union(arr1, arr2):
    res = []
    # Use two index variables i and j, initial values i = 0, j = 0
    i1, i2 = 0, 0
    while i1 < len(arr1) and i2 < len(arr2):
        if arr1[i1] < arr2[i2]:
            res.append(arr1[i1])
            i1 += 1
        elif arr1[i1] > arr2[i2]:
            res.append(arr2[i2])
            i2 += 1
        else:
            i1 += 1

    res += arr1[i1:]
    res += arr2[i2:]
    return res

# O(m+n)
def intersection(arr1, arr2):
    res = []
    i, j = 0, 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] == arr2[j]:
            res.append(arr1[i])
            i += 1
            j += 1
        elif arr1[i] < arr2[j]:
            i += 1
        else:
            j += 1
    return res

import bisect
# O(mlogn), m = len(arr1), n = len(arr2)
def intersectionBS(arr1, arr2):
    res = []
    # assume both arr1 and arr2 are sorted
    for i1 in arr1:
        'Locate the leftmost value exactly equal to i1'
        i2 = bisect.bisect_left(arr2, i1)
        if i2 != len(arr2) and arr2[i2] == i1:
            res.append(i1)
    return res

"""
Overlapping Intervals
http://www.practice.geeksforgeeks.org/problem-page.php?pid=126

Given a collection of Intervals, merge all the overlapping Intervals.
For example:

Given [1,3], [2,6], [8,10], [15,18],

return [1,6], [8,10], [15,18].

Make sure the returned intervals are sorted.

Optimal:
+ sort the intervals according to starting time
+ combine all intervals in a linear traversal
"""
def mergeIntervals(meetings):
    if not meetings:
        return -1
    res = []
    meetings.sort(key=lambda x: x[0])  # Sort the intervals based on increasing order of starting time O(nlogn)
    mergedInter = meetings[0]
    for inter in meetings[1:]:  # O(n)
        if inter[0] <= mergedInter[1] < inter[1] : # merge with current inter
            mergedInter[1] = inter[1]
        else:   # add mergedInter to res
            res.append(mergedInter)
            mergedInter = inter
    res.append(mergedInter)
    return res

# Given a list meetings. Each meeting has a start time and end time.
# Return True if no overlap; otherwise, False
def isValidSchedule(meetings):
    if not meetings:
        return -1
    meetings.sort(key=lambda x: x[0])  # Sort the intervals based on increasing order of starting time O(nlogn)
    currInter = meetings[0]
    for inter in meetings[1:]:
        if currInter[1] > inter[0]:
            return False
        else:
            currInter = inter

    return True


"""
Kth Largest Element in an Array
http://www.geeksforgeeks.org/kth-smallestlargest-element-unsorted-array/
http://www.geeksforgeeks.org/kth-smallestlargest-element-unsorted-array-set-2-expected-linear-time/
http://www.geeksforgeeks.org/kth-smallestlargest-element-unsorted-array-set-3-worst-case-linear-time/

Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

For example,
Given [3,2,1,5,6,4] and k = 2, return 5.

Note:
You may assume k is always valid, 1 <= k <= array's length.

# The worst case time complexity of QuickSelect is O(n^2), but it works in O(n) on average.
# Expected time complexity of "randomized" QuickSelect is O(n).
# Get worst case linear time by selecting a pivot that divides array in a "balanced" way (there are not very few elements on one side and many on other side).
"""

"""
[3,6,1,2,9,4,5]
pivot = 0 (6)
"""

import random
def randomPartitionDesc(nums, l, r):
    pivot = random.randint(l, r)    # random pivot
    nums[pivot], nums[r] = nums[r], nums[pivot] # swap pivot with the last
    i = l
    for j in range(0, r - l):
        if nums[j] >= nums[r]:  # swap j with i if j is greater than last
            nums[j], nums[i] = nums[i], nums[j]
            i += 1
    nums[i], nums[r] = nums[r], nums[i]
    return i


def findKthLargestQuickSelect2(nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    # return heapq.nlargest(k, nums)[k-1]

    if nums:
        pos = randomPartitionDesc(nums, 0, len(nums)-1)
        if k > pos + 1:  # on the right
            return findKthLargestQuickSelect2(nums[pos + 1:], k - pos - 1)
        elif k < pos + 1:  # on the left
            return findKthLargestQuickSelect2(nums[:pos], k)
        else:
            return nums[pos]

# print (findKthLargestQuickSelect2([3, 2, 1, 5, 6, 6, 4], 1))


# Similar to QuickSort, it considers the last element as pivot
# and moves all smaller element to left of it and greater elements to right
def partition(nums, l, r):
    pos = l
    while l < r:
        if nums[l] < nums[r]:
            nums[l], nums[pos] = nums[pos], nums[l]
            pos += 1
        l += 1
        # print (l, r, (nums))
    nums[pos], nums[r] = nums[r], nums[pos]
    # print(l, r, (nums))
    return pos

# arr = [3,6,1,2,9,4,5]
# print (arr, partition(arr, 0, 6))

# Picks a random pivot element between l and r and
# partitions arr[l..r] arount the randomly picked element using partition()
import random
def randomPartition(nums, l, r):
    n = r-l+1
    pivot = random.randint(0,n-1)
    nums[l+pivot], nums[r] = nums[r], nums[l+pivot]
    return partition(nums, l, r)


"""
The idea is, not to do complete quicksort,
but stop at the point where pivot itself is randomly chosen (or at the smallest element)
"""
def findKthSmallestQuickSelect(nums, k):
    if nums:
        # Partition the array around last element and get
        # pos of pivot element in sorted array
        pos = randomPartition(nums, 0, len(nums) - 1)
        if k > pos + 1: # If pos is less, recur for right subarray
            return findKthSmallestQuickSelect(nums[pos + 1:], k - pos - 1)
        elif k < pos + 1: # Else, recur for left subarray
            return findKthSmallestQuickSelect(nums[:pos], k)
        else:   # If position is same as k
            return nums[pos] # return that element

# O(n) time, quick selection
def findKthLargestQuickSelect(nums, k):
    # convert the kth largest to smallest
    return findKthSmallestQuickSelect(nums, len(nums) + 1 - k)

# print (findKthLargest([3, 6, 1, 2, 9, 4, 5], 5))

# O(nlgn) time
def findKthLargestSortFirst(nums, k):
    return sorted(nums)[-k]


# O(nk) time, bubble sort idea, TLE
def findKthLargestBubbleSort(nums, k):
    for i in range(k):
        for j in range(len(nums) - i - 1):
            if nums[j] > nums[j + 1]:
                # exchange elements
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
    return nums[len(nums) - k]

# O(nk) time, selection sort idea
def findKthLargestSelectionSort(self, nums, k):
    for i in range(len(nums), len(nums)-k, -1):
        tmp = 0
        for j in range(i):
            if nums[j] > nums[tmp]:
                tmp = j
        nums[tmp], nums[i-1] = nums[i-1], nums[tmp] # swap the maximum value with the last
    return nums[len(nums)-k]

# O(nlogn) time?, min-heap
# Not really, the complexity of heap construction is only O(n)
# -> O(k + (n-k)logk)
import heapq
def findKthLargestMinHeap(nums, k):
    h = nums[:k]
    heapq.heapify(h)
    if len(nums) > k:
        for num in nums[k:]:
            heapq.heappushpop(h, num)
    return h[0] # h is sorted in inc

def findKthLargestMinHeap2(nums, k):
    return heapq.nlargest(k, nums)[k-1]


"""
Find K closest points to the origin
https://www.careercup.com/question?id=15974664

O(klogk): Build a max heap of first k elements. Now for every element left,
check if it is smaller than the root of max heap.
If it is, then replace the root and call heapify. Do it till the end

O(N): A linear time solution using quickselect to find the k-th euclidean distance,
and then iterating through the list, first to find all the points that are closer
than this k-th point, and then second to find all the points that have the same
euclidean distance as the k-th point, until we reach k elements, at which point
we stop and return those points.
"""
import math

def kClosestPoints(arr, k):
    def euclidean_dist(point):
        return math.pow(math.pow(point[0], 2) + math.pow(point[1], 2), 0.5)

    dist_arr = [euclidean_dist(item) for item in arr]
    kth_closest_dist = findKthSmallestQuickSelect(dist_arr, k) # quickselect to find the k-th euclidean distance,
    count = 0
    res = []
    # first to find all the points that are closer than this k-th point
    for i in range(len(arr)):
        if euclidean_dist(arr[i]) < kth_closest_dist:
            res.append(arr[i])
            count += 1

    # find all the points that have the same euclidean distance as the k-th point, until we reach k elements
    i = 0
    while count < k:
        if euclidean_dist(arr[i]) == kth_closest_dist:
            res.append(arr[i])
            count += 1
        i += 1
    return res


"""
Median of Two Sorted Arrays

There are two sorted arrays nums1 and nums2 of size m and n respectively.

Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).

Example 1:
nums1 = [1, 3]
nums2 = [2]

The median is 2.0
Example 2:
nums1 = [1, 2]
nums2 = [3, 4]

The median is (2 + 3)/2 = 2.5

https://leetcode.com/problems/median-of-two-sorted-arrays/
"""





"""
Find the Duplicate Number
https://leetcode.com/problems/find-the-duplicate-number/

Given an array nums containing n + 1 integers where each integer is between 1 and n (inclusive), prove that at least one duplicate number must exist.
Assume that there is only one duplicate number, find the duplicate one.

Note:
You must not modify the array (assume the array is read only).
You must use only constant, O(1) extra space.
Your runtime complexity should be less than O(n2).
There is only one duplicate number in the array, but it could be repeated more than once.

Solution:
The main idea is the same with problem Linked List Cycle II,https://leetcode.com/problems/linked-list-cycle-ii/.
Use two pointers the fast and the slow. The fast one goes forward two steps each time, while the slow one goes only step each time.
They must meet the same item when slow==fast. In fact, they meet in a circle, the duplicate number must be the entry point of the circle
when visiting the array from nums[0]. Next we just need to find the entry point. We use a point(we can use the fast one before)
to visit form begining with one step each time, do the same job to slow.
When fast==slow, they meet at the entry point of the circle. The easy understood code is as follows.
"""

def findDuplicate(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if len(nums) > 1:
        slow, fast = nums[0], nums[nums[0]]
        while slow != fast:
            slow, fast = nums[slow], nums[nums[fast]]

        fast = 0
        while fast != slow:
            fast = nums[fast]
            slow = nums[slow]
        return slow

    return -1


"""
Search for a Range
https://leetcode.com/problems/search-for-a-range/

Given an array of integers sorted in ascending order, find the starting and ending position of a given target value.

Your algorithm's runtime complexity must be in the order of O(log n).

If the target is not found in the array, return [-1, -1].

For example,
Given [5, 7, 7, 8, 8, 10] and target value 8,
return [3, 4].
"""

def searchRange(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    left = bisect.bisect_left(nums, target)
    if len(nums) <= left or nums[left] != target:
        return [-1, -1]
    right = bisect.bisect(nums, target)
    return [left, right - 1]