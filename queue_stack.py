"""
Third Maximum Number
https://leetcode.com/problems/third-maximum-number/
Given a non-empty array of integers, return the third maximum number in this array. If it does not exist, return the maximum number. The time complexity must be in O(n).

Example 1:
Input: [3, 2, 1]

Output: 1

Explanation: The third maximum is 1.
Example 2:
Input: [1, 2]

Output: 2

Explanation: The third maximum does not exist, so the maximum (2) is returned instead.
Example 3:
Input: [2, 2, 3, 1]

Output: 1

Explanation: Note that the third maximum here means the third maximum distinct number.
Both numbers with value 2 are both considered as second maximum.
"""
def thirdMax(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    l = len(nums)

    if l < 3:
        return max(nums)
    # heapq.heapify(heap)    # Transform a list into a heap, in-place, in linear time.

    heap = []  # min heap
    for v in nums:
        if v not in heap:
            if len(heap) < 3:
                heapq.heappush(heap, v)
            else:
                heapq.heappushpop(heap, v)  # Push item on the heap, then pop and return the smallest item from the heap

    return heapq.heappop(heap) if len(heap) == 3 else max(heap)

    # nums = set(nums)
    # if len(nums) < 3:
    #     return max(nums)
    # nums.remove(max(nums))
    # nums.remove(max(nums))
    # return max(nums)

"""
K-Messed Array Sort

Given an array arr of length n where each element is at most k places away from its sorted position,
Plan and code an efficient algorithm to sort arr.
Analyze the runtime and space complexity of your solution.

Example: n=10, k=2. The element belonging to index 6 in the sorted array,
may be at indices 4, 5, 6, 7 or 8 on the given array.

Naive solution: modified insertion sort
"""
import heapq
def sort_arr(arr, k):
    heap = [] # O(k) storage

    for e in arr[0: k+1]:
        heapq.heappush(heap, e)

    i = 0
    for e in arr[k +1:]: # N-k
        min_val = heapq.heappop(heap)
        arr[i] = min_val
        heapq.heappush(heap, e)
        i += 1

    while heap: # k values
        min_val = heapq.heappop(heap)
        arr[i] = min_val
        i += 1


# arr = [3,2,1,5,4,6,7,8,9]
# sort_arr(arr, 3)
# print arr


"""
Valid Parentheses
https://leetcode.com/problems/valid-parentheses/

Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

The brackets must close in the correct order, "()" and "()[]{}" are all valid but "(]" and "([)]" are not.
"""
def isValidParentheses(s):
    """
    :type s: str
    :rtype: bool
    """
    """
    use a stack to put every char
        every time find a char on top of stack -> pop stack
        else push it into stack
    at the end if the stack is not empty --> False
    otherwise, True
    """

    """
    test
    ()
    stack = (
    stack =

    ()[]{}
    stack = (
    stack =
    stack = [
    stack =
    stack = {
    stack =

    (]
    stack = (
    stack = (]

    ([)]
    stack = (
    stack = ([
    stack = ([)
    stack = ([)]
    """
    if not s or len(s) <= 1:
        return False
    stack = []
    p = {")": "(", "]": "[", "}": "{"}  # mapping close parentheses -> open parentheses
    for c in s:
        if c in p:  # close parenthesis
            if stack and p[c] == stack[-1]:
                stack.pop()
            else:  # found an invalid pair
                return False
        else:
            stack.append(c)
    # valid parentheses should have empty stack
    return len(stack) == 0

"""
Write a function to find the longest common prefix string amongst an array of strings.

Run Time Complexity: if n is the length of shortest string in array of strings and there are m strings in the array. Run time complexity would be O(m*n)
Space Complexity: Constant space
"""


"""
[1,2,3,{"4":5,"6":7}]
[1,2,3,{"4":5,..}]

queue
while queue:
    pop first

seriable
    string number boolean list
"""

"""
{1:1,2:2}

{1:1,2:2}
res = ""
"""

from collections import deque

def serialize_dict(data):
    queue = deque([(data, data)])
    serialized_str = "{"
    while queue:  # while q is not empty
        item = queue.popleft()  # removed
        key, value = item[0], item[1]
        t = type(item[1])
        if t in [int, str, bool, list]:
            if len(serialized_str) > 1:
                serialized_str += ", "  # with a space for pretty printing
            serialized_str += repr(key) + ":" + str(value)
        elif t is dict:
            for key, val in value.items():
                queue.append((key, val))  # store tuple in queue

    serialized_str += "}"
    return serialized_str


# d = dict({"1": 1, "2": 2})
# print serialize_dict({"1": 1, "2": 2})

"""
Test cases
List of tuples (Input, Expected output)
"""
test_cases = [
    # corner cases
    (dict([]), "{}"),

    # simple cases
    (dict({'1': 1, '2': 2}), "{'1':1, '2':2}"),
    (dict({1: 1, 2: 2}), "{1:1, 2:2}"),
    (dict({'1': False, '2': True}), "{'1':False, '2':True}"),
    (dict({'1': [1, 2], '2': [3, 4]}), "{'1':[1, 2], '2':[3, 4]}"),

    # complex cases
    (dict({'1': 1, '2': {'3': 3}}), "{'1':1, '2':{'3':3}}")
]

"""
This function test all pairs of (Input, Expected output)
Return True if the designed function passes all test cases; otherwise, return False
"""


def test_serialize_dict(test_cases):
    """
    list of ([array of values], results)
    """
    results = [False] * len(test_cases)  # initialize all tests to False
    test = 0

    for case in test_cases:
        if serialize_dict(case[0]) == case[1]:
            results[test] = True
        test += 1

    test = 0  # current test
    for res in results:
        print ("Test case " + str(test + 1) + ": " + str(results[test]))
        test += 1
    fail_tests = results.count(False)  # number of fail tests
    # summary
    print ("The number of test passed: " + str(len(test_cases) - fail_tests))
    print ("The number of test failed: " + str(fail_tests))

    if fail_tests >= 1:
        return False

    return True


# if test_serialize_dict(test_cases):
#     print "Passed all test cases!!!"

"""
Results
Test case 1: True
Test case 2: True
Test case 3: True
Test case 4: True
Test case 5: True
Test case 6: False
The number of test passed: 5
The number of test failed: 1
"""

"""
In the last 10 minutes, I am trying to fix the issue with a complex case (nested json). I am thinking of using a stack
instead of queue to handle the closing brackets {'}'. However if using the stack, the order of output would be different.
"""


# This modified function handle '}' using stack
def serialize_dict_stack(data):
    stack = [(data, data)]
    serialized_str = "{"
    prev_item_is_dict = False  # handle corner case
    while stack:  # while q is not empty
        item = stack.pop()  # removed
        key, value = item[0], item[1]
        t = type(item[1])
        if item[1] == '}':
            serialized_str += item[1]
        elif t in [int, str, bool, list]:
            if len(serialized_str) > 1 and not prev_item_is_dict:
                serialized_str += ","
            serialized_str += repr(key) + ":" + str(value)
            prev_item_is_dict = False
        elif t is dict:
            prev_item_is_dict = True
            if value != data:
                serialized_str += repr(key) + ":{"
                stack.append(('{', '}'))
            for key, val in value.items():
                stack.append((key, val))  # store tuple in queue

    serialized_str += "}"
    return serialized_str

# print serialize_dict_stack(dict({'1': 1, '2': {'3': 3}}))
"""
Output: {'2':{'3':3},'1':1}.
"""



"""
Number of Islands
https://leetcode.com/problems/number-of-islands/

Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

Example 1:

11110
11010
11000
00000
Answer: 1

Example 2:

11000
11000
00100
00011
Answer: 3

2d matrix M

there can be multiple islands

6 islands

meet first 1
   use DFS to explore all 1-cells
   inc islands
   reset all 1-cells in curr island to zeros (play the role of visted set)

keep searching the next 1

DFS for all 1-cells: #1-cells
+ m*n
= O(m*n)

"""

def all_neighbors(M, u, v):
    # later (only look for 1-cells)
    # check u+-1 and v+-1 are valid
    return [t for t in [(u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)] if
            0 <= t[0] < len(M) and 0 <= t[1] < len(M[0]) and M[t[0]][t[1]] == 1]


def dfsFillIterative(M, i, j):
    stack = [(i, j)]
    while stack:
        u, v = stack.pop()
        M[u][v] = 0  # visited
        neighbors = all_neighbors(M, u, v)
        stack.extend(neighbors)


def dfsFill(M, i, j):
   if i>=0 and j>=0 and i<len(M) and j<len(M[0]) and M[i][j] == 1:
        M[i][j]=0;
        dfsFill(M, i + 1, j);
        dfsFill(M, i - 1, j);
        dfsFill(M, i, j + 1);
        dfsFill(M, i, j - 1);

def numIslands(grid):
    """
    :type grid: List[List[str]]
    :rtype: int
    """
    # convert grid to M
    M = []
    for row in grid:
        M.append(map(int, list(row)))

    islands = 0
    for i in range(len(M)):
        for j in range(len(M[0])):
            if M[i][j] == 1:
                dfsFillIterative(M, i, j)
                # dfsFill(M, i, j)  # update M within this func
                islands += 1
    return islands