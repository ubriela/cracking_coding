"""
Sort Characters By Frequency
https://leetcode.com/problems/sort-characters-by-frequency/
Given a string, sort it in decreasing order based on the frequency of characters.

Example 1:

Input:
"tree"

Output:
"eert"

Explanation:
'e' appears twice while 'r' and 't' both appear once.
So 'e' must appear before both 'r' and 't'. Therefore "eetr" is also a valid answer.
Example 2:

Input:
"cccaaa"

Output:
"cccaaa"

Explanation:
Both 'c' and 'a' appear three times, so "aaaccc" is also a valid answer.
Note that "cacaca" is incorrect, as the same characters must be together.
Example 3:

Input:
"Aabb"

Output:
"bbAa"

Explanation:
"bbaA" is also a valid answer, but "Aabb" is incorrect.
Note that 'A' and 'a' are treated as two different characters.
"""
def frequencySort(s):
    """
    :type s: str
    :rtype: str
    """

    """
    "bbAa"

    use an hash array to compute frequency of each character
    chash = [0]*123
    for each character (key)
        inc value in chash by 1
    sort chash by value
    for each non-zero value in chash
        append curr char to res = []
    """
    chash = [(0, 0)] * 123
    for c in s:
        chash[ord(c)] = (c, chash[ord(c)][1] + 1)
    chash.sort(reverse=True, key=lambda x: x[1])
    i = 0
    res = []
    while chash[i][1] > 0:
        res += [chash[i][0]] * chash[i][1]
        i += 1
    return "".join(res)

"""
Given an array of numbers arr and a number S, find 4 different numbers in arr that sum up to S.

Write a function that gets arr and S and returns an array with 4 indices of such numbers in arr, or null if no such combination exists.
Explain and code the most efficient solution possible, and analyze its runtime and space complexity.

arr = [1, 3, 5, 7, 9]
arr = [1, 5, 7, 9, 15] < -
S = 22
# answer = [0,2,3,4]
(1 + 5) + (7 + 9) < -16
16(7 + 9)
16(1 + 15)
6
looking
16

S - 1
S - 3

[]

3

1, 3, 5, 7, 9
9
# 0,1,2
1: findSum([3, 5, 7, 9], 9 - 1)
3: findSum([1, 5, 7, 9], 9 - 3)
1: findSum([3, 5, 7, 9], 9 - 1)
3: findSum([1, 5, 7, 9], 9 - 3)
3: findSum([1, 5, 7, 9], 9 - 3)

Complexity: O(N*N) space and time
"""
def findSum4(arr, S):
    d = dict()
    for i, v1 in enumerate(arr):
        for j, v2 in enumerate(arr):
            if i != j:
                if (v1 + v2) in d:
                    d[v1 + v2].append((i, j))
                else:
                    d[v1 + v2] = [(i, j)]

    # 16 -> [(7,9), (1,15)]
    # 6 -> [(1,5)]
    # 22 -> (7,9), (1,5)
    for key, val in d.items():
        if S - key in d:
            val2 = d[S-key]
            for t1 in val:
                for t2 in val2:
                    if (t1[0] == t2[0] or t1[0] == t2[1]) or (t1[1] == t2[0] or t1[1] == t2[1]):
                        continue
                    # if len(set([t1[0],t1[1],t2[0],t2[1]])) == 4:
                    return t1[0],t1[1],t2[0],t2[1]

print (findSum4([1, 3, 5, 7, 9], 22))


"""
Two Sum
Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution.
"""
def twoSum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    if not nums or len(nums) <= 1:
        return []

    d = {}
    for i in range(len(nums)):
        if nums[i] in d:
            return [d[nums[i]], i]
        else:
            d[target - nums[i]] = i
    return []
