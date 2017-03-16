"""
Longest Substring Without Repeating Characters
https://leetcode.com/problems/longest-substring-without-repeating-characters/?tab=Description
Given a string, find the length of the longest substring without repeating characters.

Examples:

Given "abcabcbb", the answer is "abc", which the length is 3.

Given "bbbbb", the answer is "b", with the length of 1.

Given "pwwkew", the answer is "wke", with the length of 3. Note that the answer must be a substring, "pwke" is a subsequence and not a substring.

"""
def lengthOfLongestSubstring(s):
    """
    :type s: str
    :rtype: int
    """

    """
    abcabcbb
    used = {}
    start=0
    i=0,c=a
        ml=1
        used={a:0}
    i=1,c=b
        ml=2
        used={a:0,b:1}
    i=2,c=c
        ml=3
        used={a:0,b:1,c:2}
    i=3,c=a
        start=1
        used={a:3,b:1,c:2}
    i=4,c=b
        start=2
        used={a:3,b:4,c:2}
    i=5,c=c
        start=3
        used={a:3,b:4,c:5}
    i=6,c=b
        start=5
        used={a:3,b:6,c:5}
    i=7,c=b
        start=7
        used={a:3,b:7,c:5}
    """
    used = {}
    max_length = start = 0
    for i, c in enumerate(s):
        if c in used and start <= used[c]:
            start = used[c] + 1
        else:
            max_length = max(max_length, i - start + 1)

        used[c] = i

    return max_length
"""
Longest Increasing Subsequence
https://leetcode.com/problems/longest-increasing-subsequence/

Given an unsorted array of integers, find the length of longest increasing subsequence.

For example,
Given [10, 9, 2, 5, 3, 7, 101, 18],
The longest increasing subsequence is [2, 3, 7, 101], therefore the length is 4.
Note that there may be more than one LIS combination, it is only necessary for you to return the length.

Your algorithm should run in O(n2) complexity.

Follow up: Could you improve it to O(n log n) time complexity?
"""

def lengthOfLIS(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """

    """
    Input: [10, 9, 2, 5, 3, 7, 101, 18]
    Return 4 # [2, 3, 7, 101]
    10, 9, 2, 5, 3, 7, 101, 18
    1   1  1  2  2  3   4   4
    a[i]: maximum subarray ending up at i
    result[i]==max(result[j]) + 1 where 0<=j<i and arr[i] > arr[j]; otherwise result[i] = 1
    O(N^2)

    a[i] is an array storing the smallest tail of all increasing subsequences with length i+1
        curent value at 3 --> use binary search to find smallest

    [10]
    [9]
    [2]
    [2, 5]
    [2, 3]
    [2, 3, 7]
    [2, 3, 7, 101]
    [2, 3, 7, 18]
    """
    a = []
    size = 0
    for x in nums:
        i, j = 0, size
    while i != j:
        m = (i + j) / 2
        if a[m] < x:
            i = m + 1
        else:
            j = m
    if i >= len(a):
        a.append(x)
    else:
        a[i] = x
    size = max(i + 1, size)

    return size



"""
Maximum Subarray
https://leetcode.com/problems/maximum-subarray/

Find the contiguous subarray within an array (containing at least one number) which has the largest sum.

For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
the contiguous subarray [4,-1,2,1] has the largest sum = 6.

O(n) time
O(1) space
"""
def maxSubArray(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """

    """
    input [-2,1,-3,4,-1,2,1,-5,4]
    return 6 # [4,-1,2,1]

    -2,1,-3,4,-1,2,1,-5,4

    a[i]: maximum subarray ending up at i
    a[i] = a[i-1] + nums[i]


    a[i] = max(nums[i], nums[i] + a[i-1])

    """
    if len(nums) < 1:
        return 0
    prev_a = nums[0]
    max_sum = nums[0]
    for i, v in enumerate(nums[1:]):
        a = max(v, v + prev_a)
        max_sum = max(a, max_sum)
        prev_a = a

    return max_sum


"""
Best Time to Buy and Sell Stock with Cooldown
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/

Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like
(ie, buy one and sell one share of the stock multiple times) with the following restrictions:

You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day)
Example:

prices = [1, 2, 3, 0, 2]
maxProfit = 3
transactions = [buy, sell, cooldown, buy, sell]
"""

def maxProfit(prices):
    """
    :type prices: List[int]
    :rtype: int
    """

    """
    This problem can be nicely represented using a diagram of different states.
    s0: the state of resting after selling (or resting)
    s1: state after buying (or resting)
    s2: state after selling
    """
    if not prices:
        return 0

    # O(1) space, O(n) time
    s0, s1, s2 = 0, -prices[0], 0
    for i in range(1, len(prices)):
        s0, s1, s2 = max(s0, s2), max(s0 - prices[i], s1), s1 + prices[i]
    return max(s0, s2)

    # O(n) space, O(n) time
    # base case
    # l = len(prices)
    # s0, s1, s2 = [0] * l, [0] * l, [0] * l
    # # s0[0] = 0
    # s1[0] = -prices[0]
    # s2[0] = 0

    # for i in range(1,l):
    #     s0[i] = max(s0[i-1], s2[i-1])
    #     s1[i] = max(s0[i-1] - prices[i], s1[i-1])
    #     s2[i] = s1[i-1] + prices[i]
    # return max(s0[l-1], s2[l-1])

"""
Best Time to Buy and Sell Stock II
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/

Say you have an array for which the ith element is the price of a given stock on day i.
Design an algorithm to find the maximum profit. You may complete as many transactions as
you like (ie, buy one and sell one share of the stock multiple times).
However, you may not engage in multiple transactions at the same time
(ie, you must sell the stock before you buy again).

Basically, if tomorrow's price is higher than today's, we buy it today and sell tomorrow. Otherwise, we don't.
"""
def maxProfit(prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    return sum(max(prices[i + 1] - prices[i], 0) for i in range(len(prices) - 1))

"""
Best Time to Buy and Sell Stock
https://leetcode.com/problems/best-time-to-buy-and-sell-stock/

Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.

Example 1:
Input: [7, 1, 5, 3, 6, 4]
Output: 5

max. difference = 6-1 = 5 (not 7-1 = 6, as selling price needs to be larger than buying price)
Example 2:
Input: [7, 6, 4, 3, 1]
Output: 0

In this case, no transaction is done, i.e. max profit = 0.
"""

def maxProfit(prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    if not prices:
        return 0
    max_gain = 0
    min_price = prices[0]
    for p in prices[1:]:
        gain = p - min_price
        max_gain = max(gain, max_gain)
        min_price = min(p, min_price)
    return max_gain





"""
Word Break
https://leetcode.com/problems/word-break/

Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, determine if s can be segmented into a space-separated sequence of one or more dictionary words. You may assume the dictionary does not contain duplicate words.

For example, given
s = "leetcode",
dict = ["leet", "code"].

Return true because "leetcode" can be segmented as "leet code".

worse case O(N^2), N is the length of the input string
"""
def wordBreak(s, wordDict):
    """
    :type s: str
    :type wordDict: List[str]
    :rtype: bool
    """
    # res[i] means it is possible to split res[0,i] into word
    res = [False] * (len(s) + 1)
    res[0] = True
    for i in range(1,len(s)+1):
        for j in range(i):
            if res[j] and s[j:i] in wordDict:
                res[i] = True
                break
    return res[-1]


"""
Minimum Path Sum
https://leetcode.com/problems/minimum-path-sum/

Given a m x n grid filled with non-negative numbers, find a path from top left to bottom
right which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.
"""
def minPathSum(self, grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    if len(grid) <= 1 or len(grid[0]) <= 1:
        total = sum([val for row in grid for val in row])
        return total
    for i in range(1, len(grid)):
        grid[i][0] += grid[i - 1][0]
    for j in range(1, len(grid[0])):
        grid[0][j] += grid[0][j - 1]

    for i in range(1, len(grid)):
        for j in range(1, len(grid[0])):
            grid[i][j] += min(grid[i][j - 1], grid[i - 1][j])

    return grid[len(grid) - 1][len(grid[0]) - 1]



"""
Continuous sequence against target number
Question: Given a sequence of positive integers A and an integer T,
return whether there is a continuous sequence of A that sums up to exactly T

Example

[23, 5, 4, 7, 2, 11], 20. Return True because 7 + 2 + 11 = 20
[1, 3, 5, 23, 2], 8. Return True because 3 + 5 = 8
[1, 3, 5, 23, 2], 7 Return False because no sequence in this array adds up to 7

My solution used sliding window.
The window expands to the right when current sum is less than T,
it shrinks from left when sum is greater than T and algorithm
return true in case current sum is T.
"""
def hasSequence(nums, T):
    if T <=0:
        return False
    if len(nums) == 0:
        return False
    i = 0
    start = 0
    sum = 0
    while i < len(nums):
        if sum + nums[i] < T:    # expand to the right
            sum += nums[i]
        elif sum + nums[i] == T:    # return true in case current sum is T
            return True
        else:  # sum is greater than T
            sum += nums[i]    # include current val
            while sum > T:    # shrink from left
                sum -= nums[start]
                start += 1
            if sum == T:
                return True
        i+=1
    return False

# print (hasSequence([2,4,6,10,11,15], 21))

"""
Solution 2: we can store cumulative sum at each position in a hash table and
check for the sum of T along the way. If we see Sum at current position i, and
saw Sum-T at some previous position j, then all numbers between j and i will sum up to T.
Remember to check the case Sum-T = Sum. To avoid that case, need to check the hash table before inserting current Sum into it.
Still O(N) time, but O(N) space needed.
"""
def checkSum(nums, T):
    sum = 0
    hashset = set()

    if nums and nums[0] == T:
        return True

    for i in range(len(nums)):
        sum += nums[i]

        # if see a value of  (sum-T) previously, report True
        if sum - T in hashset:
            return True
        else:    # insert current sum in hashset
            hashset.add(sum)

    return False

# print (checkSum([23, 5, 4, 7, 2, 11], 23))
# print (checkSum([23, 5, 4, 7, 2, 11], 20))
# print (checkSum([1, 3, 5, 23, 2], 8))

# print (checkSum([23, 5, 4, 7, 2, 11], 22))
# print (checkSum([1, 3, 5, 23, 2], 7))

"""
Find the longest substring with k unique characters in a given string
Given a string you need to print longest possible substring that has exactly M unique characters.
If there are more than one substring of longest possible length, then print any one of them.

Examples:

"aabbcc", k = 1
Max substring can be any one from {"aa" , "bb" , "cc"}.

"aabbcc", k = 2
Max substring can be any one from {"aabb" , "bbcc"}.

"aabbcc", k = 3
There are substrings with exactly 3 unique characters
{"aabbcc" , "abbcc" , "aabbc" , "abbc" }
Max is "aabbcc" with length 6.

"aaabbb", k = 3
There are only two unique characters, thus show error message.
http://www.geeksforgeeks.org/find-the-longest-substring-with-k-unique-characters-in-a-given-string/
"""
from collections import Counter

def longest_substr(s, k):
    def isValid(s, k):
        return True if len(Counter(s)) <= k else False
    if len(Counter(s)) < k:
        return False
    last = 0
    max_len, max_str = 0, s[0:k]
    for i in range(k,len(s)+1):
        while not isValid(s[last:i], k):
            last += 1
        if i - last > max_len:
            max_len, max_str = i - last, s[last:i]
    return max_str
# print (longest_substr('aabbcccbaaaaaaa', 2))

"""
Contains Duplicate II
Given an array of integers and an integer k, find out whether there are two distinct indices i and j in the array such that
nums[i] = nums[j] and the absolute difference between i and j is at most k.
"""
def containsNearbyDuplicate(nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: bool
    """
    h = {}
    for i, v in enumerate(nums):
        if v in h and i - h[v] <= k:
            return True
        h[v] = i
    return False



"""
House Robber
https://leetcode.com/problems/house-robber/

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.
"""
def rob(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums:
        return 0

    now, last = 0, 0
    for i in nums:
        now, last = max(last + i, now), now
    return now



"""
Jump Game
https://leetcode.com/problems/jump-game/
Given an array of non-negative integers, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.

For example:
A = [2,3,1,1,4], return true.

A = [3,2,1,0,4], return false.
"""

def canJump(nums):
    """
    :type nums: List[int]
    :rtype: bool
    """

    """
    [1,2]
    minYes  i   v
    1       0

    First idea:
    The idea is backtracking. If index i can be jumped, index i+ can be jumped
        --> keep track of minimum index that can jump

    Second idea:
    Forwarding. If index i can be jumped, index i- can be jumped
        --> keep track of maximum index that can jump
    """
    # if not nums:
    #     return False
    # l = len(nums)
    # if l == 1:
    #     return True
    # minYes = l - 1
    # for i in range(l-2,-1,-1):
    #     if i + nums[i] >= minYes:
    #         minYes = i

    # return minYes == 0

    maxJump = 0
    for i, step in enumerate(nums):
        if i > maxJump:
            return False
        maxJump = max(maxJump, i + step)
    return True

# print (canJump([3,2,1,0,4]))