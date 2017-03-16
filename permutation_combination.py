"""
Letter Combinations of a Phone Number
https://leetcode.com/problems/letter-combinations-of-a-phone-number/
Given a digit string, return all possible letter combinations that the number could represent.

A mapping of digit to letters (just like on the telephone buttons) is given below.

Input:Digit string "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
Note:
Although the above answer is in lexicographical order, your answer could be in any order you want.
"""
def letterCombinations(digits):
    """
    :type digits: str
    :rtype: List[str]
    """
    if '' == digits: return []
    kvmaps = {
        '2': 'abc',
        '3': 'def',
        '4': 'ghi',
        '5': 'jkl',
        '6': 'mno',
        '7': 'pqrs',
        '8': 'tuv',
        '9': 'wxyz'
    }
    ret = ['']
    for c in digits:
        tmp = []
        for y in ret:
            for x in kvmaps[c]:
                tmp.append(y + x)
        ret = tmp

    return ret

"""
Combinations
https://leetcode.com/problems/combinations/

Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.

For example,

If n = 4 and k = 2, a solution is:

[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
"""
from itertools import combinations
def combine(self, n, k):
    """
    :type n: int
    :type k: int
    :rtype: List[List[int]]
    """
    # return list(combinations(range(1, n+1), k))
    if k == 1:
        return [[i] for i in range(1,n+1)]
    elif n == k:
        return [range(1,n+1)]
    elif n>k:
        return self.combine(n-1,k) + map(lambda x : x + [n], self.combine(n-1,k-1))
