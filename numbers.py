"""
String to Integer (atoi)
https://leetcode.com/problems/string-to-integer-atoi/?tab=Description

Implement atoi to convert a string to an integer.

Hint: Carefully consider all possible input cases. If you want a challenge, please do not see below and ask yourself what are the possible input cases.

Notes: It is intended for this problem to be specified vaguely (ie, no given input specs). You are responsible to gather all the input requirements up front.
"""
def myAtoi(s):
    """
    :type str: str
    :rtype: int
    """
    if len(s) == 0: return 0
    ls = list(s.strip())

    sign = -1 if ls[0] == '-' else 1

    if ls[0] in ['-', '+']: del ls[0]
    ret, i = 0, 0
    while i < len(ls) and ls[i].isdigit():
        ret = ret * 10 + ord(ls[i]) - ord('0')
        i += 1
    return max(-2 ** 31, min(sign * ret, 2 ** 31 - 1))

"""
Count Primes
https://leetcode.com/problems/count-primes/

Description:

Count the number of prime numbers less than a non-negative number, n.
"""
def isPrime(self, n):
    if n <= 1:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def countPrimes(n):
    """
    :type n: int
    :rtype: int
    """

    # O(n^1.5)
    # count = 0
    # i = 1
    # while i < n:
    #     if self.isPrime(i):
    #         count += 1
    #     i+=1
    # return count

    # The Sieve of Eratosthenes uses an extra O(n) memory and its runtime complexity is O(n log log n)
    if n < 3:
        return 0
    primes = [True] * n
    primes[0] = primes[1] = False

    #   Loop's ending condition is i < sqrt(n)
    #   to avoid repeatedly calling an expensive function sqrt().
    for i in range(2, int(n ** 0.5) + 1):
        primes[i * i: n: i] = [False] * len(primes[i * i: n: i])
    return primes.count(True)


"""
Pascal's Triangle II
Given an index k, return the kth row of the Pascal's triangle.

For example, given k = 3,

Return [1,3,3,1].
"""
def getRow(self, rowIndex):

    """
    :type rowIndex: int
    :rtype: List[int]
    """
    r = [1]
    for i in xrange(1,rowIndex+1):
      r.append( r[-1]*(rowIndex-i+1)/i )
    return r

"""
Pascal's Triangle

https://leetcode.com/problems/pascals-triangle/

Given numRows, generate the first numRows of Pascal's triangle.

For example, given numRows = 5,

Return

[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]
"""
def generate(self, numRows):

    """
    :type numRows: int
    :rtype: List[List[int]]
    """
    res = [[1]]
    for i in range(1, numRows):
        res += [map(lambda x, y: x+y, res[-1] + [0], [0] + res[-1])]
    return res[:numRows]



"""
Pow(x, n)
https://leetcode.com/problems/powx-n/
Implement pow(x, n).
Valid Perfect Square
Given a positive integer num, write a function which returns True if num is a perfect square else False.

Note: Do not use any built-in library function such as sqrt.

Example 1:

Input: 16
Returns: True

Example 2:

Input: 14
Returns: False
"""

def isPerfectSquare(self, num):
    """
    :type num: int
    :rtype: bool
    """

    if num < 1: return False
    if num == 1: return True
    t = num/2
    while t*t > num:
        t = (t + num/t)/2
    return t*t == num

"""
Integer Replacement
https://leetcode.com/problems/integer-replacement/

Given a positive integer n and you can do operations as follow:

1. If n is even, replace n with n/2.
2. If n is odd, you can replace n with either n + 1 or n - 1.
What is the minimum number of replacements needed for n to become 1?

Example 1:

Input:
8

Output:
3

Explanation:
8 -> 4 -> 2 -> 1

Example 2:

Input:
7

Output:
4

Explanation:
7 -> 8 -> 4 -> 2 -> 1
or
7 -> 6 -> 3 -> 2 -> 1
"""
def integerReplacement(self, n):
    """
    :type n: int
    :rtype: int
    """
    count = 0
    while n > 1: # n=3
        if n%2 == 0:
            n = n/2
        elif (n+1)%4==0 and n!=3:
            n+=1
        else:
            n-=1
        count += 1
    return count

"""
Happy Number
Write an algorithm to determine if a number is "happy".
A happy number is a number defined by the following process: Starting with any positive integer,
replace the number by the sum of the squares of its digits, and repeat the process until the number equals 1 (where it will stay),
or it loops endlessly in a cycle which does not include 1. Those numbers for which this process ends in 1 are happy numbers.

Example: 19 is a happy number

* 12 + 92 = 82
* 82 + 22 = 68
* 62 + 82 = 100
* 12 + 02 + 02 = 1
"""
def isHappy(self, n):
    """
    :type n: int
    :rtype: bool
    """
    d = set([n])
    while True:
        n = sum(map(lambda x : int(x)**2, str(n)))
        if n == 1:
            return True
        elif n in d:
            return False
        d.add(n)



"""
Rotate Array
Rotate an array of n elements to the right by k steps.

For example, with n = 7 and k = 3, the array [1,2,3,4,5,6,7] is rotated to [5,6,7,1,2,3,4].
"""
def rotate(self, nums, k):

    """
    :type nums: List[int]
    :type k: int
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    dq = deque(nums)
    dq.rotate(k)
    nums = list(dq)



"""
Sqrt(x)
https://leetcode.com/problems/sqrtx/

Implement int sqrt(int x).

Compute and return the square root of x.
"""
def mySqrt(self, x):

    """
    :type x: int
    :rtype: int
    """
    if x == 0:
        return 0
    left, right = 1, x
    while True:
        mid = left + (right - left)/2
        if mid > x/mid:
            right = mid - 1
        else:
            if x/mid == mid:
                return mid
            left = mid + 1

"""
Hamming Distance
The Hamming distance between two integers is the number of positions at which the corresponding bits are different.

Given two integers x and y, calculate the Hamming distance.

Note:

0 <= x, y < 231.

Example:

Input: x = 1, y = 4

Output: 2
"""
def hammingDistance(x, y):
    """
    :type x: int
    :type y: int
    :rtype: int
    """
    return bin(x^y).count('1')


"""
Find All Numbers Disappeared in an Array
Given an array of integers where 1 <= a[i] <= n (n = size of array), some elements appear twice and others appear once.

Find all the elements of [1, n] inclusive that do not appear in this array.

Could you do it without extra space and in O(n) runtime? You may assume the returned list does not count as extra space.

Example:

Input:
[4,3,2,7,8,2,3,1]

Output:
[5,6]
"""
def findDisappearedNumbers(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    return list(set(range(1, len(nums)+1)) - set(nums))




"""
Single Number
Given an array of integers, every element appears twice except for one. Find that single one.
"""
def singleNumber(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    res = 0
    for val in nums:
        res ^= val
    return res



"""
Add Digits
Given a non-negative integer num, repeatedly add all its digits until the result has only one digit.

For example:

Given num = 38, the process is like: 3 + 8 = 11, 1 + 1 = 2. Since 2 has only one digit, return it.

Follow up:
Could you do it without any loop/recursion in O(1) runtime?
"""
def addDigits(num):

    """
    :type num: int
    :rtype: int
    """
    # while(num >= 10):
    #     temp = 0
    #     while(num > 0):
    #         temp += num % 10
    #         num /= 10
    #     num = temp
    # return num
    if num == 0 : return 0
    return (num - 1) % 9 + 1




"""
Power of Four
Given an integer (signed 32 bits), write a function to check whether it is a power of 4.

Example:

Given num = 16, return true. Given num = 5, return false.

Follow up: Could you solve it without loops/recursion?
"""
def isPowerOfFour(self, num):
    """"
    :type num: int
    :rtype: bool
    """
    # return num > 0 and (num & (num - 1)) == 0 and (num - 1) % 3 == 0

    if num == 1 or num == 4 or num == 16:
        return True
    elif num < 16:
        return False
    elif num % 16 == 0:
        return self.isPowerOfFour(num/16)
    else:
        return False


"""
Power of Three
Given an integer, write a function to determine if it is a power of three.
"""
def isPowerOfThree(self, n):
    """
    :type n: int
    :rtype: bool
    """
    if n == 1 or n == 3:
        return True
    if n <= 0 or n == 2 or n %3 !=0:
        return False

    return self.isPowerOfThree(n/3)


"""
Power of Two
https://leetcode.com/problems/power-of-two/
Given an integer, write a function to determine if it is a power of two.
"""
def isPowerOfTwo(self, n):
    """
    :type n: int
    :rtype: bool
    """
    # if n == 1 or n == 2:
    #     return True
    # if n == 0 or n == 3 or n%4 != 0:
    #     return False

    # return self.isPowerOfTwo(n/4)
    return n>0 and (not n & (n-1))


"""
Assign Cookies
https://leetcode.com/problems/assign-cookies/

Assume you are an awesome parent and want to give your children some cookies.
But, you should give each child at most one cookie.
Each child i has a greed factor gi, which is the minimum size of a cookie that
the child will be content with; and each cookie j has a size sj. If sj >= gi,
we can assign the cookie j to the child i, and the child i will be content.
Your goal is to maximize the number of your content children and output the maximum number.

Note:

You may assume the greed factor is always positive.

You cannot assign more than one cookie to one child.

Example 1:


Input: [1,2,3], [1,1]

Output: 1

Explanation: You have 3 children and 2 cookies. The greed factors of 3 children are 1, 2, 3.
And even though you have 2 cookies, since their size is both 1, you could only make the child whose greed factor is 1 content.
You need to output 1.

Example 2:


Input: [1,2], [1,2,3]

Output: 2

Explanation: You have 2 children and 3 cookies. The greed factors of 2 children are 1, 2.
You have 3 cookies and their sizes are big enough to gratify all of the children,
You need to output 2.
"""
def findContentChildren(self, g, s):
    """
    :type g: List[int]
    :type s: List[int]
    :rtype: int
    """
    g.sort()
    s.sort()

    res = 0
    while s and g:
        size = s.pop()
        while g:
            greed_factor = g.pop()
            if greed_factor <= size:
                res += 1
                break
    return res

"""
Palindrome Number
https://leetcode.com/problems/palindrome-number/

Determine whether an integer is a palindrome. Do this without extra space.
"""
def isPalindrome(x):
    """
    :type x: int
    :rtype: bool
    """
    """
    reverse the number first and see if it is equal to the original number.
    12321
    rem = 12321, rev = 0
    rev = 1, rem = 1232
    rev = 12, rem = 123
    rev = 123, rem = 12
    rev = 1232, rem = 1
    12321, rem = 0
    """
    if x < 0:
        return False
    remainder, reversed_x = x, 0
    while remainder > 0:
        reversed_x = 10 * reversed_x + (remainder % 10)
        remainder = (remainder - remainder % 10) / 10

    return reversed_x == x

"""
Magical String
https://leetcode.com/problems/magical-string/

A magical string S consists of only '1' and '2' and obeys the following rules:

The string S is magical because concatenating the number of contiguous occurrences of characters '1' and '2' generates the string S itself.

The first few elements of string S is the following: S = "1221121221221121122..."

If we group the consecutive '1's and '2's in S, it will be:

1 22 11 2 1 22 1 22 11 2 11 22 ......

and the occurrences of '1's or '2's in each group are:

1 2	2 1 1 2 1 2 2 1 2 2 ......

You can see that the occurrence sequence above is the S itself.

Given an integer N as input, return the number of '1's in the first N number in the magical string S.

Note: N will not exceed 100,000.

Example 1:
Input: 6
Output: 3
Explanation: The first 6 elements of magical string S is "12211" and it contains three 1's, so return 3.
"""

from collections import deque

def magicalString(n):
    """
    :type n: int
    :rtype: int
    """

    """
    curr    prev    queue   count   count_1
    1       2       2       3       1
    2       1       1,1     5       3
    1       2       2     6       3
    2               1,1
    """
    if not n:
        return 0
    count_1 = 1  # 1
    count = 3  # 1,2,2
    queue = deque([2])
    prev = 2
    while queue and count < n:
        curr = queue.popleft()
        if prev == 1:
            queue.extend([2] * curr)
        else:
            queue.extend([1] * curr)
            count_1 += curr if count + curr <= n else 1  # extending [1,1]
        count += curr
        prev = (prev + 1) % 2  # 2<->1
    return count_1


"""
Nim Game
https://leetcode.com/problems/nim-game/

You are playing the following Nim Game with your friend: There is a heap of stones on the table, each time one of you take turns to remove 1 to 3 stones. The one who removes the last stone will be the winner. You will take the first turn to remove the stones.

Both of you are very clever and have optimal strategies for the game. Write a function to determine whether you can win the game given the number of stones in the heap.

For example, if there are 4 stones in the heap, then you will never win the game: no matter 1, 2, or 3 stones you remove, the last stone will always be removed by your friend.

Hint:

If there are 5 stones in the heap, could you figure out a way to remove the stones such that you will always be the winner?
"""
def canWinNim(self, n):
    """
    :type n: int
    :rtype: bool
    """
    return n % 4 != 0 # Theorem: all 4s shall be false




"""
Water and Jug Problem
You are given two jugs with capacities x and y litres. There is an infinite amount of water supply available. You need to determine whether it is possible to measure exactly z litres using these two jugs.

If z liters of water is measurable, you must have z liters of water contained within one or both buckets by the end.

Operations allowed:

Fill any of the jugs completely with water.
Empty any of the jugs.
Pour water from one jug into another till the other jug is completely full or the first jug itself is empty.
Example 1: (From the famous "Die Hard" example)

Input: x = 3, y = 5, z = 4
Output: True
Example 2:

Input: x = 2, y = 6, z = 5
Output: False
"""
def canMeasureWater(self, x, y, z):
    """
    :type x: int
    :type y: int
    :type z: int
    :rtype: bool
    """

    """
    if x and y are coprime,  then we can and only can reach every integer z in [0, x + y].
    """
    from fractions import gcd
    return z == 0 or x + y >= z and z % gcd(x, y) == 0


"""
Minimum Moves to Equal Array Elements
https://leetcode.com/problems/minimum-moves-to-equal-array-elements/

Given a non-empty integer array of size n, find the minimum number of moves required to make all array elements equal, where a move is incrementing n - 1 elements by 1.

Example:

Input:
[1,2,3]

Output:
3

Explanation:
Only three moves are needed (remember each move increments two elements):

[1,2,3]  ->  [2,3,3]  ->  [3,4,3]  ->  [4,4,4]
"""
def minMoves(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    return sum(nums) - len(nums) * min(nums)

"""
You are planning the amount of fuel need to complete a drone flight.
To fly higher, the drone burns 1 liter of fuel per feet. However, flying lower charges the drone with the amount of energy equivalent to 1 liter of fuel for every feet. Flying sideways takes no energy (only flying up and down takes/charges energy).

Given an array of 3D coordinates named route, find the minimal amount of fuel the drone would need to fly through this route.
Explain and code the most efficient solution possible, with the minimal number of actions and variables.

Example:
Completing the route [{z:10}, {z:0}, {z:6}, {z:15}, {z:8}] requires a minimum of 5 liters of fuel.

0,2,10
10,10,10
10,10,8

10,0,6,15,8
10,-6,-9,7 # gain/loss
10,4,-5,2

10,4,0
"""
# function calcFuelSimple(zRoute):
#    maxHeight = zRoute[0]
#    for i from 1 to length(zRoute)-1:
#       if (zRoute[i] > maxHeight):
#          maxHeight = zRoute[i]
#    return maxHeight - zRoute[0]

def minFuel(arr): # 10,0,6,15,8
   gain_loss = [arr[0]] + [arr[i]-arr[i+1] for i in range(1,len(arr))] # 10,-6,-9,7
   min_inc_sum, inc_sum = 0, arr[0]
   for i, gl in enumerate(gain_loss): # 10,-6,-9,7
      inc_sum += (arr[i]-arr[i+1])
      if inc_sum < min_inc_sum:
         min_inc_sum = inc_sum # -5
   return abs(min_inc_sum)