"""
Longest Palindromic Subsequence
https://leetcode.com/problems/longest-palindromic-subsequence/?tab=Description

Given a string s, find the longest palindromic subsequence's length in s.
You may assume that the maximum length of s is 1000.

Example 1:
Input:

"bbbab"
Output:
4
One possible longest palindromic subsequence is "bbbb".
Example 2:
Input:

"cbbd"
Output:
2
"""
def longestPalindromeSubseq(s):
    """
    :type s: str
    :rtype: int
    """
    # dp[i][j] = longest palindrome subsequence of s[i to j].
    # If s[i] == s[j], dp[i][j] = 2 + dp[i+1][j - 1]
    # Else, dp[i][j] = max(dp[i+1][j], dp[i][j-1])

    # Rolling array O(2n) space
    # n = len(s)
    # dp = [[1] * 2 for _ in range(n)]
    # for j in xrange(1, len(s)):
    #     for i in reversed(xrange(0, j)):
    #         if s[i] == s[j]:
    #             dp[i][j%2] = 2 + dp[i + 1][(j - 1)%2] if i + 1 <= j - 1 else 2
    #         else:
    #             dp[i][j%2] = max(dp[i + 1][j%2], dp[i][(j - 1)%2])
    # return dp[0][(n-1)%2]

    # Further improve space to O(n)
    n = len(s)
    dp = [1] * n
    for j in range(1, len(s)):
        pre = dp[j]
        for i in reversed(range(0, j)):
            tmp = dp[i]
            if s[i] == s[j]:
                dp[i] = 2 + pre if i + 1 <= j - 1 else 2
            else:
                dp[i] = max(dp[i + 1], dp[i])
            pre = tmp
    return dp[0]

"""
Longest Palindromic Substring
https://leetcode.com/problems/longest-palindromic-substring/
Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.

Example:

Input: "babad"

Output: "bab"

Note: "aba" is also a valid answer.
Example:

Input: "cbbd"

Output: "bb"

Solution 1:
We observe that a palindrome mirrors around its center.
Therefore, a palindrome can be expanded from its center,
and there are only 2N-1 such centers.

Time O(N*N), space O(1)

Solution 2: Dynamic programming
Time O(N*N), space O(N*N)

The manchester algorithm is O(1) time
http://articles.leetcode.com/longest-palindromic-substring-part-ii
"""
def longestPalindrome(s):
    """
    :type s: str
    :rtype: str
    """
    def _expandAroundCenter(s, c1, c2):
        l, r = c1, c2
        while l >= 0 and r <= len(s) - 1 and s[l] == s[r]:
            l -= 1
            r += 1
        return s[l + 1: r]

    res = s[0]
    for i in range(len(s) - 1):
        curr = _expandAroundCenter(s, i, i)
        if len(curr) > len(res):
            res = curr
        curr = _expandAroundCenter(s, i, i + 1)
        if len(curr) > len(res):
            res = curr
    return res

"""
Repeated Substring Pattern
https://leetcode.com/problems/repeated-substring-pattern/
Given a non-empty string check if it can be constructed by taking a substring of it and
appending multiple copies of the substring together. You may assume the given string consists of
 lowercase English letters only and its length will not exceed 10000.

Example 1:
Input: "abab"

Output: True

Explanation: It's the substring "ab" twice.
Example 2:
Input: "aba"

Output: False
Example 3:
Input: "abcabcabcabc"

Output: True

Explanation: It's the substring "abc" four times. (And the substring "abcabc" twice.)

"""
def repeatedSubstringPattern(str):
    """
    :type str: str
    :rtype: bool
    """
    # Sol 1
    # return any(str[:i] * (len(str) / i) == str for i in range(1, len(str)) if len(str) % i == 0)

    # Sol 2
    # if str == None:
    #     return False
    # strLen = len(str)
    # for i in range(strLen/2):
    #     substring = str[0:i+1]
    #     if strLen % (i+1) == 0:
    #         multiplier = strLen/(i+1)
    #         if str == substring*multiplier:
    #             return True
    # return False

    """
    Sol 3: the fastest
    check every length of the string
    complexity: n+n/2+n/3+n/4+...+1/2
    """
    if len(str) <= 1:
        return False
    l = len(str)
    for i in range(1, int(l/2)+1):    # in the worst case length of substring is l/2
        if l % i == 0:  # l is a multiple of i
            res = True
            # check every substring, making sure they are all equal to the first substring
            for j in range(1, int(l/i)):
                if str[0:i] != str[i*j:i*j + i]:
                    res = False
                    break
            if res:
                return res
    return False

"""
https://www.careercup.com/question?id=5659201272545280
Given an input string and ordering string, need to return true if
the ordering string is present in Input string

Input = "hello world!"
ordering = "!od"
result = False

Input = "hello world!"
ordering = "he!"
result = True

Assume no duplicate chars in ordering

The idea is to keep track of the last position of every char in input string

Size of ordering string = m
Size of input string = n
Time Complexity = O(n) + O(m) == O(n+m)
Space complexity = O(min(n, 26))

pos_map = {'!': 11, ' ': 5, 'e': 1, 'd': 10, 'h': 0, 'l': 9, 'o': 7, 'r': 8, 'w': 6}
------
With duplicate chars -> the value of a particular char would be a list of positions it appears
"""
def isOrdered(input, order):
    # input = "hello world!"
    # pos_map = {'!': 11, ' ': 5, 'e': 1, 'd': 10, 'h': 0, 'l': 9, 'o': 7, 'r': 8, 'w': 6}
    pos_map = dict() # the last index of a character
    for i,c in enumerate(input):
        pos_map[c] = i

    for i in range(1,len(order)):
        # pos_map[order[i]] should be greater than pos_map[order[i-1]]
        if pos_map[order[i]] - pos_map[order[i-1]] < 0:
            return False

    return True


# print isOrdered("hello world!", "hee!")


"""
Smallest Substring of All Characters
https://www.pramp.com/question/wqNo9joKG6IJm67B6z34

Given an array with UNIQUE characters arr and a string str, find the smallest substring of str
containing all characters of arr.

Example:
arr: [x,y,z], str: xyyzyzyx
result: zyx

Scanning though the str, using two pointers to keep track of the number of unique characters between the two pointer
    move head pointer forward
        skip characters that are not in arr
        update count_map and number of unique chars
        while number of unique chars equals to len(arr)
            move tail pointer forward

O(m+n) time complexity
O(m) space complexity, m is len(arr)

"""
def minWindow0(arr, str):
    if not arr or not str:
        return None
    len_arr, len_str = len(arr), len(str)
    if len_arr > len_str:
        return None

    # initialize count map from arr
    unique_counter = 0
    count_map = dict()
    for c in arr:
        count_map[c] = 0
    res = None
    t = 0 # tail

    # scan through str
    for h, head in enumerate(str):
        if head not in count_map:
            continue
        head_count = count_map[head]
        count_map[head] = head_count + 1
        if head_count == 0:
            unique_counter += 1

        # print head, unique_counter

        # push tail forward
        while unique_counter == len_arr:
            curr_len = h - t + 1
            if curr_len == len_arr: # found the minimum
                return str[t:h+1]

            if not res or curr_len < len(res): # update the smaller substring
                res = str[t:h+1]

            tail = str[t]
            if tail in count_map:
                tail_count = count_map[tail] - 1
                count_map[tail] = tail_count
                if tail_count == 0:
                    unique_counter -= 1
            t += 1

    return res

"""
Minimum Window Substring
https://leetcode.com/problems/minimum-window-substring/

Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).

For example,
S = "ADOBECODEBANC"
T = "ABC"
Minimum window is "BANC".

Note:
If there is no such window in S that covers all characters in T, return the empty string "".

If there are multiple such windows, you are guaranteed that there will always be only one unique minimum window in S.

Solution:
The current window is s[i:j] and the result window is s[I:J].
In count_map[c] I store how many times I need character c (can be negative) and missing tells how many characters are still missing.
In the loop, first add the new character to the window.
Then, if nothing is missing, remove as much as possible from the window start and then update the result.

print (minWindow("xyyzyzyx", "xyz"))
log:
1 x Counter({'y': 1, 'z': 1, 'x': 0})
2 y Counter({'z': 1, 'x': 0, 'y': 0})
3 y Counter({'z': 1, 'x': 0, 'y': -1})
4 z Counter({'x': 0, 'z': 0, 'y': -1})
    0 4 x Counter({'x': 0, 'z': 0, 'y': -1})
5 y Counter({'x': 0, 'z': 0, 'y': -2})
    0 5 x Counter({'x': 0, 'z': 0, 'y': -2})
6 z Counter({'x': 0, 'z': -1, 'y': -2})
    0 6 x Counter({'x': 0, 'z': -1, 'y': -2})
7 y Counter({'x': 0, 'z': -1, 'y': -3})
    0 7 x Counter({'x': 0, 'z': -1, 'y': -3})
8 x Counter({'x': -1, 'z': -1, 'y': -3})
    5 8 z Counter({'x': 0, 'y': 0, 'z': 0})
"""
import collections
def minWindow(str, arr):
    """
    :type s: str
    :type t: str
    :rtype: str
    """
    count_map, missing = collections.Counter(arr), len(arr)
    head = tail = i = 0
    # scan through str
    for j, c in enumerate(str, 1):
        # only reduce missing if c in arr
        missing -= count_map[c] > 0
        count_map[c] -= 1  # when value < 0 -> [tail:j] is a valid window
        # push tail forward
        if missing == 0:
            while i < j and count_map[str[i]] < 0:
                count_map[str[i]] += 1
                i += 1
            if not head or j - i <= head - tail:
                head, tail = j, i
    return str[tail:head]

"""
Reverse a string without using extra space
input: "to trong hien"
output: "hien trong to"
"""
# 1. reverse input string
# 2. reverse individual word
def reverseString(s):
    s = list(s)
    s.reverse()

    index = 0
    for i in range(len(s)):
        if s[i] == " ":
            s[index: i] = reversed(s[index: i])
            index = i + 1

    s[index:] = reversed(s[index:])
    return ''.join(s)

# print (reverseString("hello world!"))

"""
Remove characters from the first string which are present in the second string.

For example, given a str of "Battle of the Vowels: Hawaii vs. Grozny" and a remove of "aeiou",
the function should transform str to "Bttl f th Vwls: Hw vs. Grzny".
Justify any design decisions you make and discuss the efficiency of your solution.

complexity = O(m+n) where m,n are the lengths of str and remove strings, respectively.

"""
def removeDirtyChars(s, mask):
    flags = [False] * 128 # set flags for characters to be removed
    for c in mask:
        flags[ord(c)] = True

    dst = 0
    s_list = list(s)
    # loop through all characters, copying if they are not flagged
    for c in s_list:
        if not flags[ord(c)]:
            s_list[dst] = c # this code simulates in-line sort
            dst += 1
    return ''.join(s_list[0:dst])

# print (removeDirtyChars("Battle of the Vowels: Hawaii vs. Grozny", "aeiou"))


"""
Remove Duplicate Letters
Given a string which contains only lowercase letters, remove duplicate letters so that every letter appear once and only once.
You must make sure your result is the smallest in lexicographical order among all possible results.

Example:

Given "bcabc"
Return "abc"

Given "cbacdcbc"
Return "acdb"
"""
def removeDuplicateLetters(s):
    """
    :type s: str
    :rtype: str
    """
    if not s:
        return s
    shash = [0] * 26
    for i,c in enumerate(s):
        shash[ord(c)-ord('a')] += 1

    pos = 0 #
    for i,c in enumerate(s):
        if c < s[pos]:
            pos = i
        shash[ord(c)-ord('a')] -= 1
        if shash[ord(c)-ord('a')] == 0:
            break
    return str(s[pos]) + removeDuplicateLetters(s[pos+1:].replace(s[pos], '')) if pos < len(s)-1 else str(s[pos])

"""
First Unique Character in a String
Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1.

Examples:

s = "leetcode"
return 0.

s = "loveleetcode",
return 2.
"""

def firstUniqChar(s):
    """
    :type s: str
    :rtype: int
    """
    shash = [0] * 26
    for c in s:
        shash[ord(c) - ord('a')] += 1
    for i,c in enumerate(s):
        if shash[ord(c) - ord('a')] == 1: # return the first unique char
            return i
    # not found
    return -1



"""
Find All Anagrams in a String
Given a string s and a non-empty string p, find all the start indices of p's anagrams in s.

Strings consists of lowercase English letters only and the length of both strings s and p will not be larger than 20,100.

The order of output does not matter.

Example 1:

Input:
s: "cbaebabacd" p: "abc"

Output:
[0, 6]

Explanation:
The substring with start index = 0 is "cba", which is an anagram of "abc".
The substring with start index = 6 is "bac", which is an anagram of "abc".

Example 2:

Input:
s: "abab" p: "ab"

Output:
[0, 1, 2]

Explanation:
The substring with start index = 0 is "ab", which is an anagram of "ab".
The substring with start index = 1 is "ba", which is an anagram of "ab".
The substring with start index = 2 is "ab", which is an anagram of "ab".
"""
def findAnagrams(s, p):
    """
    :type s: str
    :type p: str
    :rtype: List[int]
    """
    res = []
    m, n = len(s), len(p)
    phash, shash = [0]*123, [0]*123

    # init shash
    for c in p:
        phash[ord(c)] += 1

    # init chash
    for c in s[0:n-1]:
        shash[ord(c)] += 1

    # increase window shash
    for i in range(n-1,m):
        shash[ord(s[i])] += 1 # add one more char
        if phash == shash:
            res.append(i-n+1)
        shash[ord(s[i-n+1])] -= 1   # subtract one char
    return res


"""
Valid Anagram
Given two strings s and t, write a function to determine if t is an anagram of s.

For example,

s = "anagram", t = "nagaram", return true.

s = "rat", t = "car", return false.

Note:

You may assume the string contains only lowercase alphabets.

Follow up:

What if the inputs contain unicode characters? How would you adapt your solution to such case?
"""
def isAnagram1(s, t):
    dic1, dic2 = {}, {}
    for item in s:
        dic1[item] = dic1.get(item, 0) + 1
    for item in t:
        dic2[item] = dic2.get(item, 0) + 1
    return dic1 == dic2

def isAnagram2(s, t):
    dic1, dic2 = [0]*26, [0]*26
    for item in s:
        dic1[ord(item)-ord('a')] += 1
    for item in t:
        dic2[ord(item)-ord('a')] += 1
    return dic1 == dic2