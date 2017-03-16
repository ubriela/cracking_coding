"""
Gray Code
https://leetcode.com/problems/gray-code/
The gray code is a binary numeral system where two successive values differ in only one bit.

Given a non-negative integer n representing the total number of bits in the code, print the sequence of gray code. A gray code sequence must begin with 0.

For example, given n = 2, return [0,1,3,2]. Its gray code sequence is:

00 - 0
01 - 1
11 - 3
10 - 2
Note:
For a given n, a gray code sequence is not uniquely defined.

For example, [0,2,3,1] is also a valid gray code sequence according to the above definition.

For now, the judge is able to judge based on one instance of gray code sequence. Sorry about that.
"""
from collections import defaultdict, deque
def grayCode(n):
    """
    :type n: int
    :rtype: List[int]
    """

    """
    dict of consecutive for each gray code
    """
    # d = defaultdict(list)

    # if n == 0:
    #     return [0]

    # for i in range(2**n-1):
    #     for j in range(i+1,2**n):
    #         # if i^j is a power of 2
    #         xor = i^j
    #         if not (xor & (xor-1)):
    #             d[i].append(j)
    #             d[j].append(i)

    # stack, visited = [0], set()
    # res = []
    # while stack:
    #     code = stack[-1]
    #     if code not in visited:
    #         visited.add(code)
    #         res.append(code)
    #         if len(res) == 2**n:
    #             return res
    #         for neighbor in d[code]:
    #             if neighbor not in visited:
    #                 stack.append(neighbor)

    '''
    from up to down, then left to right

    0   1   11  110
            10  111
                101
                100

    start:      [0]
    i = 0:      [0, 1]
    i = 1:      [0, 1, 3, 2]
    i = 2:      [0, 1, 3, 2, 6, 7, 5, 4]
    6 = 2 + 2^2
    7 = 3 + 2^2
    '''
    results = [0]
    for i in range(n):
        results += [x + pow(2, i) for x in reversed(results)]
    return results

"""
Word Ladder
https://leetcode.com/problems/word-ladder/
Given two words (beginWord and endWord), and a dictionary's word list, find the length of shortest transformation sequence from beginWord to endWord, such that:

Only one letter can be changed at a time.
Each transformed word must exist in the word list. Note that beginWord is not a transformed word.
For example,

Given:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log","cog"]
As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
return its length 5.

Note:
Return 0 if there is no such transformation sequence.
All words have the same length.
All words contain only lowercase alphabetic characters.
You may assume no duplicates in the word list.
You may assume beginWord and endWord are non-empty and are not the same.
"""
from collections import deque
def ladderLength(beginWord, endWord, wordList):
    """
    :type beginWord: str
    :type endWord: str
    :type wordList: List[str]
    :rtype: int
    """

    # concise solution

    # wordSet = set(wordList)
    # wordSet.add(beginWord)
    # queue = deque([[beginWord, 1]])
    # while queue:
    #     word, length = queue.popleft()
    #     if word == endWord:
    #         return length
    #     for i in range(len(word)):
    #         for c in 'abcdefghijklmnopqrstuvwxyz':
    #             next_word = word[:i] + c + word[i + 1:]
    #             if next_word in wordSet:
    #                 wordSet.remove(next_word)
    #                 queue.append([next_word, length + 1])
    # return 0

    """
    fast solution with pre-process
        Creates a map of all combinations of words with missing letters mapped
        to all words in the list that match that pattern.
        E.g. hot -> {'_ot': ['hot'], 'h_t': ['hot'], 'ho_': ['hot']}
    """
    # O(N), N is the number of words
    def construct_dict(word_list):
        d = defaultdict(list)
        for word in word_list:
            for i in range(len(word)):
                d[word[:i] + "_" + word[i+1:]].append(word)
        return d

    # O(N), N is the number of nodes
    def bfs_words(begin, end, dict_words):
        queue, visited = deque([(begin, 1)]), set()
        while queue:
            word, steps = queue.popleft()
            if word not in visited:
                visited.add(word)
                if word == end:
                    return steps
                for i in range(len(word)):
                    neigh_words = dict_words[word[:i] + "_" + word[i+1:]]
                    for neigh in neigh_words:
                        if neigh not in visited:
                            queue.append((neigh, steps + 1))
        return 0

    d = construct_dict(set(wordList))
    return bfs_words(beginWord, endWord, d)

"""
Generate a valid topological sequence
https://www.careercup.com/question?id=5710004859437056

Given a list of system packages, some packages cannot be installed until the other packages are installed.
Provide a valid sequence to install all of the packages.

e.g.,
a relies on b
b relies on c

A valid sequence is [c,b,a]
"""

from collections import defaultdict
class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    # add directed edge u->v
    def addEdge(self,u,v):
        self.graph[u].append(v)

    def dfs_recur(self,v,visited):
        visited.add(v)
        # recur for all vertices adjacent to this vertex
        for i in self.graph[v]:
            if i not in visited:
                self.dfs_recur(i, visited)

    def dfs(self,v):
        visited = set()

        # call recursive function
        self.dfs_recur(v,visited)

    # return a set of vertices
    def vertices(self):
        s = set()
        for v,l in self.graph.items():
            s.add(v)
            s.update(l)
        return list(s)


    def topological_sort_recur(self, v, visited, stack):
        visited.add(v)

        # recur for all vertices adjacent to this vertex
        for i in self.graph[v]:
            if i not in visited:
                self.topological_sort_recur(i, visited, stack)

        # push curr vertex to stack which stores result
        stack.append(v)

    def topological_sort(self):
        visited = set()
        stack = []

        # call recursive function from all vertices
        for i in self.vertices():
            if i not in visited:
                self.topological_sort_recur(i, visited, stack)

        stack.reverse()

    def check_topological_sort_recur(self, v, visited):
        # recur for all vertices adjacent to this vertex
        for i in self.graph[v]:
            return False if i in visited else self.check_topological_sort_recur(i, visited)

        return True

    """
    for each node v in seq
        add m to visited
        for each node m reachable from n
            if m in visited -> return False

    """
    def is_valid_topological_sequence(self, seq):
        visited = set()
        stack = []
        for i in seq:
            visited.add(i)
            if not self.check_topological_sort_recur(i, visited):
                return False
        return True

# g = Graph()
# g.addEdge(5,0)
# g.addEdge(4,0)
# g.addEdge(5,2)
# g.addEdge(2,3)
# g.addEdge(3,1)
# g.addEdge(4,1)
# g.dfs(5)
# g.topological_sort()
# print g.is_valid_topological_sequence([5, 2, 4, 3, 1, 0])


# print (firstUniqChar(None, "teeter"))

"""
The Celebrity Problem

In a party of N people, only one person is known to everyone. Such a person may be present in the party, if yes, (s)he doesn't know anyone in the party. We can only ask questions like "does A know B? ". Find the stranger (celebrity) in minimum number of questions.

We can describe the problem input as an array of numbers/characters representing persons in the party. We also have a hypothetical function know(A, B) which returns true if A knows B, false otherwise. How can we solve the problem.

"""

def knows(x, i):
    return True

def findCelebrity(self, n):
    """
    The first loop is to exclude n - 1 labels that are not possible to be a celebrity.
    After the first loop, x is the only candidate.
    The second and third loop is to verify x is actually a celebrity by definition.

    The key part is the first loop. To understand this you can think the knows(a,b) as a a < b comparison,
    if a knows b then a < b, if a does not know b, a > b. Then if there is a celebrity, he/she must be the "maximum" of the n people.

    However, the "maximum" may not be the celebrity in the case of no celebrity at all.
    Thus we need the second and third loop to check if x is actually celebrity by definition.

    The total calls of knows is thus 3n at most.
    One small improvement is that in the second loop we only need to check i in the range [0, x).
    """
    x = 0
    for i in range(n): # x is the only candidate
        if knows(x, i):
            x = i
    if any(knows(x, i) for i in range(n) if i != x): # check if x is actually celebrity by definition (x does not know anyone)
        return -1    # not exist
    if any(not knows(i, x) for i in range(n) if i != x):    # everyone shoud know x
        return -1
    return x
