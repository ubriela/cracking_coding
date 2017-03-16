"""
Insert Delete Search GetRandom O(1)
https://leetcode.com/problems/insert-delete-getrandom-o1/
http://www.geeksforgeeks.org/design-a-data-structure-that-supports-insert-delete-search-and-getrandom-in-constant-time/

Design a data structure that supports all following operations in average O(1) time.

insert(val): Inserts an item val to the set if not already present.
remove(val): Removes an item val from the set if present.
getRandom: Returns a random element from current set of elements. Each element must have the same probability of being returned.
Example:

// Init an empty set.
RandomizedSet randomSet = new RandomizedSet();

// Inserts 1 to the set. Returns true as 1 was inserted successfully.
randomSet.insert(1);

// Returns false as 2 does not exist in the set.
randomSet.remove(2);

// Inserts 2 to the set, returns true. Set now contains [1,2].
randomSet.insert(2);

// getRandom should return either 1 or 2 randomly.
randomSet.getRandom();

// Removes 1 from the set, returns true. Set now contains [2].
randomSet.remove(1);

// 2 was already in the set, so return false.
randomSet.insert(2);

// Since 2 is the only number in the set, getRandom always return 2.
randomSet.getRandom();
"""
from collections import defaultdict
import random
class RandomizedSet(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        """
        4 -> 0
        3 -> 1
        5 -> 2
        2 -> 3
        remove 3
        3 at index 1
        2 at index 3
        4 -> 0
        5 -> 2
        2 -> (3 -> 1)
        4 3 5 2 -> 4 2 5 3


        """
        self.indices = defaultdict()
        self.values = []

    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self.indices:
            self.indices[val] = len(self.values)
            self.values.append(val)
            return True
        return False

    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.indices:
            index_remove = self.indices[val]
            self.indices.pop(val)
            if index_remove != len(self.values) - 1:
                self.indices[self.values[-1]] = index_remove
                # swap the removed value with the last
                self.values[index_remove], self.values[-1] = self.values[-1], self.values[index_remove]
            self.values.pop()
            return True
        return False

    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """
        if len(self.values) >= 1:
            return self.values[random.randint(0, len(self.values) - 1)]


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()

"""
Flatten Nested List Iterator
https://leetcode.com/problems/flatten-nested-list-iterator/

Given a nested list of integers, implement an iterator to flatten it.

Each element is either an integer, or a list -- whose elements may also be integers or other lists.

Example 1:
Given the list [[1,1],2,[1,1]],

By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: [1,1,2,1,1].

Example 2:
Given the list [1,[4,[6]]],

By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: [1,4,6].

The idea is to keep a stack of values: both lists, integer. Keep popping out items from stack until found an integer.
"""
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
# class NestedInteger(object):
#    def isInteger(self):
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        :rtype bool
#        """
#
#    def getInteger(self):
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        :rtype int
#        """
#
#    def getList(self):
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        :rtype List[NestedInteger]
#        """
from collections import deque


class NestedIterator(object):
    def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        # self.nestedList = nestedList
        self.stack = []
        for nestedInteger in reversed(nestedList):
            self.stack.append(nestedInteger)

    def next(self):
        """
        :rtype: int
        """
        i = self.stack.pop()
        return i.getInteger()

    def hasNext(self):
        """
        :rtype: bool
        """
        while self.stack:  # keep popping from stack until we have integer
            peak = self.stack[-1]
            if peak.isInteger():
                return True
            self.stack.pop()
            nestedList = peak.getList()
            for elem in reversed(nestedList):
                self.stack.append(elem)
        return False

# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())

"""
Peeking Iterator
https://leetcode.com/problems/peeking-iterator/
Given an Iterator class interface with methods: next() and hasNext(),
design and implement a PeekingIterator that support the peek() operation --
it essentially peek() at the element that will be returned by the next call to next().

Here is an example. Assume that the iterator is initialized to the beginning of the list:
[1, 2, 3].

Call next() gets you 1, the first element in the list.

Now you call peek() and it returns 2, the next element. Calling next() after that still return 2.

You call next() the final time and it returns 3, the last element. Calling hasNext()
after that should return false.
"""

class PeekingIterator(object):
    """
    The idea is to ooking ahead: every time we perform next()
    we return previous value as the next value
    """
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.iterator = iterator
        self.prev = self.iterator.next() if self.iterator.hasNext() else None

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        return self.prev

    def next(self):
        """
        :rtype: int
        """
        ret = self.prev
        self.prev = self.iterator.next() if self.iterator.hasNext() else None
        return ret

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.prev is not None


"""
Min Stack
https://leetcode.com/problems/min-stack/

Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

push(x) -- Push element x onto stack.
pop() -- Removes the element on top of the stack.
top() -- Get the top element.
getMin() -- Retrieve the minimum element in the stack.
Example:
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> Returns -3.
minStack.pop();
minStack.top();      --> Returns 0.
minStack.getMin();   --> Returns -2.
"""
class MinStack(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.s1 = []  # main stack
        self.s2 = []  # min stack

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.s1.append(x)

        # only append to s2 if empty or curr < s2's peak
        if not self.s2 or x <= self.s2[-1]:  # equal condition is important to make sure that s2 is empty if s2 is empty
            self.s2.append(x)

    def pop(self):
        """
        :rtype: void
        """
        peek = self.s1.pop()

        # only remove from s2 if not empty and curr == s2's peek
        if self.s2 and self.s2[-1] == peek:
            self.s2.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.s1[-1]

    def getMin(self):
        """
        :rtype: int
        """
        if self.s2:
            return self.s2[-1]

"""
Design Twitter
https://leetcode.com/problems/design-twitter/

Design a simplified version of Twitter where users can post tweets, follow/unfollow another user and is able to see the 10 most recent tweets in the user's news feed. Your design should support the following methods:

postTweet(userId, tweetId): Compose a new tweet.
getNewsFeed(userId): Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
follow(followerId, followeeId): Follower follows a followee.
unfollow(followerId, followeeId): Follower unfollows a followee.
Example:

Twitter twitter = new Twitter();

// User 1 posts a new tweet (id = 5).
twitter.postTweet(1, 5);

// User 1's news feed should return a list with 1 tweet id -> [5].
twitter.getNewsFeed(1);

// User 1 follows user 2.
twitter.follow(1, 2);

// User 2 posts a new tweet (id = 6).
twitter.postTweet(2, 6);

// User 1's news feed should return a list with 2 tweet ids -> [6, 5].
// Tweet id 6 should precede tweet id 5 because it is posted after tweet id 5.
twitter.getNewsFeed(1);

// User 1 unfollows user 2.
twitter.unfollow(1, 2);

// User 1's news feed should return a list with 1 tweet id -> [5],
// since user 1 is no longer following user 2.
twitter.getNewsFeed(1);
"""
from collections import defaultdict
import heapq
class Twitter(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.time = 0
        self.followees = defaultdict(set)  # uid, {friends's uid}
        self.posts = defaultdict(list)  # uid, [tweetid]

    def postTweet(self, userId, tweetId):
        """
        Compose a new tweet.
        :type userId: int
        :type tweetId: int
        :rtype: void
        """
        self.time += 1
        self.posts[userId].append((self.time, tweetId))

    def getNewsFeed(self, userId):
        """
        Retrieve the 10 most recent tweet ids in the user's news feed.
        Each item in the news feed must be posted by users who the user
        followed or by the user herself. Tweets must be ordered from most
        recent to least recent.
        :type userId: int
        :rtype: List[int]
        """
        # obtain the list of tweetid with highest time
        h = []  # min heap

        # add 10 recent tweets of this user
        for followee in self.followees[userId].union(set([userId])):
            for post in self.posts[followee][-10:]:
                if len(h) >= 10:
                    heapq.heappushpop(h, post)
                else:
                    heapq.heappush(h, post)

        # print
        res = []
        while h:
            res.append(heapq.heappop(h)[1])
        return res[::-1]

    def follow(self, followerId, followeeId):
        """
        Follower follows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int
        :type followeeId: int
        :rtype: void
        """
        if followerId == followeeId:
            return
        self.followees[followerId].add(followeeId)

    def unfollow(self, followerId, followeeId):
        """
        Follower unfollows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int
        :type followeeId: int
        :rtype: void
        """
        self.followees[followerId].discard(followeeId)


# Your Twitter object will be instantiated and called as such:
# obj = Twitter()
# obj.postTweet(userId,tweetId)
# param_2 = obj.getNewsFeed(userId)
# obj.follow(followerId,followeeId)
# obj.unfollow(followerId,followeeId)


"""
Delete a Linked List node at a given position
http://quiz.geeksforgeeks.org/delete-a-linked-list-node-at-a-given-position/
Given a singly linked list and a position, delete a linked list node at the given position. (without modifying value of the node)
Example:
Input: position = 1, Linked List = 8->2->3->1->7
Output: Linked List =  8->3->1->7

Input: position = 0, Linked List = 8->2->3->1->7
Output: Linked List = 2->3->1->7

If node to be deleted is root, simply delete it.
To delete a middle node, we must have pointer to the node previous to the node to be deleted.
So if positions is not zero, we run a loop position-1 times and get pointer to the previous node.
"""


# Python program to delete a node in a linked list
# at a given position

# Node class
class Node:
    # Constructor to initialize the node object
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    # Constructor to initialize head
    def __init__(self):
        self.head = None

    # Function to insert a new node at the beginning
    def push(self, new_data):
        new_node = Node(new_data)
        new_node.next = self.head
        self.head = new_node

    # Given a reference to the head of a list
    # and a position, delete the node at a given position
    def deleteNode(self, position):

        # If linked list is empty
        if self.head == None:
            return

        # Store head node
        temp = self.head

        # If head needs to be removed
        if position == 0:
            self.head = temp.next
            temp = None
            return

        # Find previous node of the node to be deleted
        for i in range(position - 1):
            temp = temp.next
            if temp is None:
                break

        # If position is more than number of nodes
        if temp is None:
            return
        if temp.next is None:
            return

        # Node temp.next is the node to be deleted
        # store pointer to the next of node to be deleted
        next = temp.next.next

        # Unlink the node from linked list
        temp.next = None

        temp.next = next

    # Utility function to print the linked LinkedList
    def printList(self):
        temp = self.head
        while (temp):
            print (" %d " % temp.data)
            temp = temp.next


# Driver program to test above function
# llist = LinkedList()
# llist.push(7)
# llist.push(1)
# llist.push(3)
# llist.push(2)
# llist.push(8)
#
# print "Created Linked List: "
# llist.printList()
# llist.deleteNode(4)
# print "\nLinked List after Deletion at position 4: "
# llist.printList()