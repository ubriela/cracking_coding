class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class RandomListNode(object):
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None

"""
Merge k Sorted Lists
https://leetcode.com/problems/merge-k-sorted-lists/

Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.
"""
def mergeKLists(lists):
    """
    :type lists: List[ListNode]
    :rtype: ListNode
    """
    from heapq import heappop, heapreplace, heapify
    dummy = node = ListNode(0)
    h = [(n.val, n) for n in lists if n]
    heapify(h)  # in-place linear time
    while h:
        v, n = h[0]
        if n.next is None:
            heappop(h)  # only change heap size when necessary
        else:
            heapreplace(h, (n.next.val, n.next))
        node.next = n
        node = node.next

    return dummy.next

"""
LRU Cache
Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: get and put.

get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
put(key, value) - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.

Follow up:
Could you do both operations in O(1) time complexity?

Example:

LRUCache cache = new LRUCache( 2 /* capacity */ );

cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // returns 1
cache.put(3, 3);    // evicts key 2
cache.get(2);       // returns -1 (not found)
cache.put(4, 4);    // evicts key 1
cache.get(1);       // returns -1 (not found)
cache.get(3);       // returns 3
cache.get(4);       // returns 4
"""
class Node:
    def __init__(self, k, v):
        self.key = k
        self.val = v
        self.prev = None
        self.next = None

class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        # self.dic = collections.OrderedDict()
        # self.remain = capacity

        self.capacity = capacity
        self.dic = dict()
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        # if key not in self.dic:
        #     return -1
        # v = self.dic.pop(key) # key in dic -> return
        # self.dic[key] = v   # set key as the newest one
        # return v

        if key in self.dic:
            n = self.dic[key]
            self._remove(n)
            self._add(n)
            return n.val
        return -1

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        # if key in self.dic:
        #     self.dic.pop(key)
        # else:
        #     if self.remain > 0:
        #         self.remain -= 1
        #     else:  # self.dic is full
        #         self.dic.popitem(last=False)
        # self.dic[key] = value

        if key in self.dic:
            self._remove(self.dic[key])
        n = Node(key, value)
        self._add(n)  # add n to double-linked list
        self.dic[key] = n  # update dic
        if len(self.dic) > self.capacity:  # remove head.next from double-linked list
            n = self.head.next
            self._remove(n)
            del self.dic[n.key]

    def _add(self, node):
        self.tail.prev.next, node.next, node.prev, self.tail.prev = node, self.tail, self.tail.prev, node

    def _remove(self, node):
        # n is in double-linked list
        node.prev.next, node.next.prev = node.next, node.prev


"""
Copy List with Random Pointer
https://leetcode.com/problems/copy-list-with-random-pointer/

A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null.

Return a deep copy of the list.

Show Company Tags
Show Tags
Show Similar Problems
"""
from collections import defaultdict
def copyRandomList(head):
    """
    :type head: RandomListNode
    :rtype: RandomListNode
    """

    # O(2n)
    dic = dict()
    m = n = head
    while m:
        dic[m] = RandomListNode(m.label)
        m = m.next
    while n:
        dic[n].next = dic.get(n.next)
        dic[n].random = dic.get(n.random)
        n = n.next
    return dic.get(head)

    # O(n)
    # dic = defaultdict(lambda: RandomListNode(0))
    # dic[None] = None
    # n = head
    # while n:
    #     dic[n].label = n.label
    #     dic[n].next = dic[n.next]
    #     dic[n].random = dic[n.random]
    #     n = n.next
    # return dic[head]

"""
Add Two Numbers
https://leetcode.com/problems/add-two-numbers/
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
"""
def addTwoNumbers(l1, l2):
    """
    :type l1: ListNode
    :type l2: ListNode
    :rtype: ListNode
    """
    """
    2->4->3
    5->6->4
    sum     carry
    7       0
    0       1
    8       0
    """
    prev = dummy = ListNode(0)
    carry = 0
    while l1 or l2 or carry:
        v1, v2 = 0, 0
        if l1:
            v1 = l1.val
            l1 = l1.next
        if l2:
            v2 = l2.val
            l2 = l2.next
        sum = v1 + v2 + carry
        carry = sum / 10
        prev.next = ListNode(sum % 10)
        prev = prev.next
    return dummy.next

"""
Palindrome Linked List
https://leetcode.com/problems/palindrome-linked-list/
Given a singly linked list, determine if it is a palindrome.

Follow up:
Could you do it in O(n) time and O(1) space?
"""
def isPalindrome(head):
    """
    :type head: ListNode
    :rtype: bool
    """
    # Reverse the first half while finding the middle.
    slow = fast = head
    rev = None
    while fast and fast.next:
        fast = fast.next.next
        slow.next, rev, slow = rev, slow, slow.next

    if fast:
        slow = slow.next  # when odd elements

    # Compare the reversed first half with the second half.
    while rev and rev.val == slow.val:
        slow, rev = slow.next, rev.next

    return not rev

"""
Intersection of Two Linked Lists
https://leetcode.com/problems/intersection-of-two-linked-lists/

Write a program to find the node at which the intersection of two singly linked lists begins.


For example, the following two linked lists:

A:          a1 -> a2
                    \
                     c1 -> c2 -> c3
                    /
B:     b1 -> b2 -> b3
begin to intersect at node c1.


Notes:

If the two linked lists have no intersection at all, return null.
The linked lists must retain their original structure after the function returns.
You may assume there are no cycles anywhere in the entire linked structure.
Your code should preferably run in O(n) time and use only O(1) memory.
"""

def getIntersectionNode(headA, headB):
    """
    :type head1, head1: ListNode
    :rtype: ListNode
    """
    # if not headA or not headB:
    #     return None
    # nodeA, nodeB = headA, headB

    # # compute lengths of each list
    # lenA = lenB = 0
    # while nodeA:
    #     lenA += 1
    #     nodeA = nodeA.next
    # while nodeB:
    #     lenB += 1
    #     nodeB = nodeB.next

    # # move longer list by a number of steps
    # long, short = (headA, headB) if lenA >= lenB else (headB, headA)
    # for i in range(abs(lenA-lenB)):
    #     if long and long.next:
    #         long = long.next
    #     else:
    #         return None

    # # move both long and short until they equal
    # while long and short and long != short:
    #     long, short = long.next, short.next


    # the idea is if you switch head, the possible difference between length would be countered.
    # On the second traversal, they either hit or miss.
    # if they meet, pa or pb would be the node we are looking for,
    # if they didn't meet, they will hit the end at the same iteration, pa == pb == None, return either one of them is the same,None
    # return long if long else None
    if headA is None or headB is None:
        return None

    pa = headA  # 2 pointers
    pb = headB

    while pa is not pb:
        # if either pointer hits the end, switch head and continue the second traversal,
        # if not hit the end, just move on to next
        pa = headB if pa is None else pa.next
        pb = headA if pb is None else pb.next

    return pa  # only 2 ways to get out of the loop, they meet or the both hit the end=None

"""
Linked List Cycle
https://leetcode.com/problems/linked-list-cycle/

Given a linked list, determine if it has a cycle in it.

Follow up:
Can you solve it without using extra space?
"""
def hasCycle(head):
    """
    :type head: ListNode
    :rtype: bool
    """
    if not head:
        return False
    slow = fast = head
    while slow.next and fast.next:
        slow = slow.next
        if fast.next.next:
            fast = fast.next.next
        else:
            return False
        if slow == fast:
            return True

    return False

def linkedList2Arr(l):
    arr = []
    while l:
        arr.append(l.val)
        l = l.next
    # print (arr)
    return arr

"""
Merge Two Sorted Lists
https://leetcode.com/problems/merge-two-sorted-lists/

Merge two sorted linked lists (ascending order) and return it as a new list.
The new list should be made by splicing together the nodes of the first two lists.

"""
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

def mergeTwoSortedLists(l1, l2):
    """
    :type l1: ListNode
    :type l2: ListNode
    :rtype: ListNode
    """
    """
    1->3->5->7->None
    2->4-->None

    1->2->3->4->5->7->None
    """

    # Sol 1: iteratively
    if not l1 or not l2:
        return l1 or l2

    dummy = prev = ListNode(0)  # head = dummy.next
    while l1 and l2:
        if l1.val > l2.val:
            l1, l2 = l2, l1 # swap l1, l2
        # now l1.val <= l2.val
        prev.next = l1
        prev = l1
        l1 = l1.next
    # 1->2->3->4

    # remaining elements l1->5->7->None
    prev.next = l1 or l2 # l1 if l1 else l2
    return dummy.next

    # Sol 2: recursive
    # if not l1 or not l2:
    #     return l1 or l2
    # if l1.val > l2.val:
    #     l1, l2 = l2, l1 # swap l1, l2
    #
    # # now l1.val <= l2.val
    # l1.next = mergeTwoSortedLists(l1.next, l2)
    # return l1

"""
Interleave two linked list
"""
def mergeTwoLinkedLists(l1, l2):
    """
    :type l1: ListNode
    :type l2: ListNode
    :rtype: ListNode
    """
    """
    1->3->5->7->None
    2->4->None

    1->2->3->4->5->7->None
    """
    if not l1:
        return l2
    if not l2:
        return l1
    head = l1
    prev_l1, prev_l2 = None, None
    while l1 and l2:
        l1_next, l2_next = l1.next, l2.next
        l1.next, l2.next = l2, l1_next
        prev_l1, prev_l2 = l1, l2
        l1, l2 = l1_next, l2_next
    # 1->2->3->4->None(l2)

    # remaining elements l1->5->7->None
    if l1:
        prev_l2.next = l1
    elif l2:
        prev_l1.next = l2
    return head

"""
Reverse Linked List
https://leetcode.com/problems/reverse-linked-list/

Hint:
A linked list can be reversed either iteratively or recursively. Could you implement both?
"""

newHead = None  # update of new head

def reverseList(head, prev = None):
    """
    :type head: ListNode
    :rtype: ListNode
    """

    """
    iteratively
    """
    # Sol 1.A
    # if not head or not head.next:
    #     return head

    # prev = None
    # while head:
    #     curr = head
    #     head = head.next
    #     curr.next = prev
    #     prev = curr
    # return prev

    # Sol 1.B: short solution
    # rev = None
    # while head:
    #     head.next, rev, head = rev, head, head.next
    # return rev

    """
    recursive
    """
    if not head:
        return prev
    next, head.next = head.next, prev
    return reverseList(next, head)


def _reverse(node, prev=None):
    if not node:
        return prev
    next = node.next
    node.next = prev
    return _reverse(next, node)


"""
Odd Even Linked List
https://leetcode.com/problems/odd-even-linked-list/

Given a singly linked list, group all odd nodes together followed by the even nodes.
Please note here we are talking about the node number and not the value in the nodes.

You should try to do it in place. The program should run in O(1) space complexity and O(nodes) time complexity.

Example:
Given 1->2->3->4->5->NULL,
return 1->3->5->2->4->NULL.

Note:
The relative order inside both the even and odd groups should remain as it was in the input.
The first node is considered odd, the second node even and so on ...

Solution:
the idea is two different linked list, odd and even, by iterating one loop though
the input linked list
"""

# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

def oddEvenList(head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if not head:
        return None
    if not head.next:
        return head

    odd = head_odd = head
    even = head_even = head.next
    head = head.next.next
    while head:
        odd.next = head
        even.next = head.next
        odd = odd.next
        even = even.next
        head = head.next.next if even else None

    odd.next = head_even
    return head_odd


"""
Submission Details
https://leetcode.com/problems/circular-array-loop/

You are given an array of positive and negative integers. If a number n at an index is positive, then move forward n steps. Conversely, if it's negative (-n), move backward n steps. Assume the first element of the array is forward next to the last element, and the last element is backward next to the first element. Determine if there is a loop in this array. A loop starts and ends at a particular index with more than 1 element along the loop. The loop must be "forward" or "backward'.

Example 1: Given the array [2, -1, 1, 2, 2], there is a loop, from index 0 -> 2 -> 3 -> 0.

Example 2: Given the array [-1, 2], there is no loop.

Note: The given array is guaranteed to contain no element "0".

Can you do it in O(n) time complexity and O(1) space complexity?
"""
def getIndex(self, i, nums):
    n = len(nums)
    if i + nums[i] >= 0:
        return (i + nums[i]) % n
    else:
        return n - (i + nums[i]) % n

def circularArrayLoop(self, nums):
    """
    :type nums: List[int]
    :rtype: bool
    """

    """
    no zero

     [2, -1, 1, 2, 2]
     2(0) -> 1(2) -> 2(3)->2(0)
     0->2->3->0
     loop happen when an index is traversed before
     how?
     linear scan
        if curr val is 0 -> found loop
            traverse back to find index of the fist 0
        else:
            assign curr val to 0
            jump val step
     0,0 -> 0,2 -> 0,3 -> 0,0

     back-traverse, while prev val == 0
     0,0
        previous index = 3, gap = 2 -> prev val = 2
     2,3
        prev index = 2, gap=1 -> prev val = 1
     1,2
        prev index = 0, gap=2 -> prev val = 2
     end

     0,3,2,0

     [-1, 2]
     -1(0)->2(1)->2(1)
     not exist when index of next val is the same as current one
     linear scan
        0,0->0,1->0,1
    """
    n = len(nums)
    for i in range(n):
        if nums[i] == 0:
            continue
        # slow/fast pointer
        j = i
        k = self.getIndex(i, nums)
        while nums[k] * nums[i] > 0 and nums[self.getIndex(k, nums)] * nums[i] > 0:
            if j == k:
                # check for loop with only one element
                if j == self.getIndex(j, nums):
                    break
                return True
            j = self.getIndex(j, nums)
            k = self.getIndex(self.getIndex(k, nums), nums)
        # loop not found, set all element along the way to 0
        j = i
        val = nums[i]
        while nums[j] * val > 0:
            next = self.getIndex(j, nums)
            nums[j] = 0
            j = next
    return False


"""
Remove Duplicates from Sorted List
Given a sorted linked list, delete all duplicates such that each element appear only once.

For example,

Given 1->1->2, return 1->2.

Given 1->1->2->3->3, return 1->2->3.
"""

def deleteDuplicates(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if not head:
        return None
    curr = head
    s = {curr.val}
    while curr.next:
        temp = curr.next
        if temp.val not in s:
            curr = temp
            s.add(temp.val)
            continue
        else:
            curr.next = temp.next
    return head