# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

"""
Find successor of a node in a BST. Return None if not found.

The idea is to use binary search to find the node.
Keep track of the successor while searching.
Return the successor of the node when found

O(logN)
"""
def findSuccessorBST(root, node):
    if not root:
       return None
    return root if (root.left and root.left == node) or (root.right and root.right == node) else \
        findSuccessorBST(root.left, node) if node.val <= root.val else findSuccessorBST(root.right, node)

def findSuccessorBST_iter(root, node):
    curr = root
    while curr:
        if (curr.left and curr.left == node) or (curr.right and curr.right == node):
            return curr
        curr = curr.left if node.val <= curr.val else curr.right
    return curr

"""
To find largest element smaller than K in a BST

Given a root of a binary search tree (BST) and a key x, find the largest key in the tree that is smaller than x.

E.g.,
if an in-order list of all keys in the tree is {1, 2, 3, 4, 7, 17, 19, 21, 35, 89} and x is 19,
the biggest key that is smaller than x is 17.

node.left.val <= node.val < node.right.val
    4
   / \
  2   7
/ \  / \
1  3 6  9
"""
# iterative solution is better b/c we need to keep track of result
def largestSmallerKey(root, x):
    result = None
    while root:
        if root.val < x:
            result = root.val
            root = root.right
        else:
            root = root.left
    return result


"""
Find two nodes in a binary tree that sum up to a target.
Return those two nodes.

Brute force solution would be O(n^2)

Input
        4
      /   \
    2      7
   / \    / \
  1   3  6   9

Sum = 13

Return (4,9) or (6,7)
"""

"""
Better solution would be create a dictionary of traversed nodes (e.g., in-order traverse). Key is node.val and value is node. Then
find every time a node is checked, we check if the remainder of S-node.val is in the dictionary or not. If yes, then return (node, d[S-node.val]).
Return False if no pair found. This algorihm is O(n) but requires O(n) space.
"""
from collections import defaultdict
def findSumPairs(root, target):
    if not root:
        return False
    dic = defaultdict()
    s = [] # stack
    while s or root:
        if root:
            s.append(root)
            root = root.left
        else:
            root = s.pop()
            remainder = target - root.val
            if remainder in dic:
                return (dic[remainder].val, root.val)
            else:
                dic[root.val] = root
            root = root.right
    return False

"""
Space optimized solution: in-place convert BST to doubly linked list (DLL) and then find pair in DLL in O(n) time.
But this solution requires updates to BST.
"""

"""
The following solution is O(n) complexity and only requires O(logn) space.
The idea is to traverses BST in both in-order and reverse in-order. Given two current values of the traverse methods,
check if sum of them equal to target. If yes -> Found. If sum < target -> traverse to the next in-order value (move left); otherwise,
traverse to the next reverse in-order value (move right).
"""
def findSumPairsBi(root, target):
    """
    Return 2 nodes if found; otherwise, return Fasle
    :param root:
    :param target:
    :return:
    """
    s1 = [] # stack of inorder traverse
    s2 = [] # stack of reverse inorder traverse
    l, r = root, root
    n1, n2 = None, None
    move_left, move_right = True, True
    while True:
        # find left most and right most branch
        while l:
            s1.append(l)
            l = l.left
        while r:
            s2.append(r)
            r = r.right

        if move_left and s1:
            n1 = s1.pop()
        if move_right and s2:
            n2 = s2.pop()

        if n1 == n2: # not exist different nodes summing up to target
            return False
        sum = n1.val + n2.val

        if sum == target:
            return (n1.val, n2.val)
        elif sum < target: # move left
            move_left, move_right = True, False
        elif sum > target: # move right
            move_left, move_right = False, True

        # traverse right branch of the left cursor and left branch of the right cursor
        if n1:
            l = n1.right
        if n2:
            r = n2.left

    return False

"""
Validate Binary Search Tree
https://leetcode.com/problems/validate-binary-search-tree/

Given a binary tree, determine if it is a valid binary search tree (BST).

Assume a BST is defined as follows:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.
Example 1:
    2
   / \
  1   3
Binary tree [2,1,3], return true.
Example 2:
    1
   / \
  2   3
Binary tree [1,2,3], return false.
"""
def isValidBST(root, lessThan=float('inf'), largerThan=float('-inf')):
    """
    :type root: TreeNode
    :rtype: bool
    """
    """
    Recursive: making sure that both the right and the left branch are greater than
    and smaller than the current node
    """
    if not root:
        return True
    if root.val <= largerThan or root.val >= lessThan:
        return False
    left = isValidBST(root.left, min(lessThan, root.val), largerThan)
    right = isValidBST(root.right, lessThan, max(largerThan, root.val))
    return left and right

    """
        inorder traverse
        inorder traverse output node values in strictly increasing order
    """
    # def isValidBST(self, root):
    #     if not root:
    #         return True
    #     s = [] # stack
    #     res = [float('-inf')]
    #     while s or root:
    #         if root:
    #             s.append(root)
    #             root = root.left
    #         else:
    #             root = s.pop()
    #             if root.val <= res[-1]:
    #                 return False
    #             res.append(root.val)
    #             root = root.right
    #     return True

"""
Convert Sorted Array to Binary Search Tree (BST)
https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/

Given an array where elements are sorted in ascending order,
convert it to a height balanced BST.
"""
def sortedArrayToBST(nums):
    """
    :type nums: List[int]
    :rtype: TreeNode
    """
    """
    Solution: recursion
        root node at index : mid
        left branch : mid)
        right branch mid + 1 :
    """

    if not nums:
        return None

    mid = int(len(nums) / 2)

    root = TreeNode(nums[mid])
    root.left = sortedArrayToBST(nums[:mid])
    root.right = sortedArrayToBST(nums[mid + 1:])

    return root

"""
Serialize and Deserialize BST
https://leetcode.com/problems/serialize-and-deserialize-bst/

Serialization is the process of converting a data structure or object into a sequence of bits so
that it can be stored in a file or memory buffer, or transmitted across a network connection link
to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary search tree.
There is no restriction on how your serialization/deserialization algorithm should work.
You just need to ensure that a binary search tree can be serialized to a string and
this string can be deserialized to the original tree structure.

The encoded string should be as compact as possible.

Note: Do not use class member/global/static variables to store states.
Your serialize and deserialize algorithms should be stateless.


Solution:
    Use preorder traverse to serialize BST and
"""

# O(N)
def serializeBST(root):
    """Encodes a tree to a single string.

    :type root: TreeNode
    :rtype: str
    """

    """
            4
          /   \
        2      7
       / \    / \
      1   3  6   9
    preorder traverse: 4 2 1 3 7 6 9
    """
    res = []
    if not root:
        return ""
    s = [root]  # stack
    while s:
        node = s.pop()
        res.append(node.val)
        if node.right:
            s.append(node.right)
        if node.left:
            s.append(node.left)

    return ",".join(map(str, res))


# O(NlogN)
def deserializeBST(data):
    """
    Decodes your encoded data to tree.
    :type data: str
    :rtype: TreeNode
    """
    if not data:
        return None
    values = list(map(int, data.split(",")))
    root = TreeNode(values[0])
    for val in values[1:]:
        # find position to insert val into BST
        node = root
        while node:
            if val <= node.val:
                if not node.left:
                    node.left = TreeNode(val)
                    break
                node = node.left
            elif val > node.val:
                if not node.right:
                    node.right = TreeNode(val)
                    break
                node = node.right
    return root

"""
Lowest Common Ancestor in a Binary Search Tree.
http://www.geeksforgeeks.org/lowest-common-ancestor-in-a-binary-search-tree/

Given values of two nodes in a Binary Search Tree, write a c program to find the Lowest Common Ancestor (LCA).
You may assume that both the values exist in the tree.

If we are given a BST where every node has parent pointer, then LCA can be easily determined by traversing up using parent pointer and printing the first intersecting node.

We can solve this problem using BST properties. We can recursively traverse the BST from root.
The main idea of the solution is, while traversing from top to bottom, the first node n we encounter
with value between n1 and n2, i.e., n1 < n < n2 or same as one of the n1 or n2, is LCA of n1 and n2
(assuming that n1 < n2). So just recursively traverse the BST in, if node's value is greater than both n1 and n2
then our LCA lies in left side of the node, if it's is smaller than both n1 and n2, then LCA lies on right side.
Otherwise root is LCA (assuming that both n1 and n2 are present in BST)
"""

# A recursive python program to find LCA of two nodes
# n1 and n2


# Function to find LCA of n1 and n2. The function assumes
# that both n1 and n2 are present in BST
def lowestCommonAncestorBST(root, n1, n2):
    """
    :type root: TreeNode
    :type n1: TreeNode
    :type n2: TreeNode
    :rtype: TreeNode
    """
    # Base Case
    if root is None:
        return None

    if root.val >= n1.val and root.val < n2.val:
        return root

    # If both n1 and n2 are smaller than root, then LCA lies in left
    if root.val > n1.val and root.val > n2.val:
        return lowestCommonAncestorBST(root.left, n1, n2)

    # If both n1 and n2 are greater than root, then LCA lies in right
    if root.val < n1.val and root.val < n2.val:
        return lowestCommonAncestorBST(root.right, n1, n2)

    return root

