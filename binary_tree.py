# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

"""
Binary Tree Right Side View
https://leetcode.com/problems/binary-tree-right-side-view/

Given a binary tree, imagine yourself standing on the right side of it,
return the values of the nodes you can see ordered from top to bottom.

For example:
Given the following binary tree,
   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---
You should return [1, 3, 4].
"""
def rightSideView(self, root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    # DFS-traverse the tree right-to-left, add values to the view whenever we first reach a new record depth. This is O(n).
    # def traverseDFS(node, height):
    #     if node:
    #         if len(res) == height:
    #             res.append(node.val)
    #         traverseDFS(node.right, height + 1)
    #         traverseDFS(node.left, height + 1)
    # res = []
    # traverseDFS(root, 0)
    # return res

    # Traverse the tree level by level and add the last value of each level to the view. This is O(n).
    res = []
    if root:
        level = [root]
        while level:
            res += level[-1].val,
            level = [child for node in level for child in (node.left, node.right) if child]
    return res

import pickle
def serializeBT(root):
    """Encodes a tree to a single string.

    :type root: TreeNode
    :rtype: str
    """

    # return pickle.dumps(root)
    # Recursive preorder
    def preorder(node):
        if node:
            vals.append(node.val)
            preorder(node.left)
            preorder(node.right)
        else:
            vals.append('#')

    vals = []
    preorder(root)
    return ' '.join(map(str, vals))

def deserializeBT(data):
    """Decodes your encoded data to tree.

    :type data: str
    :rtype: TreeNode
    """

    # return pickle.loads(data)
    def deserialize():
        val = next(vals)
        if val == '#':
            return None
        node = TreeNode(int(val))
        node.left = deserialize()
        node.right = deserialize()
        return node

    vals = iter(data.split())
    return deserialize()

"""
Most Frequent Subtree Sum
https://leetcode.com/problems/most-frequent-subtree-sum

Given the root of a tree, you are asked to find the most frequent subtree sum.
The subtree sum of a node is defined as the sum of all the node values formed by the
subtree rooted at that node (including the node itself). So what is the most frequent
subtree sum value? If there is a tie, return all the values with the highest frequency in any order.

Examples 1
Input:

  5
 /  \
2   -3
return [2, -3, 4], since all the values happen only once, return all of them in any order.
Examples 2
Input:

  5
 /  \
2   -5
return [2], since 2 happens twice, however -5 only occur once.
Note: You may assume the sum of values in any subtree is in the range of 32-bit signed integer.
"""

from collections import Counter

def findFrequentTreeSum(root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    def frequentTreeSum(root):
        if not root:
            return (Counter(), 0)

        left, right = frequentTreeSum(root.left), frequentTreeSum(root.right)
        sum = root.val + left[1] + right[1]
        return (left[0] + right[0] + Counter({sum: 1}), sum)

    counter = frequentTreeSum(root)[0]
    max_count = max(counter.values())
    res = [c[0] for c in counter.items() if c[1] == max_count]
    return res

"""
Binary Tree Level Order Traversal
https://leetcode.com/problems/binary-tree-level-order-traversal/
Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its level order traversal as:
[
  [3],
  [9,20],
  [15,7]
]
"""
def levelOrder(root):
    """
    :type root: TreeNode
    :rtype: List[List[int]]
    """
    # recursive
    def _levelOrder(nodes):
        curr_nodes = [node.val for node in nodes]
        children = [leaf for node in nodes for leaf in (node.left, node.right) if leaf]
        return [curr_nodes] + _levelOrder(children) if children else [curr_nodes]

    return _levelOrder([root]) if root else []

    # iterative
    # res, level = [], [root]
    # while root and level:
    #     res.append([node.val for node in level])
    #     level = [child for node in level for child in (node.left, node.right) if child]
    # return res

"""
Find maximum path from root to leaf
"""
def maxSumOfRootToLeafPath(root):
    # recursive
    if not root:
        return 0
    maxLeft = maxRight = 0
    if root.left:
        maxLeft = maxSumOfRootToLeafPath(root.left)
    if root.right:
        maxRight = maxSumOfRootToLeafPath(root.right)
    return root.val + max(maxLeft, maxRight)

"""
Invert Binary Tree (BT)

https://leetcode.com/problems/invert-binary-tree/

Invert a binary tree.

     4
   /   \
  2     7
 / \   / \
1   3 6   9
to
     4
   /   \
  7     2
 / \   / \
9   6 3   1
"""
def invertTree(root):
    """
    :type root: TreeNode
    :rtype: TreeNode
    """
    stack = [root]
    while stack:
        node = stack.pop()
        if node:
            node.left, node.right = node.right, node.left
            stack += node.left, node.right
    return root


"""
Maximum Depth of Binary Tree (BT)
https://leetcode.com/problems/maximum-depth-of-binary-tree/

Given a binary tree, find its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
"""
def maxDepth(root):
    """
    :type root: TreeNode
    :rtype: int
    """
    if not root:
        return 0
    if not root.left and not root.right:
        return 1
    return max(maxDepth(root.left) + 1, maxDepth(root.right) + 1)

"""
Sum Root to Leaf Numbers
Given a binary tree (BT) containing digits from 0-9 only, each root-to-leaf path could represent a number.

An example is the root-to-leaf path 1->2->3 which represents the number 123.

Find the total sum of all root-to-leaf numbers.

For example,

    1
   / \
  2   3

The root-to-leaf path 1->2 represents the number 12.

The root-to-leaf path 1->3 represents the number 13.

Return the sum = 12 + 13 = 25.
"""

from collections import deque
def sumNumbers(root):
    """
    :type root: TreeNode
    :rtype: int
    """
    if not root:
        return 0

    queue, res = deque([(root, root.val)]), 0
    while queue:
        node, val = queue.popleft()
        if node:
            if not node.left and not node.right: # leaf node
                res += val
            if node.left:
                queue.append((node.left, val * 10 + node.left.val))
            if node.right:
                queue.append((node.right, val * 10 + node.right.val))
    return res


"""
Count Complete Tree Nodes
https://leetcode.com/problems/count-complete-tree-nodes/

Given a complete binary tree, count the number of nodes.

Definition of a complete binary tree from Wikipedia:
In a complete binary tree every level, except possibly the last, is completely filled,
and all nodes in the last level are as far left as possible.
It can have between 1 and 2h nodes inclusive at the last level h.
"""

def countNodes(root):
    """
    :type root: TreeNode
    :rtype: int
    """
    # find tree height, by going to the left most node
    def height(root):
        if not root:
            return -1
        h, leftNode = 0, root
        while leftNode.left:
            leftNode = leftNode.left
            h += 1
        return h

    if not root:
        return 0
    res = 0
    h = height(root)
    while root:
        if height(root.right) == h - 1:  # root.left is a perfect tree
            res += 1 << h
            root = root.right
        else:  # root.right is a perfect tree
            res += 1 << (h - 1)
            root = root.left
        h -= 1
    return res


"""
Lowest Common Ancestor in a Binary Tree
http://www.geeksforgeeks.org/lowest-common-ancestor-binary-tree-set-1/
https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/

        _______3______
       /              \
    ___5__          ___1__
   /      \        /      \
   6      _2       0       8
         /  \
         7   4
For example, the lowest common ancestor (LCA) of nodes 5 and 1 is 3.
Another example is LCA of nodes 5 and 4 is 5, since a node can be a
descendant of itself according to the LCA definition.

Method 1 (By Storing root to n1 and root to n2 paths):
Following is simple O(n) algorithm to find LCA of n1 and n2.
1) Find path from root to n1 and store it in a vector or array.
2) Find path from root to n2 and store it in another vector or array.
3) Traverse both paths till the values in arrays are same. Return the common element just before the mismatch.


Method 2 : do not require extra storage
(Using Single Traversal)
The method 1 finds LCA in O(n) time, but requires three tree traversals plus extra spaces for path arrays.
If we assume that the keys n1 and n2 are present in Binary Tree, we can find LCA using single traversal of Binary Tree
and without extra storage for path arrays.
The idea is to traverse the tree starting from root. If any of the given keys (n1 and n2) matches with root,
then root is LCA (assuming that both keys are present). If root doesn't match with any of the keys, we
recur for left and right subtree. The node which has one key present in its left subtree and the other key
present in right subtree is the LCA. If both keys lie in left subtree, then left subtree has LCA also,
otherwise LCA lies in right subtree.
"""

# Python program to find LCA of n1 and n2 using one
# traversal of Binary tree


# This function returns pointer to LCA of two given
# nodes p and q
# This function assumes that p and q are present in
# Binary Tree
def lowestCommonAncestorBT(root, p, q):
    """
    :type root: TreeNode
    :type p: TreeNode
    :type q: TreeNode
    :rtype: TreeNode
    """
    # Base Case
    if root is None:
        return None

    # If either p or q matches with root's key, report
    #  the presence by returning root (Note that if a key is
    #  ancestor of other, then the ancestor key becomes LCA
    if root == p or root == q:
        return root

    # Look for keys in left and right subtrees
    left = lowestCommonAncestorBT(root.left, p, q)
    right = lowestCommonAncestorBT(root.right, p, q)

    # If both of the above calls return Non-NULL, then one key
    # is present in once subtree and other is present in other,
    # So this node is the LCA
    if left and right:
        return root

    # Otherwise check if left subtree or right subtree is LCA
    return left if left else right


"""
Sum of Left Leaves
https://discuss.leetcode.com/category/530/sum-of-left-leaves
"""
def sumOfLeftLeaves(root):
    """
    :type root: TreeNode
    :rtype: int
    """
    def isLeaf(node):
        return False if node.left or node.right else True

    """
    using recursive call
    """
    res = 0
    if not root:  # root is leaf node
        return 0
    if isLeaf(root):
        return root.val
    elif root.left:
        temp = root.left.val if isLeaf(root.left) else sumOfLeftLeaves(root.left)
        res += temp
    if root.right and not isLeaf(root.right):
        res += sumOfLeftLeaves(root.right)

    print (res)

    return res


"""
Flatten Binary Tree to Linked List
https://leetcode.com/problems/flatten-binary-tree-to-linked-list/
Given a binary tree, flatten it to a linked list in-place.

For example,
Given

         1
        / \
       2   5
      / \   \
     3   4   6
The flattened tree should look like:
   1
    \
     2
      \
       3
        \
         4
          \
           5
            \
             6
"""
def flatten(root):
    curr = root
    while curr:
        if curr.left:
            # find current node's prenode that links to current node's right subtree
            pre = curr.left
            while pre.right:
                pre = pre.right
            pre.right = curr.right
            # use current node's left subtree to replace its right subtree
            # original right subtree is already linked by current node's prenode
            curr.right = curr.left
            curr.left = None
        curr = curr.right

    # recursive
    # def _flatten(root):
    #     if not root:
    #         return
    #     _flatten(root.right)
    #     _flatten(root.left)
    #     root.right = prev
    #     root.left = None
    #     prev = root
    # prev = None
    # _flatten(root)

"""
Binary tree preorder traverse
    4
   / \
  2   7
/ \  / \
1  3 6  9
"""

def preorder(root):
    if not root:
        return
    s = [root]  # stack
    while s:
        node = s.pop()
        # print (node.val)
        if node.right:
            s.append(node.right)
        if node.left:
            s.append(node.left)

def inorder(root):
    if not root:
        return
    s = [] # stack
    while s or root:
        if root:
            s.append(root)
            root = root.left
        else:
            root = s.pop()
            # print(root.val)
            root = root.right

def postorder(root):
    s = [] # stack
    last_visited = None
    while s or root:
        if root:
            s.append(root)
            root = root.left
        else:
            peek = s[-1]
            # if right child exists and traversing node from left child, then move right
            if peek.right and last_visited != peek.right:
                root = peek.right
            else:
                last_visited = s.pop()
                print (last_visited.val)

# root = TreeNode(4)
# node2 = TreeNode(2)
# node7 = TreeNode(7)
# node1 = TreeNode(1)
# node3 = TreeNode(3)
# node6 = TreeNode(6)
# node9 = TreeNode(9)
# root.left, root.right = node2, node7
# node2.left, node2.right = node1, node3
# node7.left, node7.right = node6, node9
#
# post_order(root)