import unittest

from numbers import *
from linked_list import *
from binary_search_tree import *
from dynamic_programming import *
from sorting import *
from dictionary import *
from string_manipulation import *
from binary_tree import *
from queue_stack import *
from string_manipulation import *
from numbers import *
from binary_search import *

class TestMethods(unittest.TestCase):

    def setUp(self):
        """
        Let us construct the BST shown in the figure
                20
              /   \
            8      22
          /  \    / \
         4    12 21  25
             /  \
            10  14
        """
        self.root = TreeNode(20)
        self.root.left = TreeNode(8)
        self.root.right = TreeNode(22)
        self.root.right.left = TreeNode(21)
        self.root.right.right = TreeNode(25)
        self.root.left.left = TreeNode(4)
        self.root.left.right = TreeNode(12)
        self.root.left.right.left = TreeNode(10)
        self.root.left.right.right = TreeNode(14)

        """
        Linked list
        1->3->5->7->None
        2->4->None
        """
        self.head1 = ListNode(1)
        self.head1.next = ListNode(3)
        self.head1.next.next = ListNode(5)
        self.head1.next.next.next = ListNode(7)
        self.head2 = ListNode(2)
        self.head2.next = ListNode(4)
        self.head3 = ListNode(1)
        self.head3.next = ListNode(2)
        self.head3.next.next = ListNode(3)
        self.head3.next.next.next = ListNode(4)
        self.head3.next.next.next.next = ListNode(5)
        self.head3.next.next.next.next.next = ListNode(7)

        """
        Linked list with cycle
        """
        self.head_cycle = ListNode(1)
        self.head_cycle.next = ListNode(2)
        self.head_cycle.next.next = ListNode(3)
        self.head_cycle.next.next.next = ListNode(4)
        self.head_cycle.next.next.next.next = ListNode(5)
        self.head_cycle.next.next.next.next.next = self.head_cycle.next

        """
        Intersected linked lists
              1 -> 2
                    \
                     6 -> 7 -> 8
                    /
        3 -> 4 -> 5
        """
        self.headA = ListNode(1)
        self.headA.next = ListNode(2)
        self.headA.next.next = ListNode(6)
        self.headA.next.next.next = ListNode(7)
        self.headA.next.next.next.next = ListNode(8)
        self.headB = ListNode(3)
        self.headB.next = ListNode(4)
        self.headB.next.next = ListNode(5)
        self.headB.next.next.next = self.headA.next.next

        """
        Sorted matrix in both rows and cols in increasing order
        0  1  2  3
        2  3  4  5
        7  8  9  10
        10 11 12 13
        """
        self.sortedM = [[0, 1, 2, 3], [2, 3, 4, 5], [7, 8, 9, 10], [10, 11, 12, 13]]

    """
    Test canJump()
    """
    def test_canJump(self):
        self.assertTrue(canJump([2,3,1,1,4]))
        self.assertFalse(canJump([3, 2, 1, 0, 4]))

    def test_searchRange(self):
        self.assertEqual(searchRange([5, 7, 7, 8, 8, 10], 8), [3,4])

    def test_reverseList(self):
        node1 = ListNode(1)
        node2 = ListNode(2)
        node3 = ListNode(3)
        node1.next, node2.next = node2, node3
        self.assertEqual(node1, reverseList(reverseList(node1)))
        self.assertEqual(node3, reverseList(node1))

    def test_lowestCommonAncestorBST(self):
        self.assertEqual(12, lowestCommonAncestorBST(self.root, TreeNode(10), TreeNode(14)).val)
        self.assertEqual(8, lowestCommonAncestorBST(self.root, TreeNode(14), TreeNode(8)).val)
        self.assertEqual(20, lowestCommonAncestorBST(self.root, TreeNode(10), TreeNode(22)).val)

    def test_twoSum(self):
        self.assertEqual([0,1], twoSum([2, 7, 11, 15], 9))
        self.assertEqual([], twoSum([2, 7, 11, 15], 16))

    def test_repeatedSubstringPattern(self):
        self.assertEqual(True, repeatedSubstringPattern("abab"))
        self.assertEqual(False, repeatedSubstringPattern("aba"))
        self.assertEqual(True, repeatedSubstringPattern("abcabcabcabc"))

    def test_maxSumOfRootToLeafPath(self):
        self.assertEqual(0, maxSumOfRootToLeafPath(None))
        self.assertEqual(1, maxSumOfRootToLeafPath(TreeNode(1)))
        self.assertEqual(67, maxSumOfRootToLeafPath(self.root))
        self.assertEqual(34, maxSumOfRootToLeafPath(self.root.left))
        self.assertEqual(47, maxSumOfRootToLeafPath(self.root.right))

    def test_isValidParentheses(self):
        self.assertEqual(True, isValidParentheses("()"))
        self.assertEqual(True, isValidParentheses("()[]{}"))
        self.assertEqual(False, isValidParentheses("(]"))
        self.assertEqual(False, isValidParentheses("([)]"))

    def test_maxProfit(self):
        self.assertEqual(5, maxProfit([7, 1, 5, 3, 6, 4]))
        self.assertEqual(0, maxProfit([7, 6, 4, 3, 1]))

    def test_firstUniqChar(self):
        self.assertEqual(0, firstUniqChar("leetcode"))
        self.assertEqual(2, firstUniqChar("loveleetcode"))

    def test_twoSumSorted(self):
        self.assertEqual(False, twoSumSorted([], 9))
        self.assertEqual([1,2], twoSumSorted([2, 7, 11, 15], 9))
        self.assertEqual(False, twoSumSorted([2, 7, 11, 15], 10))

    def test_mergeTwoLinkedLists(self):
        self.assertEqual(linkedList2Arr(self.head3), linkedList2Arr(mergeTwoLinkedLists(self.head1, self.head2)))

    def test_mergeTwoSortedLists(self):
        self.assertEqual(linkedList2Arr(None), linkedList2Arr(mergeTwoSortedLists(None, None)))
        self.assertEqual(linkedList2Arr(ListNode(1)), linkedList2Arr(mergeTwoSortedLists(ListNode(1), None)))
        self.assertEqual(linkedList2Arr(self.head3), linkedList2Arr(mergeTwoSortedLists(self.head1, self.head2)))

    def test_hasCycle(self):
        self.assertEqual(True, hasCycle(self.head_cycle))
        self.assertEqual(False, hasCycle(self.head1))

    def test_getIntersectionNode(self):
        self.assertEqual(None, getIntersectionNode(self.headA, self.head1))
        self.assertEqual(6, getIntersectionNode(self.headA, self.headB).val)

    def test_UnionIntersection(self):
        self.assertEqual([], union([], []))
        self.assertEqual([1], union([1], [1]))
        self.assertEqual([1, 2, 3, 4, 5, 6, 7], union([1, 3, 4, 5, 7], [2, 3, 5, 6]))
        self.assertEqual([3, 5], intersection([1, 3, 4, 5, 7], [2, 3, 5, 6]))
        self.assertEqual([3, 5], intersectionBS([1, 3, 4, 5, 7], [2, 3, 5, 6]))

    def test_thirdMax(self):
        self.assertEqual(1, thirdMax([3,2,1]))
        self.assertEqual(2, thirdMax([1,2]))

    def test_isPalindrome(self):
        self.assertEqual(True, isPalindrome(1))
        self.assertEqual(True, isPalindrome(12521))
        self.assertEqual(True, isPalindrome(123454321))
        self.assertEqual(False, isPalindrome(1234521))

    def test_findKthLargestMinHeap(self):
        self.assertEqual(5, findKthLargestMinHeap([3, 2, 1, 5, 6, 4], 2))
        self.assertEqual(6, findKthLargestMinHeap([3, 2, 1, 5, 6, 6, 4], 2))
        self.assertEqual(1, findKthLargestMinHeap([3, 2, 1, 5, 6, 6, 4], 10))
        self.assertEqual(5, findKthLargestMinHeap2([3, 2, 1, 5, 6, 4], 2))
        self.assertEqual(6, findKthLargestMinHeap2([3, 2, 1, 5, 6, 6, 4], 2))
        self.assertEqual(1, findKthLargestMinHeap2([3, 2, 1, 5, 6, 6, 4], 7))

    """
        0  1  2  3
        2  3  4  5
        7  8  9  10
        10 11 12 13
    """
    def test_kthSmallest(self):
        # self.assertEqual(0, kthSmallest(self.sortedM, 1))
        # self.assertEqual(1, kthSmallest(self.sortedM, 2))
        # self.assertEqual(2, kthSmallest(self.sortedM, 3))
        # self.assertEqual(2, kthSmallest(self.sortedM, 4))
        # self.assertEqual(3, kthSmallest(self.sortedM, 5))
        # self.assertEqual(3, kthSmallest(self.sortedM, 6))
        self.assertEqual(4, kthSmallest(self.sortedM, 7))

    def test_sortedMatrix2SortedArr(self):
        sortedArr = [0,1,2,2,3,3,4,5,7,8,9,10,10,11,12,13]
        self.assertEqual(sortedArr, sortedMatrix2SortedArr(self.sortedM))

    def test_searchMatrix(self):
        self.assertEqual(False, searchMatrix(self.sortedM, 6))
        self.assertEqual(True, searchMatrix(self.sortedM, 10))

    def test_findFrequentTreeSum(self):
        """
          5
         /  \
        2   -3
        """
        root = TreeNode(5)
        root.left = TreeNode(2)
        root.right = TreeNode(-3)
        self.assertEqual([2,-3,4], findFrequentTreeSum(root))

        root = TreeNode(5)
        root.left = TreeNode(2)
        root.right = TreeNode(-5)
        self.assertEqual([2], findFrequentTreeSum(root))

    def test_isValidBST(self):
        self.assertEqual(True, isValidBST(None))
        self.assertEqual(True, isValidBST(self.root))

    # def test_sortedArrayToBST(self):
    #     self.assertEqual(self.root, sortedArrayToBST([4, 8, 10, 12, 14, 20, 21, 22, 25]))

    def test_serializeBST(self):
        self.assertEqual('20,8,4,12,10,14,22,21,25', serializeBST(self.root))
        self.assertEqual(serializeBST(self.root), serializeBST(deserializeBST('20,8,4,12,10,14,22,21,25')))

    def test_serializeBT(self):
        self.assertEqual('20 8 4 # # 12 10 # # 14 # # 22 21 # # 25 # #', serializeBT(self.root))
        self.assertEqual('20 8 4 # # 12 10 # # 14 # # 22 21 # # 25 # #',
                         serializeBT(deserializeBT('20 8 4 # # 12 10 # # 14 # # 22 21 # # 25 # #')))

    def test_lengthOfLongestSubstring(self):
        self.assertEqual(3, lengthOfLongestSubstring("abcabcbb"))
        self.assertEqual(1, lengthOfLongestSubstring("bbbbb"))
        self.assertEqual(3, lengthOfLongestSubstring("pwwkew"))

    def test_findGrantsCap(self):
        self.assertEqual(3.3333333333333333, findGrantsCap([20, 30, 40], 10))
        self.assertEqual(3.3333333333333333, findGrantsCap2([20, 30, 40], 10))
        self.assertEqual(2.5, findGrantsCap([10,20,30,40], 10))
        self.assertEqual(2.5, findGrantsCap2([10, 20, 30, 40], 10))
        self.assertEqual(2.3333333333333335, findGrantsCap([1, 2, 3, 4, 5], 10))
        self.assertEqual(2.3333333333333335, findGrantsCap2([1,2,3,4,5], 10))
        self.assertEqual(4, findGrantsCap([1, 2, 3, 4, 5, 6, 7, 8, 9], 30))
        self.assertEqual(4, findGrantsCap2([1, 2, 3, 4, 5, 6, 7, 8, 9], 30))

    def test_findSuccessorBST(self):
        """
                20
              /   \
            8      22
          /  \    / \
         4    12 21  25
             /  \
            10  14
        """
        self.assertEqual(self.root.left.right, findSuccessorBST(self.root, self.root.left.right.left))
        self.assertEqual(None, findSuccessorBST(self.root, self.root))
        self.assertEqual(self.root.left.right, findSuccessorBST_iter(self.root, self.root.left.right.left))
        self.assertEqual(None, findSuccessorBST_iter(self.root, self.root))

    def test_largestSmallerKey(self):
        self.assertEqual(14, largestSmallerKey(self.root, 16))
        self.assertEqual(None, largestSmallerKey(self.root, 4))
        self.assertEqual(25, largestSmallerKey(self.root, 30))
        self.assertEqual(20, largestSmallerKey(self.root, 21))

    def test_findSumPairs(self):
        self.assertEqual((10,14), findSumPairs(self.root, 24))
        self.assertEqual((4, 21), findSumPairs(self.root, 25))

        self.assertEqual((4, 20), findSumPairsBi(self.root, 24))
        self.assertEqual((4, 21), findSumPairsBi(self.root, 25))

    def test_findFrequentTreeSum(self):
        self.assertEqual([4, 10, 14, 36, 48, 21, 25, 68, 136], findFrequentTreeSum(self.root))

    def test_levelOrder(self):
        self.assertEqual([[20], [8, 22], [4, 12, 21, 25], [10, 14]], levelOrder(self.root))

    """
            20
          /   \
        8      22
      /  \    / \
     4    12 21  25
         /  \
        10  14
    """
    def test_maxSumOfRootToLeafPath(self):
        self.assertEqual(67, maxSumOfRootToLeafPath(self.root))

    def test_maxDepth(self):
        self.assertEqual(4, maxDepth(self.root))

    def test_sumNumbers(self):
        self.assertEqual(48434, sumNumbers(self.root))

    # def test_countNodes(self):
    #     self.assertEqual(1, countNodes(self.root))

    def test_sumOfLeftLeaves(self):
        self.assertEqual(35, sumOfLeftLeaves(self.root))

    def test_flatten(self):
        flatten(self.root)
        self.assertEqual('20 # 8 # 4 # 12 # 10 # 14 # 22 # 21 # 25 # #', serializeBT(self.root))

    def test_findAnagrams(self):
        self.assertEqual([0, 1, 2], findAnagrams("abab", "ab"))

    def test_firstUniqChar(self):
        self.assertEqual(0, firstUniqChar("leetcode"))
        self.assertEqual(2, firstUniqChar("loveleetcode"))

    def test_reverseString(self):
        self.assertEqual(" apple  ", reverseString("  apple "))
        self.assertEqual("hien trong to", reverseString("to trong hien"))

    def test_minWindow(self):
        self.assertEqual("BANC", minWindow("ADOBECODEBANC", "ABC"))
        self.assertEqual("zyx", minWindow("xyyzyzyx", "xyz"))

    def test_isOrdered(self):
        self.assertEqual(False, isOrdered("hello world!", "!od"))
        self.assertEqual(True, isOrdered("hello world!", "he!"))

    def test_repeatedSubstringPattern(self):
        self.assertEqual(True, repeatedSubstringPattern("abab"))
        self.assertEqual(False, repeatedSubstringPattern("aba"))
        self.assertEqual(True, repeatedSubstringPattern("abcabcabcabc"))

    def test_longestPalindrome(self):
        self.assertEqual("bab", longestPalindrome("babad"))
        self.assertEqual("bb", longestPalindrome("cbbd"))

    def test_findDuplicate(self):
        self.assertEqual(4, findDuplicate([1,2,3,4,4]))
        self.assertEqual(2, findDuplicate([1, 2, 2, 3, 4]))

    def test_kClosestPoints(self):
        arr = [[1, 2], [4, 5], [3, 1], [5, 5], [5, 1], [-1, -2], [-1, -3]]
        self.assertEqual([[1, 2], [-1, -2], [3, 1], [-1, -3]], kClosestPoints(arr, 4))

    def test_mergeIntervals(self):
        self.assertEqual([[1,6], [8,12], [15,18]], mergeIntervals([[15,18], [2,6], [9,12], [8,10], [1,3]]))

    def test_isValidSchedule(self):
        self.assertEqual(True, isValidSchedule([[15, 18], [2, 6], [9, 12]]))

    def test_searchRotatedSortedArr(self):
        self.assertEqual(0, searchRotatedSortedArr([4, 5, 6, 7, 0, 1, 2], 4))
        self.assertEqual(1, searchRotatedSortedArr([4, 5, 6, 7, 0, 1, 2], 5))
        self.assertEqual(5, searchRotatedSortedArr([4, 5, 6, 7, 0, 1, 2], 1))

    def test_twoSumSorted(self):
        self.assertEqual([1,2], twoSumSorted([2, 7, 11, 15], 9))

"""
def repeatedSubstringPattern(str):
    """
if __name__ == '__main__':
    unittest.main()