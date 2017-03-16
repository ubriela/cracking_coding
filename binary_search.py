"""
Award Budget Cuts
https://www.pramp.com/question/r1Kw0vwG6OhK9AEGAyWV

The awards committee had planned to give n research grants this year, out of a its total yearly budget.
However, the budget was reduced to b dollars. The committee members has decided to affect the minimal number of highest grants, by applying a maximum cap c on all grants: every grant that was planned to be higher than c will now be c dollars.
Help the committee to choose the right value of c that would make the total sum of grants equal to the new budget.

Given an array of grants g and a new budget b, explain and code an efficient method to find the cap c. Assume that each grant is unique.
Analyze the time and space complexity of your solution.
"""
def cappedSum(g, partial_sum, i):
    return partial_sum[i - 1] + g[i] * (len(g) - i)

from itertools import accumulate
def findGrantsCap(g, b):
    if not g or len(g) == 0:
        return 0

    # precompute partial sum
    accum_sums = list(accumulate(g))

    if accum_sums[len(g) - 1] <= b: return g[-1]

    start, end = 0, len(g) - 1
    while start < end:
        mid = (start + end)/2
        if mid > 0:
            if cappedSum(g, accum_sums, int(mid)) > b:
                if cappedSum(g, accum_sums, int(mid) - 1):
                    break
                else:
                    end = mid - 1
            else:
                start = mid + 1
    if b <= accum_sums[int(mid) - 1]:
        return b/len(g)
    c = (b - accum_sums[int(mid) - 1]) / (len(g) - int(mid))
    return c

"""
This is a simpler implementation.
O(n) time in the worst case, O(1) time in the best case.
"""
def findGrantsCap2(g, b):
    if not g or len(g) == 0:
        return 0
    l = len(g)
    accum_sum = 0
    for i, grant in enumerate(g):
        if accum_sum + grant * (l - i) > b:
            break
        accum_sum += grant
    return (b - accum_sum) / (l - i)