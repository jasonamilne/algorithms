# Analyzing and Designing Algorithms
## Algorithm

Sorting problem:
- input: <$a_1,a_2,\ldots,a_n$> $a_i \in Z$
- output: <$a'_1,a'_2,\ldots,a'_n$> $a_i \in Z$ such that $a'_1 \le a'_2 \le \ldots \le a'_n$ (sorted permutation)

Algorithm selection:
```
INSERTION-SORT(A)
1  for j = 2 to A.length
2      key = A[j]
3      // Insert A[j] into the sorted sequence A[1..j-1]
4      i = j - 1
5      while i > 0 and A[i] > key
6          A[i + 1] = A[i]
7          i = i - 1
8      A[i + 1] = key
```

Loop invariant:

1. Initialization - check invariant before loop starts

2. Maintenance - show invariant is preserved each iteration

3. Termination - use invariant to prove correctness

Psuedocode conventions:



## Analyzing Algorithms


## Designing Algorithms