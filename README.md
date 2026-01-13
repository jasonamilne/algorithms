# Algorithms

Documentation for the algorithms project.

Exploration and application of algorithms in Python.

---

## Table of Contents

- [Sorting Algorithms](#sorting-algorithms)
- [Searching Algorithms](#searching-algorithms)
- [Graph Algorithms](#graph-algorithms)
- [Tree Algorithms](#tree-algorithms)
- [Dynamic Programming](#dynamic-programming)
- [Greedy Algorithms](#greedy-algorithms)
- [Divide and Conquer](#divide-and-conquer)
- [Backtracking Algorithms](#backtracking-algorithms)
- [String Algorithms](#string-algorithms)
- [Mathematical Algorithms](#mathematical-algorithms)
- [Number Theory Algorithms](#number-theory-algorithms)
- [Computational Geometry](#computational-geometry)
- [Data Structures](#data-structures)
- [Hashing Algorithms](#hashing-algorithms)
- [Randomized Algorithms](#randomized-algorithms)
- [Network Flow Algorithms](#network-flow-algorithms)
- [Linear Programming](#linear-programming)
- [Approximation Algorithms](#approximation-algorithms)
- [Cryptographic Algorithms](#cryptographic-algorithms)
- [Compression Algorithms](#compression-algorithms)
- [Numerical Algorithms](#numerical-algorithms)
- [Machine Learning Algorithms](#machine-learning-algorithms)
- [Parallel and Distributed Algorithms](#parallel-and-distributed-algorithms)
- [Miscellaneous Algorithms](#miscellaneous-algorithms)

---

## Sorting Algorithms

### Comparison-Based Sorting

| Algorithm | Time (Best) | Time (Avg) | Time (Worst) | Space | Stable |
|-----------|-------------|------------|--------------|-------|--------|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) | No |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | No |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | No |
| Tim Sort | O(n) | O(n log n) | O(n log n) | O(n) | Yes |
| Shell Sort | O(n log n) | O(n^1.25) | O(n²) | O(1) | No |
| Tree Sort | O(n log n) | O(n log n) | O(n²) | O(n) | Yes |
| Cocktail Shaker Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Comb Sort | O(n log n) | O(n²) | O(n²) | O(1) | No |
| Gnome Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Odd-Even Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Strand Sort | O(n) | O(n²) | O(n²) | O(n) | Yes |
| Pancake Sort | O(n) | O(n²) | O(n²) | O(1) | No |
| Cycle Sort | O(n²) | O(n²) | O(n²) | O(1) | No |
| Library Sort | O(n) | O(n log n) | O(n²) | O(n) | Yes |
| Patience Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
| Smooth Sort | O(n) | O(n log n) | O(n log n) | O(1) | No |
| Tournament Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | No |
| Block Sort | O(n) | O(n log n) | O(n log n) | O(1) | Yes |
| Intro Sort | O(n log n) | O(n log n) | O(n log n) | O(log n) | No |

### Non-Comparison Sorting

| Algorithm | Time (Best) | Time (Avg) | Time (Worst) | Space | Stable |
|-----------|-------------|------------|--------------|-------|--------|
| Counting Sort | O(n + k) | O(n + k) | O(n + k) | O(k) | Yes |
| Radix Sort | O(nk) | O(nk) | O(nk) | O(n + k) | Yes |
| Bucket Sort | O(n + k) | O(n + k) | O(n²) | O(n) | Yes |
| Pigeonhole Sort | O(n + k) | O(n + k) | O(n + k) | O(k) | Yes |
| Flash Sort | O(n) | O(n) | O(n²) | O(n) | No |
| Bead Sort | O(n) | O(S) | O(S) | O(n²) | N/A |
| Burst Sort | O(n) | O(n log n) | O(n log n) | O(n) | Yes |
| American Flag Sort | O(n·k/d) | O(n·k/d) | O(n²) | O(1) | No |

### External Sorting

- External Merge Sort
- Polyphase Merge Sort
- Replacement Selection Sort
- Distribution Sort

### Hybrid Sorting

- Timsort (Merge + Insertion)
- Introsort (Quick + Heap + Insertion)
- Cubesort
- Spreadsort

---

## Searching Algorithms

### Linear Search Variants

- Linear Search
- Sentinel Linear Search
- Meta Binary Search
- Ternary Search
- Jump Search
- Exponential Search
- Fibonacci Search
- Interpolation Search

### Binary Search Variants

- Binary Search (Iterative)
- Binary Search (Recursive)
- Lower Bound Search
- Upper Bound Search
- Binary Search on Answer
- Fractional Cascading

### Search in Specialized Structures

- Search in Sorted Matrix (Row-Column Sorted)
- Search in Rotated Sorted Array
- Search in Nearly Sorted Array
- Search in Infinite Array
- Peak Element Finding
- Local Minima/Maxima Finding

### Sublist/Subsequence Search

- Two Pointer Technique
- Sliding Window Search
- Maximum/Minimum Subarray (Kadane's Algorithm)
- Longest Increasing Subsequence Search

---

## Graph Algorithms

### Graph Traversal

- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Iterative Deepening DFS (IDDFS)
- Bidirectional Search
- Lexicographic BFS
- Random Walk

### Shortest Path Algorithms

| Algorithm | Use Case | Time Complexity |
|-----------|----------|-----------------|
| Dijkstra's Algorithm | Single-source, non-negative weights | O((V + E) log V) |
| Bellman-Ford Algorithm | Single-source, negative weights | O(VE) |
| Floyd-Warshall Algorithm | All-pairs shortest paths | O(V³) |
| Johnson's Algorithm | All-pairs, sparse graphs | O(V² log V + VE) |
| A* Search Algorithm | Heuristic-based pathfinding | O(E) |
| SPFA (Shortest Path Faster Algorithm) | Single-source, optimized Bellman-Ford | O(VE) |
| D* (Dynamic A*) | Replanning in dynamic environments | Varies |
| Bidirectional Dijkstra | Point-to-point queries | O((V + E) log V) |
| Contraction Hierarchies | Preprocessed road networks | O(E log V) |
| Hub Labeling | Preprocessed shortest paths | O(k) query |
| Viterbi Algorithm | Shortest path in HMM/DAG | O(VE) |

### Minimum Spanning Tree

- Prim's Algorithm
- Kruskal's Algorithm
- Borůvka's Algorithm
- Reverse-Delete Algorithm
- Minimum Spanning Arborescence (Edmonds' Algorithm)

### Cycle Detection

- DFS-based Cycle Detection (Directed)
- DFS-based Cycle Detection (Undirected)
- Floyd's Cycle Detection (Tortoise and Hare)
- Brent's Cycle Detection
- Union-Find Cycle Detection

### Topological Sorting

- Kahn's Algorithm (BFS-based)
- DFS-based Topological Sort
- Lexicographically Smallest Topological Order

### Strongly Connected Components

- Kosaraju's Algorithm
- Tarjan's Algorithm
- Path-Based Strong Component Algorithm
- Gabow's Algorithm

### Biconnected Components and Bridges

- Articulation Points Detection
- Bridge Detection
- Biconnected Components Decomposition
- 2-Edge-Connected Components

### Graph Coloring

- Greedy Coloring
- Welsh-Powell Algorithm
- DSatur Algorithm
- Backtracking Graph Coloring
- Chromatic Number Approximation

### Matching Algorithms

- Hungarian Algorithm (Kuhn-Munkres)
- Hopcroft-Karp Algorithm
- Edmonds' Blossom Algorithm
- Stable Marriage (Gale-Shapley)
- Maximum Bipartite Matching

### Eulerian and Hamiltonian Paths

- Hierholzer's Algorithm (Eulerian Path/Circuit)
- Fleury's Algorithm
- Hamiltonian Path/Cycle (Backtracking)
- Hamiltonian Path (Dynamic Programming)

### Planarity Testing

- Kuratowski's Theorem
- Boyer-Myrvold Algorithm
- Left-Right Planarity Test

### Other Graph Algorithms

- Graph Isomorphism (VF2, Weisfeiler-Lehman)
- Transitive Closure
- Transitive Reduction
- Dominator Trees (Lengauer-Tarjan)
- Lowest Common Ancestor (LCA)
- Heavy-Light Decomposition
- Centroid Decomposition
- Tree Isomorphism

---

## Tree Algorithms

### Tree Traversal

- Preorder Traversal (Recursive/Iterative)
- Inorder Traversal (Recursive/Iterative)
- Postorder Traversal (Recursive/Iterative)
- Level Order Traversal
- Morris Traversal (Threaded Trees)
- Vertical Order Traversal
- Diagonal Traversal
- Boundary Traversal
- Zigzag Level Order Traversal

### Binary Search Tree Operations

- Search in BST
- Insertion in BST
- Deletion in BST
- Inorder Successor/Predecessor
- Floor and Ceiling
- Kth Smallest/Largest Element
- BST to Sorted Array
- Sorted Array to Balanced BST

### Tree Construction

- Construct Tree from Inorder and Preorder
- Construct Tree from Inorder and Postorder
- Construct BST from Preorder
- Serialize and Deserialize Binary Tree

### Tree Properties and Queries

- Height/Depth of Tree
- Diameter of Tree
- Check if Balanced
- Check if BST
- Check if Symmetric
- Check if Identical Trees
- Check if Subtree
- Maximum Path Sum
- Sum of All Nodes
- Count Nodes in Complete Binary Tree

### Lowest Common Ancestor

- LCA using Parent Pointers
- LCA using Euler Tour + RMQ
- Binary Lifting for LCA
- Tarjan's Offline LCA

### Advanced Tree Algorithms

- Heavy-Light Decomposition
- Centroid Decomposition
- Tree Flattening (Euler Tour)
- Small-to-Large Merging
- DSU on Tree
- Tree Hashing (Isomorphism)
- Virtual Tree Construction
- Link-Cut Trees

---

## Dynamic Programming

### Classical DP Problems

- Fibonacci Numbers
- Factorial
- Binomial Coefficients (Pascal's Triangle)
- Catalan Numbers
- Derangements

### Sequence DP

- Longest Increasing Subsequence (LIS)
- Longest Decreasing Subsequence
- Longest Common Subsequence (LCS)
- Longest Common Substring
- Longest Palindromic Subsequence
- Longest Palindromic Substring
- Shortest Common Supersequence
- Edit Distance (Levenshtein)
- Distinct Subsequences
- Interleaving Strings

### Knapsack Problems

- 0/1 Knapsack
- Unbounded Knapsack
- Bounded Knapsack
- Fractional Knapsack
- Subset Sum Problem
- Partition Equal Subset Sum
- Minimum Subset Sum Difference
- Count of Subset Sum
- Target Sum with +/-

### Matrix/Grid DP

- Matrix Chain Multiplication
- Minimum Path Sum
- Unique Paths
- Maximum Square in Binary Matrix
- Maximal Rectangle
- Dungeon Game
- Cherry Pickup
- Gold Mine Problem
- Egg Dropping Problem

### String DP

- Word Break Problem
- Word Break II
- Regular Expression Matching
- Wildcard Pattern Matching
- Distinct Palindromic Substrings
- Count Palindromic Subsequences
- Minimum Insertions for Palindrome
- Minimum Deletions for Palindrome

### Interval DP

- Optimal Binary Search Tree
- Burst Balloons
- Minimum Cost to Merge Stones
- Boolean Parenthesization
- Palindrome Partitioning

### Tree DP

- Maximum Path Sum in Tree
- Binary Tree Maximum Path Sum
- House Robber III
- Diameter of Binary Tree
- All Longest Paths in Tree

### Bitmask DP

- Traveling Salesman Problem (TSP)
- Assignment Problem
- Hamiltonian Path
- Counting Subsets with Bitmask
- SOS DP (Sum over Subsets)
- Broken Profile DP

### Digit DP

- Count Numbers with Given Digit Sum
- Count Numbers with No Consecutive 1s
- Numbers with Non-Decreasing Digits
- Count Numbers Divisible by K

### Probability/Expected Value DP

- Expected Number of Coin Tosses
- Dice Throw Problem
- Probability of Reaching Target

### Game Theory DP

- Nim Game
- Stone Game
- Optimal Game Strategy
- Grundy Numbers
- Sprague-Grundy Theorem

### DP Optimizations

- Convex Hull Trick
- Divide and Conquer Optimization
- Knuth's Optimization
- Space Optimization (Rolling Array)
- Slope Trick

---

## Greedy Algorithms

### Interval Scheduling

- Activity Selection Problem
- Interval Scheduling Maximization
- Minimum Number of Platforms
- Meeting Rooms Problem
- Merge Intervals
- Insert Interval

### Job Scheduling

- Job Sequencing with Deadlines
- Weighted Job Scheduling
- Minimum Lateness Scheduling
- Shortest Job First
- Longest Job First

### Huffman Coding

- Huffman Encoding
- Huffman Decoding
- Adaptive Huffman Coding

### Greedy on Graphs

- Prim's Algorithm (MST)
- Kruskal's Algorithm (MST)
- Dijkstra's Algorithm

### Other Greedy Problems

- Fractional Knapsack
- Egyptian Fraction
- Minimum Coins for Change
- Maximum Product Subset
- Minimum Product Subset
- Maximum Sum with No Adjacent Elements
- Gas Station Problem
- Jump Game
- Candy Distribution
- Task Scheduler
- Reorganize String
- Queue Reconstruction by Height
- Non-Overlapping Intervals
- Minimum Arrows to Burst Balloons

---

## Divide and Conquer

### Sorting-Based

- Merge Sort
- Quick Sort
- Randomized Quick Sort

### Search-Based

- Binary Search
- Ternary Search
- Median of Medians

### Computational Geometry

- Closest Pair of Points
- Convex Hull (Divide and Conquer)
- Line Intersection

### Matrix Operations

- Strassen's Matrix Multiplication
- Karatsuba Multiplication
- Toom-Cook Multiplication

### Other Divide and Conquer

- Maximum Subarray (Divide and Conquer)
- Inversion Count
- Skyline Problem
- Count of Smaller Numbers After Self
- Closest Points in 3D
- Power of Element (Fast Exponentiation)

---

## Backtracking Algorithms

### Combinatorial Generation

- Generate All Permutations
- Generate All Combinations
- Generate All Subsets (Power Set)
- Generate Parentheses
- Letter Combinations of Phone Number
- Combination Sum (I, II, III, IV)

### Constraint Satisfaction

- N-Queens Problem
- Sudoku Solver
- Crossword Puzzle Solver
- Cryptarithmetic Puzzles
- Graph Coloring
- Map Coloring

### Path Finding

- Rat in a Maze
- Knight's Tour
- Hamiltonian Cycle
- Longest Path in Graph
- Word Search in Grid
- Word Search II (Trie + Backtracking)

### String Problems

- Palindrome Partitioning
- Restore IP Addresses
- Word Ladder (BFS + Backtracking)
- Expression Add Operators

### Other Backtracking

- Subset with Given Sum
- Partition to K Equal Sum Subsets
- Tug of War
- M-Coloring Problem
- Remove Invalid Parentheses

---

## String Algorithms

### Pattern Matching

| Algorithm | Preprocessing | Search Time | Description |
|-----------|---------------|-------------|-------------|
| Naive Pattern Search | O(1) | O(nm) | Brute force approach |
| KMP (Knuth-Morris-Pratt) | O(m) | O(n) | Failure function based |
| Rabin-Karp | O(m) | O(n) avg | Hash-based matching |
| Boyer-Moore | O(m + σ) | O(n/m) avg | Bad character + good suffix |
| Z Algorithm | O(n + m) | O(n) | Z-array based |
| Aho-Corasick | O(Σm) | O(n + z) | Multi-pattern matching |
| Commentz-Walter | O(Σm) | O(n) avg | Boyer-Moore for multiple patterns |
| Shift-And/Shift-Or | O(m) | O(n) | Bitwise pattern matching |

### Suffix Structures

- Suffix Array
- Suffix Array (DC3/Skew Algorithm)
- LCP Array (Kasai's Algorithm)
- Suffix Tree (Ukkonen's Algorithm)
- Suffix Automaton
- Suffix Links

### Trie Data Structures

- Standard Trie
- Compressed Trie (Radix Tree)
- Ternary Search Trie
- Burst Trie
- HAT-Trie
- Generalized Suffix Tree

### String Hashing

- Polynomial Rolling Hash
- Double Hashing
- Cyclic Polynomial Hash
- Substring Hash Queries

### Palindrome Algorithms

- Manacher's Algorithm
- Longest Palindromic Substring (DP)
- Palindrome Partitioning
- Count Palindromic Substrings

### String Matching Metrics

- Levenshtein Distance (Edit Distance)
- Hamming Distance
- Jaro-Winkler Distance
- Longest Common Subsequence
- Longest Common Substring
- Damerau-Levenshtein Distance

### Other String Algorithms

- Longest Repeated Substring
- Longest Unique Substring
- Smallest Rotation (Booth's Algorithm)
- String Compression (Run-Length)
- Burrows-Wheeler Transform
- Inverse Burrows-Wheeler Transform
- Lyndon Factorization
- Minimum Expression (Duval's Algorithm)
- Lexicographically Minimal String Rotation

---

## Mathematical Algorithms

### Basic Arithmetic

- Addition (Arbitrary Precision)
- Subtraction (Arbitrary Precision)
- Karatsuba Multiplication
- Long Division
- Fast Exponentiation (Binary Exponentiation)
- Modular Exponentiation
- Integer Square Root

### GCD and LCM

- Euclidean Algorithm
- Extended Euclidean Algorithm
- Binary GCD (Stein's Algorithm)
- LCM Calculation

### Primality and Factorization

- Trial Division
- Sieve of Eratosthenes
- Segmented Sieve
- Sieve of Atkin
- Miller-Rabin Primality Test
- Fermat Primality Test
- AKS Primality Test
- Pollard's Rho Algorithm
- Pollard's p-1 Algorithm
- Quadratic Sieve
- General Number Field Sieve (conceptual)
- Fermat's Factorization
- Wheel Factorization

### Modular Arithmetic

- Modular Inverse (Extended Euclidean)
- Modular Inverse (Fermat's Little Theorem)
- Chinese Remainder Theorem
- Discrete Logarithm (Baby-Step Giant-Step)
- Primitive Root Finding
- Quadratic Residues
- Legendre/Jacobi Symbol
- Tonelli-Shanks Algorithm

### Combinatorics

- Factorial (Iterative/Recursive)
- Permutations Count
- Combinations (nCr)
- Pascal's Triangle
- Catalan Numbers
- Stirling Numbers (First/Second Kind)
- Bell Numbers
- Derangements
- Partition Numbers
- Lucas' Theorem
- Stars and Bars

### Matrix Algorithms

- Matrix Addition/Subtraction
- Matrix Multiplication (Naive)
- Strassen's Matrix Multiplication
- Matrix Exponentiation
- Gaussian Elimination
- Gauss-Jordan Elimination
- LU Decomposition
- Cholesky Decomposition
- QR Decomposition
- Singular Value Decomposition
- Eigenvalue Decomposition
- Determinant Calculation
- Matrix Inverse
- Matrix Rank

### Polynomial Algorithms

- Polynomial Evaluation (Horner's Method)
- Polynomial Multiplication (Naive)
- FFT (Fast Fourier Transform)
- NTT (Number Theoretic Transform)
- Polynomial Division
- Polynomial GCD
- Polynomial Interpolation (Lagrange)
- Newton's Interpolation

---

## Number Theory Algorithms

### Divisibility

- Count Divisors
- Sum of Divisors
- Euler's Totient Function (Phi)
- Möbius Function
- Divisor Function
- Perfect Numbers Check
- Abundant/Deficient Numbers

### Special Numbers

- Fibonacci Numbers (Matrix Exponentiation)
- Lucas Numbers
- Tribonacci Numbers
- Pell Numbers
- Catalan Numbers
- Harmonic Numbers
- Bernoulli Numbers

### Diophantine Equations

- Linear Diophantine Equation
- Chicken McNugget Theorem
- Pell's Equation
- Pythagorean Triples Generation

### Continued Fractions

- Simple Continued Fraction
- Convergents Calculation
- Square Root via Continued Fractions

### Advanced Number Theory

- Quadratic Sieve Factorization
- Elliptic Curve Factorization
- Sum of Two Squares
- Four Square Theorem
- Partition Function

---

## Computational Geometry

### Basic Operations

- Point in Polygon Test
- Line Intersection
- Line Segment Intersection
- Orientation of Points
- Distance Between Points
- Distance Point to Line
- Distance Point to Segment
- Area of Polygon
- Centroid of Polygon

### Convex Hull Algorithms

- Graham Scan
- Jarvis March (Gift Wrapping)
- Chan's Algorithm
- QuickHull
- Andrew's Monotone Chain
- Divide and Conquer Convex Hull

### Closest Pair

- Closest Pair of Points (Divide and Conquer)
- Closest Pair (Sweep Line)

### Line Sweep Algorithms

- Line Segment Intersection (Bentley-Ottmann)
- Rectangle Union Area
- Maximal Points
- Skyline Problem

### Voronoi Diagrams

- Fortune's Algorithm
- Delaunay Triangulation
- Voronoi via Delaunay

### Triangulation

- Ear Clipping Triangulation
- Delaunay Triangulation
- Polygon Triangulation
- Constrained Delaunay

### Range Searching

- Range Tree
- KD-Tree
- Quadtree
- R-Tree
- Interval Tree
- Segment Tree (2D)

### Other Geometry

- Rotating Calipers
- Half-Plane Intersection
- Polygon Clipping (Sutherland-Hodgman)
- Minkowski Sum
- Boolean Operations on Polygons
- Visibility Polygon
- Art Gallery Problem
- Smallest Enclosing Circle

---

## Data Structures

### Linear Data Structures

- Array
- Dynamic Array (Vector)
- Linked List (Singly, Doubly, Circular)
- Stack
- Queue
- Deque (Double-Ended Queue)
- Priority Queue

### Tree Data Structures

- Binary Tree
- Binary Search Tree (BST)
- AVL Tree
- Red-Black Tree
- Splay Tree
- Treap
- B-Tree
- B+ Tree
- 2-3 Tree
- 2-3-4 Tree
- AA Tree
- Scapegoat Tree
- Weight-Balanced Tree

### Heap Structures

- Binary Heap (Min/Max)
- Binomial Heap
- Fibonacci Heap
- Pairing Heap
- Leftist Heap
- Skew Heap
- D-ary Heap
- Brodal Queue

### Trie Structures

- Trie (Prefix Tree)
- Radix Tree (Compressed Trie)
- Ternary Search Tree
- Suffix Tree
- Suffix Array
- Suffix Automaton

### Hash Structures

- Hash Table
- Hash Set
- Bloom Filter
- Cuckoo Filter
- Count-Min Sketch
- HyperLogLog

### Disjoint Set

- Union-Find (Quick Find)
- Union-Find (Quick Union)
- Union-Find (Weighted + Path Compression)
- Link-Cut Trees

### Range Query Structures

- Segment Tree
- Lazy Propagation Segment Tree
- Persistent Segment Tree
- 2D Segment Tree
- Merge Sort Tree
- Fenwick Tree (Binary Indexed Tree)
- 2D Fenwick Tree
- Sparse Table
- sqrt Decomposition

### Advanced Structures

- Skip List
- Rope
- Gap Buffer
- Piece Table
- Interval Tree
- Range Tree
- KD-Tree
- Quadtree
- Octree
- R-Tree
- Van Emde Boas Tree
- X-Fast Trie
- Y-Fast Trie
- Fusion Tree

### Graph Representations

- Adjacency Matrix
- Adjacency List
- Edge List
- Incidence Matrix

---

## Hashing Algorithms

### Hash Functions

- Division Method
- Multiplication Method
- Universal Hashing
- Perfect Hashing
- Cuckoo Hashing
- Hopscotch Hashing
- Robin Hood Hashing
- Consistent Hashing

### Collision Resolution

- Separate Chaining
- Linear Probing
- Quadratic Probing
- Double Hashing

### String Hashing

- Polynomial Rolling Hash
- Rabin Fingerprint
- FNV Hash
- djb2 Hash
- MurmurHash
- xxHash
- CityHash

### Cryptographic Hash Functions

- MD5
- SHA-1
- SHA-256
- SHA-3
- BLAKE2
- Argon2

### Locality-Sensitive Hashing

- MinHash
- SimHash
- LSH for Euclidean Distance
- LSH for Cosine Similarity

---

## Randomized Algorithms

### Monte Carlo Algorithms

- Monte Carlo Integration
- Monte Carlo Primality Test
- Randomized Min-Cut (Karger's Algorithm)
- Approximate Counting

### Las Vegas Algorithms

- Randomized QuickSort
- Randomized QuickSelect
- Randomized Binary Search Tree

### Sampling Algorithms

- Reservoir Sampling
- Weighted Reservoir Sampling
- Rejection Sampling
- Importance Sampling
- Fisher-Yates Shuffle
- Random Permutation
- Random Subset Selection

### Probabilistic Data Structures

- Bloom Filter
- Count-Min Sketch
- HyperLogLog
- Skip List
- Treap

### Random Number Generation

- Linear Congruential Generator
- Mersenne Twister
- Xorshift
- PCG (Permuted Congruential Generator)

---

## Network Flow Algorithms

### Maximum Flow

| Algorithm | Time Complexity | Description |
|-----------|-----------------|-------------|
| Ford-Fulkerson | O(E × max_flow) | Augmenting path method |
| Edmonds-Karp | O(VE²) | BFS-based Ford-Fulkerson |
| Dinic's Algorithm | O(V²E) | Level graph + blocking flow |
| Push-Relabel | O(V²E) | Preflow-push method |
| FIFO Push-Relabel | O(V³) | Queue-based push-relabel |
| Highest Label Push-Relabel | O(V²√E) | Priority-based push-relabel |
| MPM (Malhotra-Pramodh-Maheshwari) | O(V³) | For unit capacity networks |

### Minimum Cut

- Max-Flow Min-Cut Theorem
- Stoer-Wagner Algorithm
- Karger's Algorithm (Randomized)
- Karger-Stein Algorithm

### Minimum Cost Flow

- Successive Shortest Path
- Cycle-Canceling Algorithm
- Cost Scaling Algorithm
- Network Simplex

### Bipartite Matching

- Hopcroft-Karp Algorithm
- Hungarian Algorithm
- Kuhn's Algorithm

### Applications

- Maximum Bipartite Matching
- Vertex Cover
- Edge Cover
- Independent Set
- Baseball Elimination
- Project Selection
- Image Segmentation
- Airline Scheduling

---

## Linear Programming

### Simplex Method

- Primal Simplex
- Dual Simplex
- Revised Simplex
- Two-Phase Simplex
- Big-M Method

### Interior Point Methods

- Karmarkar's Algorithm
- Barrier Method
- Path-Following Method

### Integer Programming

- Branch and Bound
- Branch and Cut
- Cutting Plane Method
- Gomory Cuts

### Special Cases

- Transportation Problem (Northwest Corner, Vogel's)
- Assignment Problem (Hungarian)
- Transshipment Problem
- Network Simplex

---

## Approximation Algorithms

### Set Cover Variants

- Greedy Set Cover
- Primal-Dual Set Cover
- Weighted Set Cover

### Graph Problems

- Vertex Cover (2-approximation)
- Traveling Salesman (Christofides)
- Steiner Tree Approximation
- Max-Cut (Goemans-Williamson)
- Graph Coloring Approximation

### Scheduling

- Makespan Minimization
- Job Shop Scheduling
- Flow Shop Scheduling

### Knapsack Variants

- FPTAS for Knapsack
- Multi-dimensional Knapsack

### Clustering

- K-Center Problem
- K-Median Problem
- Facility Location

### Other Approximations

- Bin Packing (First Fit, Best Fit)
- Load Balancing
- MAX-SAT Approximation

---

## Cryptographic Algorithms

### Symmetric Encryption

- Caesar Cipher
- Vigenère Cipher
- Substitution Cipher
- Transposition Cipher
- DES (Data Encryption Standard)
- Triple DES
- AES (Advanced Encryption Standard)
- Blowfish
- Twofish
- RC4
- ChaCha20
- Salsa20

### Asymmetric Encryption

- RSA
- ElGamal
- Diffie-Hellman Key Exchange
- Elliptic Curve Diffie-Hellman (ECDH)
- Elliptic Curve Cryptography (ECC)

### Digital Signatures

- RSA Signatures
- DSA (Digital Signature Algorithm)
- ECDSA
- EdDSA
- Schnorr Signatures

### Key Derivation

- PBKDF2
- bcrypt
- scrypt
- Argon2

### Message Authentication

- HMAC
- CMAC
- Poly1305

### Zero-Knowledge Proofs

- Schnorr Protocol
- Fiat-Shamir Transform
- zk-SNARKs (conceptual)

---

## Compression Algorithms

### Lossless Compression

- Run-Length Encoding (RLE)
- Huffman Coding
- Arithmetic Coding
- LZ77
- LZ78
- LZW (Lempel-Ziv-Welch)
- LZSS
- Deflate
- GZIP
- Brotli
- Zstandard
- Burrows-Wheeler Transform
- Move-to-Front Transform
- bzip2

### Dictionary-Based

- LZ77 Family (LZSS, LZ4, Snappy)
- LZ78 Family (LZW, LZC, LZMW)
- LZO
- LZ4
- Snappy

### Lossy Compression (for reference)

- JPEG (DCT-based)
- MP3 (Psychoacoustic)
- Video Codecs (H.264, H.265)

---

## Numerical Algorithms

### Root Finding

- Bisection Method
- Newton-Raphson Method
- Secant Method
- Regula Falsi (False Position)
- Fixed Point Iteration
- Brent's Method
- Muller's Method

### Interpolation

- Linear Interpolation
- Polynomial Interpolation (Lagrange)
- Newton's Divided Differences
- Hermite Interpolation
- Spline Interpolation (Cubic, B-Spline)
- Chebyshev Interpolation

### Numerical Integration

- Rectangular Rule
- Trapezoidal Rule
- Simpson's Rule
- Romberg Integration
- Gaussian Quadrature
- Monte Carlo Integration
- Adaptive Quadrature

### Numerical Differentiation

- Forward Difference
- Backward Difference
- Central Difference
- Richardson Extrapolation

### ODE Solvers

- Euler's Method
- Improved Euler (Heun's Method)
- Runge-Kutta Methods (RK2, RK4)
- Adams-Bashforth Methods
- Adams-Moulton Methods
- Predictor-Corrector Methods

### Linear Algebra

- Gaussian Elimination
- LU Decomposition
- Cholesky Decomposition
- QR Decomposition
- Jacobi Method
- Gauss-Seidel Method
- Successive Over-Relaxation (SOR)
- Conjugate Gradient Method
- GMRES
- Power Iteration
- QR Algorithm (Eigenvalues)

### Optimization

- Gradient Descent
- Stochastic Gradient Descent
- Newton's Method (Optimization)
- Quasi-Newton Methods (BFGS, L-BFGS)
- Conjugate Gradient
- Simulated Annealing
- Genetic Algorithms
- Particle Swarm Optimization
- Nelder-Mead (Simplex)

---

## Machine Learning Algorithms

### Supervised Learning

#### Regression

- Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Support Vector Regression

#### Classification

- Logistic Regression
- Naive Bayes (Gaussian, Multinomial, Bernoulli)
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Trees (ID3, C4.5, CART)
- Random Forest
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- AdaBoost
- Neural Networks (MLP)

### Unsupervised Learning

#### Clustering

- K-Means Clustering
- K-Medoids (PAM)
- Hierarchical Clustering (Agglomerative, Divisive)
- DBSCAN
- OPTICS
- Mean Shift
- Spectral Clustering
- Gaussian Mixture Models (EM)
- BIRCH
- Affinity Propagation

#### Dimensionality Reduction

- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)
- t-SNE
- UMAP
- Autoencoders
- Independent Component Analysis (ICA)
- Factor Analysis
- Non-negative Matrix Factorization (NMF)

#### Association Rules

- Apriori Algorithm
- FP-Growth
- Eclat

### Reinforcement Learning

- Q-Learning
- SARSA
- Deep Q-Network (DQN)
- Policy Gradient
- Actor-Critic
- A3C (Asynchronous Advantage Actor-Critic)
- PPO (Proximal Policy Optimization)
- Monte Carlo Tree Search (MCTS)

### Neural Networks

- Perceptron
- Multi-Layer Perceptron (MLP)
- Backpropagation
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)
- LSTM
- GRU
- Transformer
- Attention Mechanism
- Generative Adversarial Networks (GAN)
- Variational Autoencoders (VAE)

### Ensemble Methods

- Bagging
- Boosting
- Stacking
- Voting Classifier

---

## Parallel and Distributed Algorithms

### Parallel Sorting

- Parallel Merge Sort
- Parallel Quick Sort
- Bitonic Sort
- Odd-Even Merge Sort
- Sample Sort
- Radix Sort (Parallel)

### Parallel Search

- Parallel BFS
- Parallel DFS
- Parallel A*

### MapReduce Algorithms

- Word Count
- Inverted Index
- PageRank (MapReduce)
- K-Means (MapReduce)
- Matrix Multiplication (MapReduce)

### Distributed Consensus

- Paxos
- Raft
- Byzantine Fault Tolerance (PBFT)
- Two-Phase Commit
- Three-Phase Commit

### Parallel Prefix

- Prefix Sum (Scan)
- Hillis-Steele Algorithm
- Blelloch Algorithm

### Load Balancing

- Round Robin
- Least Connections
- Weighted Round Robin
- Consistent Hashing

---

## Miscellaneous Algorithms

### Caching Algorithms

- LRU (Least Recently Used)
- LFU (Least Frequently Used)
- FIFO Cache
- Random Replacement
- ARC (Adaptive Replacement Cache)
- Clock Algorithm
- Second Chance Algorithm

### Memory Allocation

- First Fit
- Best Fit
- Worst Fit
- Buddy System
- Slab Allocation

### Scheduling Algorithms

- Round Robin
- Priority Scheduling
- Shortest Job First
- Shortest Remaining Time First
- Multilevel Queue
- Multilevel Feedback Queue
- Rate Monotonic Scheduling
- Earliest Deadline First

### Pattern Recognition

- Template Matching
- Feature Extraction
- Hough Transform

### Text Processing

- Tokenization
- Stemming (Porter Stemmer)
- Lemmatization
- TF-IDF
- BM25

### Miscellaneous

- Stable Marriage Problem (Gale-Shapley)
- Topological Sort
- Union-Find
- Flood Fill
- A* Search
- IDA* (Iterative Deepening A*)
- Minimax Algorithm
- Alpha-Beta Pruning
- Expectimax
- Dancing Links (Algorithm X)
- Reservoir Sampling
- Bloom Filter Operations
- Consistent Hashing
- Merkle Trees
- Undo/Redo (Command Pattern)

---

## Notes

This document provides a comprehensive reference for algorithms commonly studied in computer science and used in software development. Each algorithm serves specific purposes and has trade-offs in terms of time complexity, space complexity, and implementation complexity.

For detailed implementations and explanations, refer to the `notes/` directory.
