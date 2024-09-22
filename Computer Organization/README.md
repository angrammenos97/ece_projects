# Computer Organization Projects
This repository contains a collection of projects related to the *"Computer Organization"* subject. Each project demonstrates different concepts and techniques in computer organization, assembly programming, and optimization. Most of the projects are implemented both in assembly and C code.

## Projects
### 1. Simple Function
Performs basic arithmetic operations on four input integers. It adds the first two integers and subtracts the sum of the second two integers.

$$
f = (g + h) - (i + j)
$$

### 2. Fibonacci Function

Calculates the Fibonacci number of a given integer using recursion.
$$
F_0 = 0, F_1 = 1
$$

$$
F_n = F_{n-1} + F_{n-2}
$$

### 3. Half-Precision Floating Point Addition
Calculates the summation of two half-precision floating-point numbers using integer operations. It handles simple cases and includes conversion between full-precision and half-precision floating-point formats.

### 4. Blocked Matrix Multiplication
Performs matrix multiplication with blocking (tiling) optimization. This function utilizes cache performance by performing the matrix multiplication into smaller blocks.

$$
C_{i,j} = \sum_{k=0}^{K}A_{i,k}B_{k,j}
$$