# Parallel and Distributed Systems Projects

## Overview
This repository contains the source code for four tasks completed for the *"Parallel and Distributed Systems"* course. Each task demonstrates different parallel computing paradigms for solving concurrent computational problems using libraries such as Pthreads, OpenMP, CUDA, and MPI.

## Task 1: Parallel Vantage-Point Tree Construction
**Goal:** Implement the Vantage-Point Tree (VP-tree) in parallel using libraries such as Pthreads, OpenMP, and Cilk.

### Description:
This task focuses on building a VP-tree from a set of high-dimensional points. Given a query point, a quick find of the closest points is achievable using the tree. Parallelization techniques with Pthreads, OpenMP, and Cilk libraries were used to improve the performance of tree construction and nearest-neighbor searches.

### Files:
- vptree_pthreads.c: Implementation using Pthreads.
- vptree_openmp.c: Implementation using OpenMP.
- vptree_cilk.c: Implementation using Cilk.
- vptree_sequential.c: Sequential implementation.
- Makefile: Instructions for compiling and running each implementation.

## Task 2: Distributed k-Nearest Neighbors using MPI
**Goal:** Implement distributed k-Nearest Neighbors (k-NN) using MPI.

### Description:
In this task, the k-NN algorithm is implemented using MPI to distribute the workload across multiple processors. The algorithm calculates the k nearest neighbors for a given set of points in a distributed memory environment. It also uses OpenBLAS to accelerate matrix and vector operations.

### Files:
- knnring_sequential.c: Sequential k-NN implementation.
- knnring_mpi_s.c: Synchronous MPI implementation for distributed k-NN search.
- knnring_mpi_a.c: Asynchronous MPI implementation for distributed k-NN search.
- knnring_mpi_r.c: Asynchronous MPI implementation with global reduction for distributed k-NN search.
- Makefile: Instructions for compiling and running the MPI-based code.

## Task 3: CUDA Programming for Ising Model Calculation
**Goal:** Implement the Ising model using CUDA for parallel computation on a GPU.

### Description:
This task involves simulating the Ising model using both a serial and CUDA-based implementation. The Ising model simulates interactions between neighboring particles in a lattice. The code is optimized for parallel execution using GPU threads, shared memory, and efficient memory access patterns.

### Files:
- ising_sequential.c: Sequential implementation of the Ising model.
- ising_v1.c: First version of the CUDA implementation.
- ising_v2.c: Improved CUDA implementation using better memory access.
- ising_v3.c: Final CUDA implementation using shared memory.
- Makefile: Instructions for compiling the CUDA code.

## Task 4: GPU and Distributed Vantage-Point Tree and k-NN Search
**Goal:** Extend Task 1 by implementing the VP-tree and k-NN search on the GPU using CUDA.

## Description:
This task builds upon Task 1 by transferring the workload of constructing and searching the VP-tree to the GPU. CUDA is used to parallelize the distance computations and nearest neighbor search. The k-NN search is also implemented using distributed computing.

## Files:
- vptree_gpu.cu: CUDA implementation of VP-tree construction.
- knn_search_gpu.cu: CUDA implementation of k-NN search.
- Makefile: Instructions for compiling the GPU-based code.
