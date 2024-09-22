#include "device_global_memory.h"
#include <math.h>
#include <cuda_runtime.h>

//vptree.cu global variables
double* d_points;
double* d_pointsAux;
unsigned int* d_indexes;
unsigned int* d_indexesAux;
unsigned int* d_vpSwaps;
double* d_treeMDs;
unsigned int* d_treeIDXs;
unsigned int* d_nodesOffset;
unsigned int* d_nodesLength;

//distances.cu global variables
double* d_distances;

//quick_select.cu global variables
double* d_qsAux;
unsigned int* d_f;
unsigned int* d_t;
unsigned int* d_addr;
unsigned int* d_NFs;
char* d_e;

//knn_search.cu global variables
double* d_qpoints;
double* d_ndist;
unsigned int* d_nidx;
unsigned int* d_offsetsStack;
unsigned int* d_lengthsStack;
double* d_parentNDistStack;
double* d_parentMdStack;
char* d_isInnerStack;

/*Returns the smallest power of two*/
static unsigned int smallest_power_two(unsigned int n)
{
	unsigned int N = n;
	if ((N & (N - 1)) != 0) {	// fix if n is not power of 2
		N = 1;
		while (N < n)
			N <<= 1;
	}
	return N;
}

/*Functions to initialize memory*/
int qs_memory_allocate(unsigned int numberOfPoints, unsigned int maxParallelNodes)
{
	cudaError err;
	unsigned int fixedNoP = smallest_power_two(numberOfPoints + 1);		//quick select needs length in powers of two
	//quick_select.cu global variables
	err = cudaMalloc(&d_qsAux, fixedNoP * sizeof(double));										if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_f, fixedNoP * sizeof(unsigned int));									if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_t, fixedNoP * sizeof(unsigned int));									if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_addr, fixedNoP * sizeof(unsigned int));									if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_NFs, maxParallelNodes * sizeof(unsigned int));							if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_e, fixedNoP * sizeof(char));											if (err != cudaSuccess) return err;
	return cudaSuccess;
}

int di_memory_allocate(unsigned int numberOfPoints)
{
	cudaError err;
	unsigned int fixedNoP = smallest_power_two(numberOfPoints + 1);		//quick select needs length in powers of two
	//distances.cu global variables
	err = cudaMalloc(&d_distances, fixedNoP * sizeof(double));									if (err != cudaSuccess) return err;
	return cudaSuccess;
}

int vp_memory_allocate(unsigned int numberOfPoints, unsigned int dimensionOfPoints)
{
	cudaError err;
	unsigned int fixedNoP = smallest_power_two(numberOfPoints + 1);		//quick select needs length in powers of two
	unsigned int maxNodes = smallest_power_two(numberOfPoints + 1) / 2;	//max nodes on the last level of the tree
	//vptree.cu global variables
	err = cudaMalloc(&d_points, (numberOfPoints * dimensionOfPoints) * sizeof(double));			if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_pointsAux, (numberOfPoints * dimensionOfPoints) * sizeof(double));		if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_indexes, numberOfPoints * sizeof(unsigned int));						if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_indexesAux, numberOfPoints * sizeof(unsigned int));						if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_vpSwaps, fixedNoP * sizeof(unsigned int));								if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_treeMDs, numberOfPoints * sizeof(double));								if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_treeIDXs, numberOfPoints * sizeof(unsigned int));						if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_nodesOffset, maxNodes * sizeof(unsigned int));							if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_nodesLength, maxNodes * sizeof(unsigned int));							if (err != cudaSuccess) return err;
	return cudaSuccess;
}

int knn_memory_allocate(unsigned int n, unsigned int m, unsigned int d, unsigned int k)
{
	cudaError err;
	unsigned int maxDepth = (unsigned int)log2f(n) + 1;
	//knn_search.cu global variables
	err = cudaMalloc(&d_points, (n * d) * sizeof(double));										if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_qpoints, (m * d) * sizeof(double));										if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_ndist, (m * k) * sizeof(double));										if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_nidx, (m * k) * sizeof(unsigned int));									if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_treeMDs, n * sizeof(double));											if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_treeIDXs, n * sizeof(unsigned int));									if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_offsetsStack, (m * maxDepth) * sizeof(unsigned int));					if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_lengthsStack, (m * maxDepth) * sizeof(unsigned int));					if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_parentNDistStack, (m * maxDepth) * sizeof(double));						if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_parentMdStack, (m * maxDepth) * sizeof(double));						if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_isInnerStack, (m * maxDepth) * sizeof(char));							if (err != cudaSuccess) return err;
	return cudaSuccess;
}

/*Functions to free memory*/
void qs_memory_deallocate()
{
	cudaFree(d_qsAux);
	cudaFree(d_f);
	cudaFree(d_t);
	cudaFree(d_addr);
	cudaFree(d_e);
}

void di_memory_deallocate()
{
	cudaFree(d_distances);
}

void vp_memory_deallocate()
{
	cudaFree(d_points);
	cudaFree(d_pointsAux);
	cudaFree(d_indexes);
	cudaFree(d_indexesAux);
	cudaFree(d_vpSwaps);
	cudaFree(d_treeMDs);
	cudaFree(d_treeIDXs);
	cudaFree(d_nodesOffset);
	cudaFree(d_nodesLength);	
}

void knn_deallocate()
{
	cudaFree(d_points);
	cudaFree(d_qpoints);
	cudaFree(d_ndist);
	cudaFree(d_nidx);
	cudaFree(d_treeMDs);
	cudaFree(d_treeIDXs);
	cudaFree(d_offsetsStack);
	cudaFree(d_lengthsStack);
	cudaFree(d_parentNDistStack);
	cudaFree(d_parentMdStack);
	cudaFree(d_isInnerStack);
}
