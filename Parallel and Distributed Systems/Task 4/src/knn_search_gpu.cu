#include "knn_search.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "device_global_memory.h"
#include "vptree.h"

//Device global memory
__device__ double* corpusPoints;
__device__ double* queryPoints;
__device__ double* ndist;
__device__ unsigned int* nidx;
__device__ double* knnTreeMDs;
__device__ unsigned int* knnTreeIDXs;
__device__ unsigned int* offsetsStack;
__device__ unsigned int* lengthsStack;
__device__ double* parentNDistStack;
__device__ double* parentMdStack;
__device__ char* isInnerStack;
__device__ unsigned int knnNoP, knnMoP, knnDoP, knnK;


__global__ void memory_set_kernel(double* arr, double value, unsigned int length)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < length)
		arr[tid] = value;
}

__global__ void knn_search_init(double* d_X, double* d_Y, double* d_ndist, unsigned int* d_nidx, double* d_treeMDs, unsigned int* d_treeIDXs, unsigned int* d_offsetStack, unsigned int* d_lengthStack,
								double* d_parentNDistStack, double* d_parentMdStack, char* d_isInnerStack, unsigned int n, unsigned int m, unsigned int d, unsigned int k)
{
	if ((threadIdx.x) == 0 && (blockIdx.x == 0)) {
		knnNoP = n; knnMoP = m; knnDoP = d; knnK = k;
		//Initialize device pointers to global memory
		corpusPoints = d_X;
		queryPoints = d_Y;
		ndist = d_ndist;
		nidx = d_nidx;
		knnTreeMDs = d_treeMDs;
		knnTreeIDXs = d_treeIDXs;
		offsetsStack = d_offsetStack;
		lengthsStack = d_lengthStack;
		parentNDistStack = d_parentNDistStack;
		parentMdStack = d_parentMdStack;
		isInnerStack = d_isInnerStack;

		unsigned int blockSz = (knnMoP * knnK < MAXTHREADSPERBLOCK) ? knnMoP * knnK : MAXTHREADSPERBLOCK;
		unsigned int gridSz = ((knnMoP * knnK) + blockSz - 1) / blockSz;
		memory_set_kernel <<<gridSz, blockSz>>> (ndist, DBL_MAX, knnMoP * knnK);
	}
}

__device__ void insertion_sort(double distToVp, unsigned int vpIdx, unsigned int queryIdx)
{
	double* queryNdist = ndist + (queryIdx * knnK);
	unsigned int* queryNidx = nidx + (queryIdx * knnK);
	for (int k = knnK - 2; k >= 0; k--)
		if (distToVp < queryNdist[k]) {
			queryNdist[k + 1] = queryNdist[k];
			queryNidx[k + 1] = queryNidx[k];
		}
		else {
			queryNdist[k + 1] = distToVp;
			queryNidx[k + 1] = vpIdx;
			return;
		}
	//we have the closest neighbor yet
	queryNdist[0] = distToVp;
	queryNidx[0] = vpIdx;
}

__device__ double points_distance(unsigned int vpIdx, unsigned int queryIdx)
{
	double tempdist, pDistance = 0.0;
	for (unsigned int d = 0; d < knnDoP; d++) {
		tempdist = *(corpusPoints + (vpIdx * knnDoP) + d) - *(queryPoints + (queryIdx * knnDoP) + d);
		pDistance += tempdist * tempdist;
	}
	return sqrt(pDistance);
}

__global__ void find_nearest_kernel()
{
	unsigned int queryIdx = threadIdx.x + (blockIdx.x * blockDim.x);
	if (queryIdx < knnMoP) {
		//subtrees variables
		unsigned int nodeOffset, nodeLength;
		unsigned int innerOffset, innerLength;
		unsigned int outerOffset, outerLength;
		double parentNDist, parentMd, tau = DBL_MAX;
		char isInner;

		//each thread get it's own stack row
		unsigned int maxDepth = (unsigned int)log2((double)knnNoP) + 1;
		unsigned int* tOffsetStack = offsetsStack + (queryIdx * maxDepth);
		unsigned int* tLengthStack = lengthsStack + (queryIdx * maxDepth);
		double* tParentNDistStack = parentNDistStack + (queryIdx * maxDepth);
		double* tParentMdStack = parentMdStack + (queryIdx * maxDepth);
		char* tIsInnerStack = isInnerStack + (queryIdx * maxDepth);
		unsigned int stackIdx = 0;

		//add in stack the root of the tree
		tOffsetStack[stackIdx] = 0;					tLengthStack[stackIdx] = knnNoP;
		tParentNDistStack[stackIdx] = DBL_MAX;		tParentMdStack[stackIdx] = 0.0;		tIsInnerStack[stackIdx++] = 1;
		//search in tree
		while (stackIdx > 0) {
			//pop out from stack
			nodeOffset = tOffsetStack[--stackIdx];			nodeLength = tLengthStack[stackIdx];
			parentNDist = tParentNDistStack[stackIdx];		parentMd = tParentMdStack[stackIdx];	isInner = tIsInnerStack[stackIdx];

			//skip subtree if neccecary
			if (isInner) {
				if ((parentMd + tau) < parentNDist)
					continue;
			}
			else if (parentMd > (parentNDist + tau))
				continue;
			//calculate distance from current vp
			double distToVP = points_distance(knnTreeIDXs[nodeOffset], queryIdx);
			//add current vp if it is close to q
			tau = *(ndist + (queryIdx * knnK) + (knnK - 1));	//tau = furthest neighbor
			if (distToVP < tau) {
				insertion_sort(distToVP, knnTreeIDXs[nodeOffset], queryIdx);
			}
			tau = *(ndist + (queryIdx * knnK) + (knnK - 1));	//tau = furthest neighbor
			//get inner and outer subtrees positions
			innerOffset = nodeOffset + 1;
			if ((nodeLength - 1) & 1) {	//odd number
				innerLength = ((nodeLength - 1) / 2) + 1;
				outerOffset = innerOffset + innerLength;
				outerLength = innerLength - 1;
			}
			else {						//even number
				innerLength = (nodeLength - 1) / 2;
				outerOffset = innerOffset + innerLength;
				outerLength = innerLength;
			}
			//q inside of the vp's circle
			if (distToVP < knnTreeMDs[nodeOffset]) {
				//add to stack outer subtree
				if (outerLength) {
					tOffsetStack[stackIdx] = outerOffset;		tLengthStack[stackIdx] = outerLength;
					tParentNDistStack[stackIdx] = distToVP;		tParentMdStack[stackIdx] = knnTreeMDs[nodeOffset];	tIsInnerStack[stackIdx++] = 0;
				}
				//add to stack inner subtree
				if (innerLength) {
					tOffsetStack[stackIdx] = innerOffset;		tLengthStack[stackIdx] = innerLength;
					tParentNDistStack[stackIdx] = distToVP;		tParentMdStack[stackIdx] = knnTreeMDs[nodeOffset];	tIsInnerStack[stackIdx++] = 1;
				}
			}
			//q outside of the vp's circle
			else {
				//add to stack inner subtree
				if (innerLength) {
					tOffsetStack[stackIdx] = innerOffset;		tLengthStack[stackIdx] = innerLength;
					tParentNDistStack[stackIdx] = distToVP;		tParentMdStack[stackIdx] = knnTreeMDs[nodeOffset];	tIsInnerStack[stackIdx++] = 1;
				}
				//add to stack outer subtree
				if (outerLength) {
					tOffsetStack[stackIdx] = outerOffset;		tLengthStack[stackIdx] = outerLength;
					tParentNDistStack[stackIdx] = distToVP;		tParentMdStack[stackIdx] = knnTreeMDs[nodeOffset];	tIsInnerStack[stackIdx++] = 0;
				}
			}

		}
	}
}

/*Function to convert tree into array*/
void node_to_element(double* treeMDs, unsigned int* treeIDXs, vptree* root, unsigned length)
{
	if (length == 0) return;
	treeMDs[0] = root->md;
	treeIDXs[0] = root->idx;
	if ((length - 1) & 1) { //odd number
		node_to_element(treeMDs + 1, treeIDXs + 1, root->inner, ((length - 1) / 2) + 1);
		node_to_element(treeMDs + 1 + ((length - 1) / 2) + 1, treeIDXs + 1 + ((length - 1) / 2) + 1, root->outer, (length - 1) / 2);
	}
	else {					//even number
		node_to_element(treeMDs + 1, treeIDXs + 1, root->inner, (length - 1) / 2);
		node_to_element(treeMDs + 1 + ((length - 1) / 2), treeIDXs + 1 + ((length - 1) / 2), root->outer, (length - 1) / 2);
	}
}

knnresult kNN(double* X, double* Y, int n, int m, int d, int k)
{
	knnresult results;
	results.m = m;
	results.k = k;
	results.ndist = (double*)malloc((m * k) * sizeof(double));
	results.nidx = (int*)malloc((m * k) * sizeof(int));

	vptree* root = buildvp(X, n, d);
	if (root == NULL) { printf("Error: buildvp(%d, %d)\n", n, d); return results; }

	//Copy tree into array
	double* h_treeMDs = (double*)malloc(n * sizeof(double));
	unsigned int* h_treeIDXs = (unsigned int*)malloc(n * sizeof(unsigned int));
	node_to_element(h_treeMDs, h_treeIDXs, root, n);

	//Allocate device memory
	cudaError err = (cudaError)knn_memory_allocate(n, m, d, k);
	if (err != cudaSuccess) { printf("Error: knn_memory_allocate(%d, %d, %d, %d):%s\n", n, m, d, k, cudaGetErrorString(err)); return results; }

	//Transfer data to gpu
	err = cudaMemcpy(d_points, X, (n * d) * sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { printf("Error: cudaMemcpy(d_points<-X, %d * %d):%s\n", n, d, cudaGetErrorString(err)); return results; }
	err = cudaMemcpy(d_qpoints, Y, (m * d) * sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { printf("Error: cudaMemcpy(d_qpoints<-Y, %d * %d):%s\n", m, d, cudaGetErrorString(err)); return results; }
	err = cudaMemcpy(d_treeMDs, h_treeMDs, n * sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { printf("Error: cudaMemcpy(d_treeMDs<-h_treeMDs, %d):%s\n", n, cudaGetErrorString(err)); return results; }
	err = cudaMemcpy(d_treeIDXs, h_treeIDXs, n * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { printf("Error: cudaMemcpy(d_treeIDXs<-h_treeIDXs, %d):%s\n", n, cudaGetErrorString(err)); return results; }

	//Initialize library
	knn_search_init << <1, 1 >> > (d_points, d_qpoints, d_ndist, d_nidx, d_treeMDs, d_treeIDXs, d_offsetsStack, d_lengthsStack, d_parentNDistStack, d_parentMdStack, d_isInnerStack, n, m, d, k);
	cudaDeviceSynchronize();

	//Find k nearest neighbors
	cudaEvent_t start, stop;	cudaEventCreate(&start);	cudaEventCreate(&stop);
	unsigned int blockSz = (m < MAXTHREADSPERBLOCK) ? m : MAXTHREADSPERBLOCK;
	unsigned int gridSz = (m + blockSz - 1) / blockSz;
	cudaEventRecord(start);
	find_nearest_kernel << <gridSz, blockSz >> > ();
	cudaEventRecord(stop);

	//Wait cuda to capture execution time event
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Searches took %fmsec! ", milliseconds);

	//Copy back the results
	err = cudaMemcpy(results.ndist, d_ndist, (m * k) * sizeof(double), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { printf("Error: cudaMemcpy(results.ndist<-d_ndist, %d * %d):%s\n", m, k, cudaGetErrorString(err)); return results; }
	err = cudaMemcpy(results.nidx, d_nidx, (m * k) * sizeof(int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { printf("Error: cudaMemcpy(results.nidx<-d_nidx, %d * %d):%s\n", m, k, cudaGetErrorString(err)); return results; }

	//Free memory
	knn_deallocate();
	free(h_treeMDs);
	free(h_treeIDXs);
	return results;
}

knnresult distrAllkNN(double* X, int n, int d, int k)
{
	return kNN(X, X, n, n, d, k);
}
