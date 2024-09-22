#include "vptree.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "device_global_memory.h"
#include "distances.h"
#include "quick_select.h"

//Device global memory
__device__ double* pointsMain;
__device__ double* pointsAux;
__device__ double* vpDistances;
__device__ unsigned int* indexesMain, * indexesAux, * vpSwaps;
__device__ unsigned int vpNoP, vpDoP, vpMaxThreadsPerBlock;
//Buffers to build tree
__device__ double* vpTreeMDs;
__device__ unsigned int* vpTreeIDXs, * nodesOffset, * nodesLength;

/*Initialize memory to build vp tree*/
__global__ void vp_init_kernel(double* d_points, double* d_pointsAux, unsigned int* d_indexes, unsigned int* d_indexesAux, unsigned int* d_vpSwaps, double* d_distances, double* d_treeMDs, unsigned int* d_treeIDXs, 
								unsigned int* d_nodeOffset, unsigned int* d_nodeLength, unsigned int numberOfPoints, unsigned int dimensionOfPoints, unsigned int maxThreadsPerBlock)
{
	if ((threadIdx.x) == 0 && (blockIdx.x == 0)) {
		vpNoP = numberOfPoints;	vpDoP = dimensionOfPoints;	vpMaxThreadsPerBlock = maxThreadsPerBlock;
		//Initialize device pointers to global memory
		pointsMain = d_points;		pointsAux = d_pointsAux;
		indexesMain = d_indexes;	indexesAux = d_indexesAux;
		vpSwaps = d_vpSwaps;		vpDistances = d_distances;
		vpTreeMDs = d_treeMDs;		vpTreeIDXs = d_treeIDXs;
		nodesOffset = d_nodeOffset;	nodesLength = d_nodeLength;
		nodesOffset[0] = 0;			nodesLength[0] = vpNoP;

		unsigned int blockSz = (vpNoP < vpMaxThreadsPerBlock) ? vpNoP : vpMaxThreadsPerBlock;
		unsigned int gridSz = (vpNoP + vpMaxThreadsPerBlock - 1) / vpMaxThreadsPerBlock;
		create_indexes_kernel <<<gridSz, blockSz>>> (indexesMain, vpNoP);
	}
}

/*Helper function to fill rest distances with big numbers in fix_distace_array function*/
__global__ static void memory_set_kernel(double* arr, double value, unsigned int length)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < length)
		arr[tid] = value;
}

/*Helper function to fix distances array if it's not power of two in median function*/
__device__ unsigned int fix_distace_array(unsigned int nodeOffset, unsigned int nodeLength, cudaStream_t nodeStream)
{
	unsigned int moreLength = smallest_power_two(nodeLength) - nodeLength;
	//Fill rest distances array with big distances
	if (moreLength > 0) {
		unsigned int blockSz = (moreLength < vpMaxThreadsPerBlock) ? moreLength : vpMaxThreadsPerBlock;
		unsigned int gridSz = (moreLength + vpMaxThreadsPerBlock - 1) / vpMaxThreadsPerBlock;
		memory_set_kernel <<<gridSz, blockSz, 0, nodeStream>>> (vpDistances + nodeOffset + nodeLength, DBL_MAX, moreLength);
	}
	return moreLength;
}

/*Helper function swap points in median function*/
__global__ void swap_points_kernel(double* out, double* in, unsigned int nodeOffset, unsigned int nodeLength)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < nodeLength)
		for (unsigned int d = 0; d < vpDoP; d++)
			*(out + tid + d * vpNoP) = *(in + vpSwaps[nodeOffset + tid] + d * vpNoP);
}

/*Helper function swap indexes in median function*/
__global__ void swap_indexes_kernel(unsigned int* out, unsigned int* in, unsigned int nodeOffset, unsigned int nodeLength)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < nodeLength)
		out[tid] = in[vpSwaps[nodeOffset + tid]];
}

/*Finds the median of the distances*/
__device__ double median(double* pointsIn, unsigned int nodeOffset, unsigned int nodeLength, cudaStream_t nodeStream)
{
	if (nodeLength == 1) return 0.0;
	//Calculate distances all points to the last (vantage point)
	distance_from_last(pointsIn, nodeOffset, nodeLength, nodeStream);
	//Select the median of them
	unsigned int fixedLength = (nodeLength - 1) + fix_distace_array(nodeOffset, nodeLength - 1, nodeStream);
	return quick_select(nodeOffset, fixedLength, (nodeLength - 2) / 2, nodeStream);
}

/*Builds tree sequentially each node per depth*/
__global__ void vpt_kernel(unsigned int nodes, unsigned int maxNodes, unsigned int depth)
{
	//Create threads's stream
	cudaStream_t threadStream;
	cudaStreamCreateWithFlags(&threadStream, cudaStreamNonBlocking);
	//Set needed variables
	double* pointsIn, * pointsOut;
	unsigned int* indexesIn, * indexesOut;
	unsigned int blockSz, gridSz, nodeOffset, nodeLength, qsOffset, leftChild, rightChild;
	unsigned int node = threadIdx.x + (blockIdx.x * blockDim.x);
	//Each thread creates it's assigned node
	//Find out where the data is
	nodeOffset = nodesOffset[node * maxNodes / nodes];		nodeLength = nodesLength[node * maxNodes / nodes];
	if (!nodeLength) {cudaStreamDestroy(threadStream); return;}		//if node doesn't exist continue to the next one
	if (depth & 1) {
		pointsIn = pointsMain + nodeOffset;			pointsOut = pointsAux + nodeOffset;
		indexesIn = indexesMain + nodeOffset;		indexesOut = indexesAux + nodeOffset;
	}
	else {
		pointsIn = pointsAux + nodeOffset;		pointsOut = pointsMain + nodeOffset;
		indexesIn = indexesAux + nodeOffset;	indexesOut = indexesMain + nodeOffset;
	}
	//Select as vantage point the last point in the array
	vpTreeIDXs[nodeOffset + (nodeLength - 1)] = indexesIn[nodeLength - 1];
	//Find the median of the distances
	qsOffset = node * ((2 * maxNodes) / nodes);
	vpTreeMDs[nodeOffset + (nodeLength - 1)] = median(pointsIn, qsOffset, nodeLength, threadStream);
	//Swap points to prepare them for the next partition
	blockSz = ((nodeLength - 1) < vpMaxThreadsPerBlock) ? (nodeLength - 1) : vpMaxThreadsPerBlock;
	gridSz = ((nodeLength - 1) + blockSz - 1) / blockSz;
	swap_points_kernel <<<gridSz, blockSz, 0, threadStream>>> (pointsOut, pointsIn, qsOffset, nodeLength - 1);
	swap_indexes_kernel <<<gridSz, blockSz, 0, threadStream>>> (indexesOut, indexesIn, qsOffset, nodeLength - 1);
	//Calculate offset and length for the children nodes
	leftChild = node * maxNodes / nodes;		rightChild = leftChild + (maxNodes / (2 * nodes));
	if ((nodeLength - 1) & 1) {	//odd number of length
		//left child
		nodesOffset[leftChild] = nodeOffset;
		nodesLength[leftChild] = ((nodeLength - 1) / 2) + 1;
		//right child
		nodesOffset[rightChild] = nodeOffset + ((nodeLength - 1) / 2) + 1;
		nodesLength[rightChild] = (nodeLength - 1) / 2;
	}
	else {	//even number of length
		//left child
		nodesOffset[leftChild] = nodeOffset;
		nodesLength[leftChild] = (nodeLength - 1) / 2;
		//right child
		nodesOffset[rightChild] = nodeOffset + ((nodeLength - 1) / 2);
		nodesLength[rightChild] = (nodeLength - 1) / 2;
	}
	cudaStreamDestroy(threadStream);
}

///////////////////////////////////////////////////////

/*Helper function to create the nodes of the structured tree*/
__host__ vptree* create_node(double* points, double* h_treeMDs, unsigned int* h_treeIDXs, unsigned int length, unsigned int dimensionOfPoints)
{
	if (length == 0) return NULL;
	vptree* subtree = (vptree*)malloc(sizeof(vptree));
	subtree->vp = points + (h_treeIDXs[length - 1] * dimensionOfPoints);
	subtree->md = h_treeMDs[length - 1];
	subtree->idx = h_treeIDXs[length - 1];
	if ((length - 1) & 1) {	//odd number
		subtree->inner = create_node(points, h_treeMDs, h_treeIDXs, ((length - 1) / 2) + 1, dimensionOfPoints);
		subtree->outer = create_node(points, h_treeMDs + ((length - 1) / 2) + 1, h_treeIDXs + ((length - 1) / 2) + 1, (length - 1) / 2, dimensionOfPoints);
	}
	else {					//even number
		subtree->inner = create_node(points, h_treeMDs, h_treeIDXs, (length - 1) / 2, dimensionOfPoints);
		subtree->outer = create_node(points, h_treeMDs + ((length - 1) / 2), h_treeIDXs + ((length - 1) / 2), (length - 1) / 2, dimensionOfPoints);
	}
	return subtree;
}

/*Helper function to create tree structure out of the arrays*/
__host__ vptree* create_tree(double* points, double* d_treeMDs, unsigned int* d_treeIDXs, unsigned int numberOfPoints, unsigned int dimensionOfPoints)
{
	cudaError err;	
	//Initialize host memory
	double* h_treeMDs = (double*)malloc(numberOfPoints * sizeof(double));
	unsigned int* h_treeIDXs = (unsigned int*)malloc(numberOfPoints * sizeof(unsigned int));
	//Copy tree to host
	err = cudaMemcpy(h_treeMDs, d_treeMDs, numberOfPoints * sizeof(double), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { printf("Error: cudaMemcpy(vpTreeMDs<-d_treeMDs, %u):%s\n", numberOfPoints, cudaGetErrorString(err)); return NULL; }
	err = cudaMemcpy(h_treeIDXs, d_treeIDXs, numberOfPoints * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { printf("Error: cudaMemcpy(vpTreeIDXs<-d_treeIDXs, %u):%s\n", numberOfPoints, cudaGetErrorString(err)); return NULL; }
	//Create tree from the arrays
	vptree* root = create_node(points, h_treeMDs, h_treeIDXs, numberOfPoints, dimensionOfPoints);
	//Free memory
	free(h_treeIDXs);
	free(h_treeMDs);
	//Return builded tree
	return root;
}

///////////////////////////////////////////////////////

vptree* buildvp_gpu(double* points, int numberOfPoints, int dimensionOfPoints)
{
	//Transpose points
	double* points_t = (double*)malloc(numberOfPoints * dimensionOfPoints * sizeof(double));
	for (int i = 0; i < numberOfPoints; i++)
		for (int j = 0; j < dimensionOfPoints; j++)
			*(points_t + i + (j * numberOfPoints)) = *(points + (i * dimensionOfPoints) + j);

	//Initialize gpu memory
	unsigned int maxNodes = smallest_power_two(numberOfPoints + 1) / 2;
	cudaError err = (cudaError)qs_memory_allocate(numberOfPoints, maxNodes);
	if (err != cudaSuccess) { printf("Error: qs_memory_allocate(%d, %d):%s\n", numberOfPoints, maxNodes, cudaGetErrorString(err)); return NULL; }
	err = (cudaError)di_memory_allocate(numberOfPoints);
	if (err != cudaSuccess) { printf("Error: di_memory_allocate(%d):%s\n", numberOfPoints, cudaGetErrorString(err)); return NULL; }
	err = (cudaError)vp_memory_allocate(numberOfPoints, dimensionOfPoints);
	if (err != cudaSuccess) { printf("Error: vp_memory_allocate(%d, %d):%s\n", numberOfPoints, dimensionOfPoints, cudaGetErrorString(err)); return NULL; }

	//Copy points to the gpu
	err = cudaMemcpy(d_points, points_t, (numberOfPoints * dimensionOfPoints) * sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { printf("Error: cudaMemcpy(d_points<-points_t, %u * %u):%s\n", numberOfPoints, dimensionOfPoints, cudaGetErrorString(err)); return NULL; }
	
	//Initialize libraries
	vp_init_kernel <<<1, 1>>> (d_points, d_pointsAux, d_indexes, d_indexesAux, d_vpSwaps, d_distances, d_treeMDs, d_treeIDXs, d_nodesOffset, d_nodesLength, numberOfPoints, dimensionOfPoints, MAXTHREADSPERBLOCK);
	distance_init_kernel <<<1, 1>>> (d_distances, numberOfPoints, dimensionOfPoints, MAXTHREADSPERBLOCK);
	qs_init_kernel <<<1, 1>>> (d_distances, d_qsAux, d_vpSwaps, d_f, d_t, d_addr, d_NFs, d_e, MAXTHREADSPERBLOCK);
	cudaDeviceSynchronize();
	
	//Build tree
	unsigned int blockSz, gridSz, depth = 0;
	cudaEvent_t start, stop;	cudaEventCreate(&start);	cudaEventCreate(&stop);
	cudaEventRecord(start);
	for (unsigned int nodes = 1; nodes <= maxNodes; nodes <<= 1) {
		depth += 1;
		blockSz = (nodes < MAXTHREADSPERBLOCK) ? nodes : MAXTHREADSPERBLOCK;
		gridSz = (nodes + blockSz - 1) / blockSz;
		vpt_kernel <<<gridSz, blockSz>>> (nodes, maxNodes, depth);
		cudaDeviceSynchronize();
	}
	cudaEventRecord(stop);
	
	//Wait cuda to capture execution time event
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Calculations took %fmsec! ", milliseconds);
	
	//Copy tree
	vptree* root = create_tree(points, d_treeMDs, d_treeIDXs, numberOfPoints, dimensionOfPoints);
	
	//Free memory
	qs_memory_deallocate();
	di_memory_deallocate();
	vp_memory_deallocate();
	free(points_t);
	return root;
}