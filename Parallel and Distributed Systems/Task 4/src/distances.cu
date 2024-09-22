#include "distances.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ double* distances;
__device__ unsigned int numberOfColumns, numberOfRows, diMaxThreadsPerBlock;

__global__ void distance_kernel(double* points, unsigned int nodeOffset, unsigned int nodeLength)
{
	extern __shared__ double lastPoint[];
	unsigned int pntIdx = threadIdx.x;	//thread index
	//load last point into shared mem
	if (threadIdx.x < numberOfRows)
		for (unsigned int stride = 0; (stride + pntIdx) < numberOfRows; stride += blockDim.x)
			lastPoint[pntIdx + stride] = *(points + (nodeLength - 1) + ((pntIdx + stride) * numberOfColumns));
	__syncthreads();
	pntIdx = threadIdx.x + (blockIdx.x * blockDim.x);	//point index
	double pointDist, tempDiff;
	if (pntIdx < nodeLength - 1) {
		pointDist = 0.0;
		for (unsigned int d = 0; d < numberOfRows; d++) {
			tempDiff = *(points + pntIdx + d * numberOfColumns) - lastPoint[d];
			pointDist += tempDiff * tempDiff;
		}
		distances[nodeOffset + pntIdx] = sqrt(pointDist);		//save result back to global mem
	}
}

__device__ void distance_from_last(double* points, unsigned int nodeOffset, unsigned int nodeLength, cudaStream_t nodeStream)
{
	unsigned int totalThreads = (numberOfRows > (nodeLength - 1)) ? numberOfRows : nodeLength - 1;
	unsigned int blockSz = (totalThreads < diMaxThreadsPerBlock) ? totalThreads : diMaxThreadsPerBlock;
	unsigned int gridSz = (totalThreads + blockSz - 1) / blockSz;
	distance_kernel <<<gridSz, blockSz, numberOfRows * sizeof(double), nodeStream>>> (points, nodeOffset, nodeLength);	//"+numberOfRows" to hold the last point in shared mem
}

__global__ void distance_init_kernel(double* d_distances, unsigned int numberOfPoints, unsigned int dimensionOfPoints, unsigned int maxThreadsPerBlock)
{
	if ((threadIdx.x) == 0 && (blockIdx.x == 0)) {
		numberOfColumns = numberOfPoints;		numberOfRows = dimensionOfPoints;		diMaxThreadsPerBlock = maxThreadsPerBlock;
		//Initialize device pointers to global memory
		distances = d_distances;		
	}
}
