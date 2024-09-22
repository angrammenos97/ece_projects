#ifndef DISTANCES_H_
#define DISTANCES_H_

__global__ void distance_init_kernel(double* d_distances, unsigned int numberOfPoints, unsigned int dimensionOfPoints, unsigned int maxThreadsPerBlock);

__device__ void distance_from_last(double* points, unsigned int nodeOffset, unsigned int nodeLength, cudaStream_t nodeStream);

#endif // !DISTANCES_H_

