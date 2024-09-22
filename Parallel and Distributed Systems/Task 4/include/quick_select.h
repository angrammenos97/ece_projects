#ifndef QUICK_SELECT_H_
#define QUICKS_SELECT_H_

/*Returns the smallest power of two*/
__host__ __device__ unsigned int smallest_power_two(unsigned int n);

/*Fills out array with indexes starting from 0 to length*/
__global__ void create_indexes_kernel(unsigned int* out, unsigned int length);

/*Initialize quick select library providing the allocated memory*/
__global__ void qs_init_kernel(double* d_qsMain, double* d_qsAux, unsigned int* d_swaps, unsigned int* d_f, unsigned int* d_t, unsigned int* d_addr, unsigned int* d_NFs, char* d_e, unsigned int maxThreadsPerBlock);

/*Return the k-th value from the d_qsMain*/
__device__ double quick_select(unsigned int nodeOffset, unsigned int nodeLength, unsigned int nodeK, cudaStream_t nodeStream);

#endif // !QUICK_SELECT_H_
