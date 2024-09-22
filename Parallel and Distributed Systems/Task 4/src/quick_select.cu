#include "quick_select.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//device pointers to global memory
__device__ double* qsMain, * qsAux;
__device__ unsigned int* qsSwaps;		//pointer to swaps array
__device__ unsigned int* f, * t, * addr, * NFs;
__device__ char* e;
__device__ unsigned int qsMaxThreadsPerBlock;


/*STEP 1
*Distribute pivot across segment
*and compare input with pivot*/
__global__ void compare_pivot_kernel(double* dataIn, unsigned int offset, unsigned int length, unsigned int pvIdx, char depth)
{
	__shared__ double s_pivot;
	if (threadIdx.x == 0)	//only first thread per block
		s_pivot = dataIn[pvIdx];				//copy pivot to shared mem
	__syncthreads();
	unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid < length)
		if (depth)	//On alternating passes compare either greater or greater-or-equal
			if (dataIn[tid] >= s_pivot) {
				e[offset + tid] = (char)0;
				f[offset + tid] = 0;
			}
			else {
				e[offset + tid] = (char)1;
				f[offset + tid] = 1;
			}
		else
			if (dataIn[tid] > s_pivot) {
				e[offset + tid] = (char)0;
				f[offset + tid] = 0;
			}
			else {
				e[offset + tid] = (char)1;
				f[offset + tid] = 1;
			}
}
/*STEP 2
*Enumerate with false=1*/
__global__ void reduce_kernel(unsigned int offset, unsigned int d, unsigned int n)
{
	unsigned int k = (threadIdx.x + (blockIdx.x * blockDim.x)) * (2 * d);
	if ((k + (2 * d) - 1) < n) {
		f[offset + (k + (2 * d) - 1)] += f[offset+ (k + d - 1)];

		//x[n-1] <- 0
		if ((d == n / 2) && (k + (2 * d) == n))
			f[offset + (n - 1)] = 0;
	}
}

__global__ void down_sweep_kernel(unsigned int offset, unsigned int d, unsigned int n, unsigned int* NF)
{
	unsigned int k = (threadIdx.x + (blockIdx.x * blockDim.x)) * (2 * d);
	if ((k + (2 * d) - 1) < n) {
		double t = f[offset + (k + d - 1)];
		f[offset + (k + d - 1)] = f[offset + (k + (2 * d) - 1)];
		f[offset + (k + (2 * d) - 1)] += t;

		/*Add two last elements in e, f = total # of falses, set as shared variable NF*/
		if ((d == 1) && ((k + (2 * d) == n)))
			*NF = f[offset + (n - 1)] + (unsigned int)e[offset + (n - 1)];
	}
}

/*Note: scan_primitives works only with N in powers of 2*/
__device__ void scan_primitives(unsigned int offset, unsigned int n, unsigned int* NF, cudaStream_t nodeStream)
{
	unsigned int d, totalThreadNum, blockSz, gridSz;
	//reduce (up-sweep) phase
	for (d = 1; d <= n / 2; d *= 2) {
		totalThreadNum = (n - 1) / (2 * d) + 1;
		blockSz = (totalThreadNum < qsMaxThreadsPerBlock) ? totalThreadNum : qsMaxThreadsPerBlock;
		gridSz = (totalThreadNum + blockSz - 1) / blockSz;
		reduce_kernel <<<gridSz, blockSz, 0, nodeStream>>> (offset, d, n);
	}
	//down-sweep phase
	for (d = n / 2; d >= 1; d /= 2) {
		totalThreadNum = (n - 1) / (2 * d) + 1;
		blockSz = (totalThreadNum < qsMaxThreadsPerBlock) ? totalThreadNum : qsMaxThreadsPerBlock;
		gridSz = (totalThreadNum + blockSz - 1) / blockSz;
		down_sweep_kernel <<<gridSz, blockSz, 0, nodeStream>>> (offset, d, n, NF);
	}
}

/*STEP 3
*Each thread knows its id
*t = id - f + NF*/
__global__ void create_t_kernel(unsigned int offset, unsigned int length, unsigned int* NF)
{
	unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid < length)
		t[offset + tid] = tid - f[offset + tid] + *NF;
}

/*STEP 4
*addr = e ? f : t*/
__global__ void create_addr_kernel(unsigned int offset, unsigned int length)
{
	unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid < length)
		addr[offset + tid] = e[offset + tid] ? f[offset + tid] : t[offset + tid];
}

/*STEP 5
*out[addr] = in (scatter)*/
__global__ void swap_X_kernel(double* dataOut, double* dataIn, unsigned int offset, unsigned int length)
{
	unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid < length)
		dataOut[addr[offset + tid]] = dataIn[tid];
}

__global__ void swap_idx_kernel(unsigned int offset, unsigned int length)
{
	unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid < length)
		f[offset + addr[offset + tid]] = qsSwaps[offset + tid];
}
/*END*/

/*Returns the smallest power of two*/
__host__ __device__ unsigned int smallest_power_two(unsigned int n)
{
	unsigned int N = n;
	if ((N & (N - 1)) != 0) {	// fix if n is not power of 2
		N = 1;
		while (N < n)
			N <<= 1;
	}
	return N;
}

/*Fills out array with indexes starting from 0 to length*/
__global__ void create_indexes_kernel(unsigned int* out, unsigned int length)
{
	unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid < length)
		out[tid] = tid;
}

__global__ void qs_init_kernel(double* d_qsMain, double* d_qsAux, unsigned int* d_swaps, unsigned int* d_f, unsigned int* d_t, unsigned int* d_addr, unsigned int* d_NFs, char* d_e, unsigned int maxThreadsPerBlock)
{
	if ((threadIdx.x) == 0 && (blockIdx.x == 0)) {
		qsMaxThreadsPerBlock = maxThreadsPerBlock;
		//Initialize device pointers to global memory
		qsMain = d_qsMain;
		qsAux = d_qsAux;
		qsSwaps = d_swaps;
		f = d_f;
		t = d_t;
		addr = d_addr;
		NFs = d_NFs;
		e = d_e;
	}
}

__device__ double quick_select(unsigned int nodeOffset, unsigned int nodeLength, unsigned int nodeK, cudaStream_t nodeStream)
{
	double* dataIn, * dataOut;	//pointers to swap data
	char depth;
	int lengthDiff;
	unsigned int offset, length, * NF, pvIdx, k, fixedLength;
	
	//Create indexes for the after swap
	unsigned int blockSz = (nodeLength < qsMaxThreadsPerBlock) ? nodeLength : qsMaxThreadsPerBlock;
	unsigned int gridSz = (nodeLength + blockSz - 1) / blockSz;
	create_indexes_kernel <<<gridSz, blockSz, 0, nodeStream>>> (qsSwaps + nodeOffset, nodeLength);
	if (nodeLength == 1) { cudaDeviceSynchronize(); return qsMain[nodeOffset]; }

	offset = nodeOffset; length = nodeLength;
	NF = NFs + (threadIdx.x + (blockIdx.x * blockDim.x));
	depth = 0; pvIdx = 0;	k = nodeK;
	while (length > 1) {
		/*STEP 0 Find where data is*/
		if (depth)	{ dataIn = qsAux + offset;		dataOut = qsMain + offset; }
		else		{ dataIn = qsMain + offset;		dataOut = qsAux + offset; }
		blockSz = (length < qsMaxThreadsPerBlock) ? length : qsMaxThreadsPerBlock;
		gridSz = (length + blockSz - 1) / blockSz;

		/*STEP 1 Distribute pivot across segment and compare input with pivot*/
		compare_pivot_kernel <<<gridSz, blockSz, 0, nodeStream>>> (dataIn, offset, length, pvIdx, depth);
		
		/*STEP 2 Enumerate with false=1*/
		scan_primitives(offset, length, NF, nodeStream);
		
		/*STEP 3 Each thread knows its id t = id - f + NF*/
		create_t_kernel <<<gridSz, blockSz, 0, nodeStream>>> (offset, length, NF);
		
		/*STEP 4 addr = e ? f : t*/
		create_addr_kernel <<<gridSz, blockSz, 0, nodeStream>>> (offset, length);
		
		/*STEP 5 out[addr] = in (scatter)*/
		swap_X_kernel <<<gridSz, blockSz, 0, nodeStream>>> (dataOut, dataIn, offset, length);
		swap_idx_kernel <<<gridSz, blockSz, 0, nodeStream>>> (offset, length);
		cudaMemcpyAsync(qsSwaps + offset, f + offset, length * sizeof(unsigned int), cudaMemcpyDeviceToDevice, nodeStream);
		//Wait calculations to be done
		cudaDeviceSynchronize();
		
		/*STEP 6 continue to next segment*/
		depth = !depth;
		if (k < *NF) {	//k is in the left segment
			fixedLength = smallest_power_two(*NF);
			cudaMemcpyAsync(dataIn + *NF, dataOut + *NF, (fixedLength - *NF) * sizeof(double), cudaMemcpyDeviceToDevice, nodeStream);
			length = fixedLength;	pvIdx = 0;
		}
		else if ((k > *NF) || ((k == *NF) && (depth))) {	//k is in the right segment
			fixedLength = smallest_power_two(length - *NF);
			lengthDiff = length - fixedLength;
			cudaMemcpyAsync(dataIn + lengthDiff, dataOut + lengthDiff, (*NF - lengthDiff) * sizeof(double), cudaMemcpyDeviceToDevice, nodeStream);
			offset += lengthDiff;	length = fixedLength;	k = k - lengthDiff;		pvIdx = *NF - lengthDiff;
		}
		else
			break;
	}
	if (depth)	//return the k-th value
		return qsAux[nodeOffset + nodeK];
	else
		return qsMain[nodeOffset + nodeK];
}


