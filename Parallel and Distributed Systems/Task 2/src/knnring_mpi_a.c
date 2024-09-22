#include "knnring.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cblas.h"
#include "mpi.h"


void blas_distXY(double *D, double *X, double *Y, int n, int m, int d)
{
	// Calculate X.X**T
	double *dot_X = (double*)malloc(n * sizeof(double));
	for (int i = 0; i < n; i++)
		*(dot_X + i) = cblas_ddot(d, (X + i * d), 1, (X + i * d), 1);
	// Calculate Y.Y**T
	double *dot_Y = (double*)malloc(m * sizeof(double));
	for (int i = 0; i < m; i++)
		*(dot_Y + i) = cblas_ddot(d, (Y + i * d), 1, (Y + i * d), 1);
	// Calculate (X.X**T + Y.Y**T)
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			*(D + i * n + j) = *(dot_Y + i) + *(dot_X + j);
	// Calculate 2.Y.X**T and all distances
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, d, -2.0, Y, d, X, d, 1.0, D, n);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) {
			if (*(D + i * n + j) < 1e-12)	// threshold min value for openblas
				*(D + i * n + j) = 0.0;
			else
				*(D + i * n + j) = sqrt(*(D + i * n + j));
		}
	free(dot_Y);
	free(dot_X);
}

void SWAP(double *d, int *idx, int a, int b)
{
	double tmpd;
	tmpd = *(d + a);
	*(d + a) = *(d + b);
	*(d + b) = tmpd;
	int tmpi = *(idx + a);
	*(idx + a) = *(idx + b);
	*(idx + b) = tmpi;
}

double quick_select(double *d, int *idx, int len, int k)
{
	int i, st;
	for (st = i = 0; i < len - 1; i++) {
		if (d[i] > d[len - 1]) continue;
		SWAP(d, idx, i, st);
		st++;
	}
	SWAP(d, idx, len - 1, st);
	return k == st ? d[st]
		: st > k ? quick_select(d, idx, st, k)
		: quick_select(d + st, idx + st, len - st, k - st);
}

void my_qsort(double *d, int *idx, int len, int k)
{
	quick_select(d, idx, len, k);			// put first the k shortest
	for (int i = 0; i < k + 1; i++)			// and after short them
		quick_select(d, idx, k + 1, i);
}

void add_idx_offset(int offset, int *idx, int n, int k)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < k; j++)
			*(idx + i * k + j) += offset;
}

void kNNv(knnresult *result, double *X, double  *Y, int n, int m, int d, int k)
{
	double *distxy = (double*)calloc(m*n, sizeof(double));
	int *idx = (int*)malloc(n * sizeof(int));
	blas_distXY(distxy, X, Y, n, m, d); // find all distances
	// Find k neighbors for every query
	for (int q = 0; q < m; q++) {	// all query		
		for (int c = 0; c < n; c++)	// create index array
			*(idx + c) = c;
		my_qsort((distxy + q * n), idx, n, k);
		for (int z = 0; z < k; z++) {	// add to struct the k neighbors
			*(result->nidx + q * k + z) = *(idx + z);
			*(result->ndist + q * k + z) = *(distxy + q * n + z);
		}
	}
	free(idx);
	free(distxy);
}

knnresult distrAllkNN(double *Y, int n, int d, int k)
{
	int p, id;							// # processess and PID
	MPI_Comm_rank(MPI_COMM_WORLD, &id); // Task ID
	MPI_Comm_size(MPI_COMM_WORLD, &p);	// # tasks

	// Where receive? Where send?
	int source = (id == 0) ? p - 1 : id - 1;	// if id == 0 take from last proccess
	int dest = (id == p - 1) ? 0 : id + 1;		// if last proccess send to first proccess
	int tag = (int)'X';							// mpi tag to send X data points set
	const int num_of_requests = 2;				// mpi asychronous # of status for request
	MPI_Status Stat[num_of_requests];			// mpi asychronous receive status	
	MPI_Request mpi_requests[num_of_requests];	// mpi asychronou request status

	// Allocate memory for the jobs
	int *npidx = (int*)malloc(n * (2 * k) * sizeof(int));
	double *npdist = (double*)malloc(n * (2 * k) * sizeof(double));
	double *X = (double*)malloc(n * d * sizeof(double));
	memcpy(X, Y, n * d * sizeof(double));		// your chunk are the first points to check
	double *X_recv = (double*)malloc(n * d * sizeof(double));
	knnresult new_knnresult;
	new_knnresult.m = n;
	new_knnresult.k = k;
	new_knnresult.nidx = (int*)malloc(n * k * sizeof(int));
	new_knnresult.ndist = (double*)malloc(n * k * sizeof(double));

	// Move points into circle
	for (int ip = 0; ip < p; ip++) {
		if (ip < p - 1) {	// don't send data at the last loop, the job was done at the first
			// Send and receive data asychronous from other proccess in the ring
			MPI_Isend(X, n*d, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &mpi_requests[0]);
			MPI_Irecv(X_recv, n*d, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &mpi_requests[1]);

		}
		if (ip == 0) {	// at first loop you already have data			
			kNNv(&new_knnresult, X, Y, n, n, d, k);
			int offset = (id >= ip + 1) ? (id - ip - 1) * n : (p + id - ip - 1) * n;
			add_idx_offset(offset, new_knnresult.nidx, n, k);				// offset idx depending of the chunk
			for (int i = 0; i < n; i++) {
				memcpy((npidx + i * (2 * k)), (new_knnresult.nidx + i * k), k * sizeof(int));
				memcpy((npdist + i * (2 * k)), (new_knnresult.ndist + i * k), k * sizeof(double));
			}
		}
		else {	// work with received data
			kNNv(&new_knnresult, X, Y, n, n, d, k);
			int offset = (id >= ip + 1) ? (id - ip - 1) * n : (p + id - ip - 1) * n;
			add_idx_offset(offset, new_knnresult.nidx, n, k);				// offset idx depending of the chunk
			for (int i = 0; i < n; i++) {
				memcpy((npidx + i * (2 * k) + k), (new_knnresult.nidx + i * k), k * sizeof(int));
				memcpy((npdist + i * (2 * k) + k), (new_knnresult.ndist + i * k), k * sizeof(double));
				my_qsort((npdist + i * (2 * k)), (npidx + i * (2 * k)), 2 * k, k);
			}
		}
		if (ip < p - 1) {
			MPI_Waitall(num_of_requests, mpi_requests, Stat);	// wait Isent and Irecv
			// Swap pointers
			double *tmpdptr = X;
			X = X_recv;
			X_recv = tmpdptr;
		}
	}

	free(X_recv);
	free(X);

	knnresult final_knnresult;
	final_knnresult.nidx = (int*)malloc(n * k * sizeof(int));
	final_knnresult.ndist = (double*)malloc(n * k * sizeof(double));
	for (int i = 0; i < n; i++) {	// copy results to return structure
		memcpy((final_knnresult.nidx + i * k), (npidx + i * (2 * k)), k * sizeof(int));
		memcpy((final_knnresult.ndist + i * k), (npdist + i * (2 * k)), k * sizeof(double));
	}
	final_knnresult.m = n;
	final_knnresult.k = k;

	free(npidx);
	free(npdist);

	return final_knnresult;
}