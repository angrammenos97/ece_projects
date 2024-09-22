#include "knnring.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cblas.h"


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

knnresult kNN(double *X, double  *Y, int n, int m, int d, int k)
{
	knnresult result;
	result.nidx = (int*)malloc(m * k * sizeof(int));
	result.ndist = (double*)malloc(m * k * sizeof(double));
	result.m = m;
	result.k = k;

	double *distxy = (double*)calloc(m*n, sizeof(double));
	int *idx = (int*)malloc(n * sizeof(int));
	blas_distXY(distxy, X, Y, n, m, d); // find all distances
	// Find k neighbors for every query
	for (int q = 0; q < m; q++) {	// all query
		for (int c = 0; c < n; c++)	// create index array
			*(idx + c) = c;
		my_qsort((distxy + q * n), idx, n, k);
		for (int z = 0; z < k; z++) {	// add to struct the k neighbors
			*(result.nidx + q * k + z) = *(idx + z);
			*(result.ndist + q * k + z) = *(distxy + q * n + z);
		}
	}
	free(idx);
	free(distxy);
	return result;
}

knnresult distrAllkNN(double *X, int n, int d, int k)
{
	return kNN(X, X, n, n, d, k);
}