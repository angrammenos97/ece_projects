#include "vptree.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

/////////////////////////////////
double *distance_from_last(double *X, int n, int dim)
{
	if (n == 1)
		exit(-1);
	double *d = (double*)malloc((n - 1) * sizeof(double));
	cilk_for (int i = 0; i < n - 1; i++) {
		*(d + i) = 0.0;
		for (int j = 0; j < dim; j++)
			*(d + i) += pow(*(X + i * dim + j) - *(X + (n - 1) * dim + j), 2);
		*(d + i) = sqrt(*(d + i));
	}
	return d;
}

void SWAP(double *X, double *d, int *idx, int dim, int a, int b)
{
	double tmpd;
	for (int j = 0; j < dim; j++) {
		tmpd = *(X + a * dim + j);
		*(X + a * dim + j) = *(X + b * dim + j);
		*(X + b * dim + j) = tmpd;
	}
	tmpd = *(d + a);
	*(d + a) = *(d + b);
	*(d + b) = tmpd;
	int tmpi = *(idx + a);
	*(idx + a) = *(idx + b);
	*(idx + b) = tmpi;
}

double quick_select(double *d, double *X, int *idx, int len, int k, int dim)
{
	int i, st;
	for (st = i = 0; i < len - 1; i++) {
		if (d[i] > d[len - 1]) continue;
		SWAP(X, d, idx, dim, i, st);
		st++;
	}
	SWAP(X, d, idx, dim, len - 1, st);
	return k == st ? d[st]
		: st > k ? quick_select(d, X, idx, st, k, dim)
		: quick_select(d + st, X + st * dim, idx + st, len - st, k - st, dim);
}

double median(double *X, int *idx, int n, int dim)
{
	if (n == 1)
		return 0.0;
	double *d = distance_from_last(X, n, dim);
	double md = quick_select(d, X, idx, (n - 1), (n - 2) / 2, dim);
	free(d);
	return md;
}

vptree *vpt(double *X, int *idx, int n, int dim)
{
	if (n == 0)
		return NULL;
	vptree *tree = (vptree*)malloc(sizeof(vptree));
	tree->vp = (X + (n - 1) * dim);
	tree->md = median(X, idx, n, dim);
	tree->idx = *(idx + n - 1);
	// split and recurse
	if ((n - 1) % 2 == 0) {
		tree->inner = cilk_spawn vpt(X, idx, (n - 1) / 2, dim);
		tree->outer = vpt((X + ((n - 1) / 2)*dim), (idx + (n - 1) / 2), (n - 1) / 2, dim);
		cilk_sync;
	}
	else {
		tree->inner = cilk_spawn vpt(X, idx, (n - 1) / 2 + 1, dim);
		tree->outer = vpt((X + ((n - 1) / 2 + 1)*dim), (idx + (n - 1) / 2 + 1), (n - 1) / 2, dim);
		cilk_sync;
	}
	return tree;
}
/////////////////////////////////

vptree * buildvp(double *X, int n, int d)
{
	double *X_copy = (double*)malloc(n * d * sizeof(double));
	int *idx = (int*)malloc(n * sizeof(int));
	for (int i = 0; i < n; i++) {
		*(idx + i) = i;
		for (int j = 0; j < d; j++)
			*(X_copy + i * d + j) = *(X + i * d + j);
	}

	vptree *root = vpt(X_copy, idx, n, d);

	return root;
}

vptree * getInner(vptree * T)
{
	return T->inner;
}
vptree * getOuter(vptree * T)
{
	return T->outer;
}
double getMD(vptree * T)
{
	return T->md;
}
double * getVP(vptree * T)
{
	return T->vp;
}
int getIDX(vptree * T)
{
	return T->idx;
}
