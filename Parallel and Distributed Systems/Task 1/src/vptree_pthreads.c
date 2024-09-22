#include "vptree.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

#define BLOCK_SIZE 440000
#define MAX_THREADS 20
#define NOP_THRESHOLD 10000

typedef struct {
	double *X, *d; int n, dim, tid, nothd;
}dflargs;

typedef struct {
	double *X; vptree **tree; int *idx, n, dim;
}vptargs;

/////////////////////////////////
int nothvtp = 0;	// num of threads for vpt
pthread_attr_t attr;
pthread_mutex_t nothvtp_mutex;

void *distance_from_last_threated(void *arg)
{
	dflargs *data = (dflargs*)arg;
	if (data->n == 1)
		exit(-1);
	int block = (data->n - 1) / data->nothd;
	if (data->tid == data->nothd - 1)
		block += (data->n - 1) % data->nothd;		// the last thread maybe has extra work to do
	for (int i = 0; i < block; i++) {
		*(data->d + i) = 0.0;
		for (int j = 0; j < data->dim; j++)
			*(data->d + i) += pow(*(data->X + (data->tid * ((data->n - 1) / data->nothd) + i) * data->dim + j) - *(data->X + (data->n - 1) * data->dim + j), 2);
		*(data->d + i) = sqrt(*(data->d + i));
	}
	pthread_exit(NULL);
	return 0;
}

void distance_from_last_sequential(double *X, double *d, int n, int dim)
{
	if (n == 1)
		exit(-1);
	for (int i = 0; i < n - 1; i++) {
		*(d + i) = 0.0;
		for (int j = 0; j < dim; j++)
			*(d + i) += pow(*(X + i * dim + j) - *(X + (n - 1) * dim + j), 2);
		*(d + i) = sqrt(*(d + i));
	}
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
	double *d = (double*)malloc((n - 1) * sizeof(double));
	int nothd = (int)((n * dim) / (BLOCK_SIZE * nothvtp)) + 1; // plus 1 to do it sequential for small nop
	if (nothd == 1)
		distance_from_last_sequential(X, d, n, dim);
	else {
		pthread_t *thread = (pthread_t*)malloc(nothd * sizeof(pthread_t));
		dflargs *arg = (dflargs*)malloc(nothd * sizeof(dflargs));

		for (int t = 0; t < nothd; t++) {
			(arg + t)->X = X;		(arg + t)->d = (d + t * ((n - 1) / nothd));
			(arg + t)->n = n;		(arg + t)->dim = dim;		(arg + t)->tid = t;		(arg + t)->nothd = nothd;
			pthread_create(&thread[t], &attr, distance_from_last_threated, (void*)(arg + t));
		}
		for (int t = 0; t < nothd; t++)
			if (pthread_join(thread[t], NULL))
				printf("ERROR joining dist thread%i\n", (arg + t)->tid);
		free(thread);
	}
	double md = quick_select(d, X, idx, (n - 1), (n - 2) / 2, dim);
	free(d);
	return md;
}

vptree *vpt_sequential(double *X, int *idx, int n, int dim)
{
	if (n == 0)
		return NULL;
	vptree *tree = (vptree*)malloc(sizeof(vptree));
	tree->vp = (X + (n - 1) * dim);
	tree->md = median(X, idx, n, dim);
	tree->idx = *(idx + n - 1);
	// split and recurse
	if ((n - 1) % 2 == 0) {
		tree->inner = vpt_sequential(X, idx, (n - 1) / 2, dim);
		tree->outer = vpt_sequential((X + ((n - 1) / 2)*dim), (idx + (n - 1) / 2), (n - 1) / 2, dim);
	}
	else {
		tree->inner = vpt_sequential(X, idx, (n - 1) / 2 + 1, dim);
		tree->outer = vpt_sequential((X + ((n - 1) / 2 + 1)*dim), (idx + (n - 1) / 2 + 1), (n - 1) / 2, dim);
	}
	return tree;
}

void *vpt_threaded(void *arg)
{
	vptargs *data = (vptargs*)arg;
	if (data->n == 0) {
		*(data->tree) = NULL;
		return 0;
	}
	pthread_mutex_lock(&nothvtp_mutex);
	nothvtp++;	// one more thread created
	pthread_mutex_unlock(&nothvtp_mutex);
	vptree *node = (vptree*)malloc(sizeof(vptree));
	node->vp = (data->X + (data->n - 1) * data->dim);
	node->md = median(data->X, data->idx, data->n, data->dim);
	node->idx = *(data->idx + data->n - 1);
	*(data->tree) = node;
	// split and recurse
	vptargs vptargI, vptargO;
	vptargI.tree = &(node->inner);
	vptargO.tree = &(node->outer);
	vptargI.X = data->X;
	vptargI.idx = data->idx;
	vptargI.dim = vptargO.dim = data->dim;
	if ((data->n - 1) % 2 == 0) {
		vptargI.n = (data->n - 1) / 2;
		vptargO.X = (data->X + ((data->n - 1) / 2)*data->dim);
		vptargO.idx = (data->idx + (data->n - 1) / 2);
		vptargO.n = (data->n - 1) / 2;
	}
	else {
		vptargI.n = (data->n - 1) / 2 + 1;
		vptargO.X = (data->X + ((data->n - 1) / 2 + 1)*data->dim);
		vptargO.idx = (data->idx + (data->n - 1) / 2 + 1);
		vptargO.n = (data->n - 1) / 2;
	}
	if (nothvtp > MAX_THREADS || (data->n) < NOP_THRESHOLD) {		// continue sequential for now on
		node->inner = vpt_sequential(vptargI.X, vptargI.idx, vptargI.n, vptargI.dim);
		node->outer = vpt_sequential(vptargO.X, vptargO.idx, vptargO.n, vptargO.dim);
	}
	else {		// create new thread
		pthread_t thread;
		if (pthread_create(&thread, &attr, vpt_threaded, (void*)&vptargO)) {
			printf("ERROR: creating vpt thread");
			exit(-1);
		}
		vpt_threaded((void*)&vptargI);
		if (pthread_join(thread, NULL)) {
			printf("ERROR: joining vpt thread\n");
			exit(-1);
		}
	}

	pthread_mutex_lock(&nothvtp_mutex);
	nothvtp--;	// one more thread destroyed
	pthread_mutex_unlock(&nothvtp_mutex);

	pthread_exit(NULL);
	return 0;
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

	pthread_t rthread;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	pthread_mutex_init(&nothvtp_mutex, NULL);

	vptree *tree = NULL;
	vptargs root;
	root.X = X_copy;
	root.tree = &tree;
	root.idx = idx;
	root.n = n;
	root.dim = d;
	if (pthread_create(&rthread, &attr, vpt_threaded, (void*)&root)) {
		printf("ERROR: creating vpt thread");
		exit(-1);
	}
	if (pthread_join(rthread, NULL)) {
		printf("ERROR: joining vpt thread\n");
		exit(-1);
	}

	pthread_attr_destroy(&attr);
	pthread_mutex_destroy(&nothvtp_mutex);
	return tree;
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

