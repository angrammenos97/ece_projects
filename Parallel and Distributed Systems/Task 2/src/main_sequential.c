#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "sys/time.h"
#include "tester_helper.h"
#include "knnring.h"

#define DefaultNumCorpusPoints 897
#define DefaultNumQueryPoints 762
#define DefaultDim 7
#define DefaultNumNeighbors 13
#define MatrixOrder ROWMAJOR	// COLMAJOR | ROWMAJOR | 2 = Check both
#define DefaultValidateResults 1
#define DefaultSelf 0	// use as query the corpus

int n = DefaultNumCorpusPoints;
int m = DefaultNumQueryPoints;
int d = DefaultDim;
int k = DefaultNumNeighbors;
int ap = MatrixOrder;
int vr = DefaultValidateResults;
int self_q = DefaultSelf;

struct timeval startwtime, endwtime;

void help(int argc, char *argv[]);

int main(int argc, char *argv[])
{
	help(argc, argv);
	printf("Running with values n=%i, m=%i, d=%i, k=%i.\n", n, m, d, k);

	srand((unsigned int)time(NULL));

	// Generate random point set
	printf("Generating random data set. ");
	gettimeofday(&startwtime, NULL);
	double *corpus = (double *)malloc(n * d * sizeof(double));
	double *query = corpus;
	for (int i = 0; i < n*d; i++)
		*(corpus + i) = ((double)rand() / (RAND_MAX));
	if (!self_q) {
		query = (double *)malloc(m * d * sizeof(double));
		for (int i = 0; i < m*d; i++)
			*(query + i) = ((double)rand() / (RAND_MAX));
	}
	gettimeofday(&endwtime, NULL);
	double p_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
	printf("DONE in %fsec!\n", p_time);

	// Find neighbors
	printf("Finding all %i neighbors for all %i query points. ", k, m);
	knnresult knnres;
	gettimeofday(&startwtime, NULL);
	if (!self_q)
		knnres = kNN(corpus, query, n, m, d, k);
	else
		knnres = distrAllkNN(corpus, n, d, k);
	gettimeofday(&endwtime, NULL);
	p_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
	printf("DONE in %fsec!\n", p_time);

	if (vr) {
		// Validate results
		printf("Validating results. ");
		gettimeofday(&startwtime, NULL);
		int isValid = (MatrixOrder == 2) ? validateResult(knnres, corpus, query, n, m, d, k, COLMAJOR) || validateResult(knnres, corpus, query, n, m, d, k, ROWMAJOR)
			: validateResult(knnres, corpus, query, n, m, d, k, ap);
		printf("Tester validation: %s NEIGHBORS. ", STR_CORRECT_WRONG[isValid]);
		gettimeofday(&endwtime, NULL);
		p_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
		printf("DONE in %fsec!\n", p_time);
	}
	printf("Exiting\n");
	if (!self_q)
		free(query);
	free(corpus);
	return 0;
}


void help(int argc, char *argv[])
{
	if (argc > 1) {
		for (int i = 1; i < argc; i += 2) {
			if (*argv[i] == '-') {
				if (*(argv[i] + 1) == 'n')
					n = atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 'm')
					m = atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 'd')
					d = atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 'k')
					k = atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 'o')
					ap = atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 'v')
					vr = atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 's') {
					self_q = atoi(argv[i + 1]);
					m = n;
				}
				else {
					help(1, argv);
					return;
				}
			}
			else {
				help(1, argv);
				return;
			}
		}
		return;
	}
	printf("Flags to use:\n");
	printf("-n [Number]\t:Number of corpus points (default:%i)\n", DefaultNumCorpusPoints);
	printf("-m [Number]\t:Number of query points (default:%i)\n", DefaultNumQueryPoints);
	printf("-k [Number]\t:Number of neighbors (default:%i)\n", DefaultNumNeighbors);
	printf("-d [Dimension]\t:Dimension of the space (default: %i)\n", DefaultDim);
	printf("-o [0|1|2]\t:Matrix order with 0=ColMajor|1=RowMajor|2=Check both (default:%i)\n", MatrixOrder);
	printf("-v [0|1]\t:Validate results with elearning tester (default:%i)\n", DefaultValidateResults);
	printf("-s [0|1]\t:Use as query the corpus (default:%i)\n", DefaultSelf);
}
