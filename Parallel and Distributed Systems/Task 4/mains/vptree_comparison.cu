#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "vptree.h"

#define DefaultNumPoints 100
#define DefaultDim 4

unsigned int nop = DefaultNumPoints;
unsigned int dim = DefaultDim;

void help(int argc, char* argv[]);

int main(int argc, char* argv[])
{
	help(argc, argv);

	srand((unsigned int)time(NULL));
	double* points = (double*)malloc(nop * dim * sizeof(double));
	clock_t start_t, end_t;

	printf("Generating random points... ");
	start_t = clock();
	for (unsigned int i = 0; i < nop * dim; i++)
		points[i] = (double)rand() / RAND_MAX;
	end_t = clock();
	printf("DONE in %lfmsec!\n", ((double)(end_t - start_t) / CLOCKS_PER_SEC) * 1000.0);

	printf("Building tree in CPU... ");
	vptree* root_cpu = buildvp_cpu(points, nop, dim);

	printf("Building tree in GPU... ");
	vptree* root_gpu = buildvp_gpu(points, nop, dim);

	printf("\n");

	free(points);
	return 0;
}

void help(int argc, char* argv[])
{
	if (argc > 1) {
		for (int i = 1; i < argc; i += 2) {
			if (*argv[i] == '-') {
				if (*(argv[i] + 1) == 'n')
					nop = atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 'd')
					dim = atoi(argv[i + 1]);
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
	printf("-n [Number] :Number of points (default:%i)\n", DefaultNumPoints);
	printf("-d [Dimension] :Dimension of the space (default: %i)\n", DefaultDim);
}
 