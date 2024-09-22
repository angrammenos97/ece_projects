#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "mpi.h"
#include "sys/time.h"
#include "tester_helper.h"
#include "knnring.h"

#define DefaultNumCorpusPoints 1423
#define DefaultDim 37
#define DefaultNumNeighbors 13
#define MatrixOrder ROWMAJOR	// COLMAJOR | ROWMAJOR | 2 = Check both
#define DefaultValidateResults 1


int n = DefaultNumCorpusPoints;
int d = DefaultDim;
int k = DefaultNumNeighbors;
int ap = MatrixOrder;
int vr = DefaultValidateResults;

struct timeval startwtime, endwtime;

void help(int argc, char *argv[], int id);
double * ralloc(int sz);
void testMPI(int const n, int const d, int const k, int const ap);

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);		// initialize MPI
	int id;						// # processess and PID
	MPI_Comm_rank(MPI_COMM_WORLD, &id); // Task ID
	help(argc, argv, id);
	if (id == 0)               // ..... MASTER		
		printf("Running with values n=%i, d=%i, k=%i.\n", n, d, k);

	if (ap == 2) {
		testMPI(n, d, k, COLMAJOR);
		testMPI(n, d, k, ROWMAJOR);
	}
	else
		testMPI(n, d, k, ap);

	printf("Exiting from proccess id %i.\n", id);
	MPI_Finalize();               // clean-up
	return 0;
}

void help(int argc, char *argv[], int id)
{
	if (argc > 1) {
		for (int i = 1; i < argc; i += 2) {
			if (*argv[i] == '-') {
				if (*(argv[i] + 1) == 'n')
					n = atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 'd')
					d = atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 'k')
					k = atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 'o')
					ap = atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 'v') {
					vr = atoi(argv[i + 1]);
				}
				else {
					help(1, argv, id);
					return;
				}
			}
			else {
				help(1, argv, id);
				return;
			}
		}
		return;
	}
	if (id == 0) {
		printf("Flags to use:\n");
		printf("-n [Number]\t:Number of corpus points (default:%i)\n", DefaultNumCorpusPoints);
		printf("-d [Dimension]\t:Dimension of the space (default: %i)\n", DefaultDim);
		printf("-k [Number]\t:Number of neighbors (default:%i)\n", DefaultNumNeighbors);
		printf("-o [0|1|2]\t:Matrix order with 0=ColMajor|1=RowMajor|2=Check both (default:%i)\n", ROWMAJOR);
		printf("-v [0|1]\t:Validate results with elearning tester (default:%i)\n", DefaultValidateResults);
	}
}

void testMPI(int const n, int const d, int const k, int const ap)
{
	int p, id;                    // MPI # processess and PID
	MPI_Status Stat;              // MPI status
	int dst, rcv, tag;            // MPI destination, receive, tag
	int isValid = 0;              // return value
	MPI_Comm_rank(MPI_COMM_WORLD, &id); // Task ID
	MPI_Comm_size(MPI_COMM_WORLD, &p);  // # tasks
	double p_time;		// job time

	// Allocate corpus for each process
	double * const corpus = (double *)malloc(n*d * sizeof(double));
	if (id == 0) {                //============================== MASTER
		printf("Generating random data set. ");
		gettimeofday(&startwtime, NULL);
		// ---------- Initialize data to begin with
		double const * const corpusAll = ralloc(n*d*p);
		// ---------- Break to subprocesses
		for (int ip = 0; ip < p; ip++) {
			for (int i = 0; i < n; i++)
				for (int j = 0; j < d; j++)
					if (ap == COLMAJOR)
						corpus_cm(i, j) = corpusAll_cm(i + ip * n, j);
					else
						corpus_rm(i, j) = corpusAll_rm(i + ip * n, j);
			if (ip == p - 1)            // last chunk is mine
				break;
			// which process to send? what tag?
			dst = ip + 1;
			tag = 1;
			// send to correct process
			MPI_Send(corpus, n*d, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD);
		} // for (ip)
		p_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
		printf("DONE in %fsec!\n", p_time);

		if (id == 0) {
			printf("Finding all %i neighbors for all %i query points. ", k, n * p);
			gettimeofday(&startwtime, NULL);
		}
		// ---------- Run distributed kNN
		knnresult const knnres = distrAllkNN(corpus, n, d, k);
		// ---------- Prepare global kNN result object
		knnresult knnresall;
		knnresall.nidx = (int *)malloc(n*p*k * sizeof(int));
		knnresall.ndist = (double *)malloc(n*p*k * sizeof(double));
		knnresall.m = n * p;
		knnresall.k = k;
		// ---------- Put my results to correct spot
		for (int j = 0; j < k; j++)
			for (int i = 0; i < n; i++) {
				if (ap == COLMAJOR) {
					knnresallnidx_cm(i + (p - 1)*n, j) = knnresnidx_cm(i, j);
					knnresallndist_cm(i + (p - 1)*n, j) = knnresndist_cm(i, j);
				}
				else {
					knnresallnidx_rm(i + (p - 1)*n, j) = knnresnidx_rm(i, j);
					knnresallndist_rm(i + (p - 1)*n, j) = knnresndist_rm(i, j);
				}
			}
		// ---------- Gather results back
		for (int ip = 0; ip < p - 1; ip++) {

			rcv = ip + 1;
			tag = 1;

			MPI_Recv(knnres.nidx, n*k, MPI_INT, rcv, tag, MPI_COMM_WORLD, &Stat);
			MPI_Recv(knnres.ndist, n*k, MPI_DOUBLE, rcv, tag, MPI_COMM_WORLD, &Stat);

			for (int j = 0; j < k; j++)
				for (int i = 0; i < n; i++) {
					if (ap == COLMAJOR) {
						knnresallnidx_cm(i + ip * n, j) = knnresnidx_cm(i, j);
						knnresallndist_cm(i + ip * n, j) = knnresndist_cm(i, j);
					}
					else {
						knnresallnidx_rm(i + ip * n, j) = knnresnidx_rm(i, j);
						knnresallndist_rm(i + ip * n, j) = knnresndist_rm(i, j);
					}
				}

		}
		gettimeofday(&endwtime, NULL);
		p_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
		printf("DONE in %fsec!\n", p_time);

		if (vr) {
			// ---------- Validate results
			printf("Validating results. ");
			gettimeofday(&startwtime, NULL);
			isValid = validateResult(knnresall, corpusAll, corpusAll, n*p, n*p, d, k, ap);
			gettimeofday(&endwtime, NULL);
			printf("Tester validation: %s NEIGHBORS. ", STR_CORRECT_WRONG[isValid]);
			p_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
			printf("DONE in %fsec!\n", p_time);
		}
	}
	else {                      //============================== SLAVE
   // ---------- Get data from MASTER
		rcv = 0;
		tag = 1;
		MPI_Recv(corpus, n*d, MPI_DOUBLE, rcv, tag, MPI_COMM_WORLD, &Stat);
		// ---------- Run distributed kNN
		knnresult const knnres = distrAllkNN(corpus, n, d, k);
		// ---------- Send data back to MASTER
		dst = 0;
		tag = 1;
		MPI_Send(knnres.nidx, n*k, MPI_INT, dst, tag, MPI_COMM_WORLD);
		MPI_Send(knnres.ndist, n*k, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD);
	}
	// ~~~~~~~~~~~~~~~~~~~~ Deallocate memory
	free(corpus);
}

double * ralloc(int sz)
{
	srand((unsigned int)time(NULL));
	double *X = (double *)malloc(sz * sizeof(double));
	for (int i = 0; i < sz; i++)
		X[i] = ((double)(rand())) / (double)RAND_MAX;
	return X;
}