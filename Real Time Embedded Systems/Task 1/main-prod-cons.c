#define HAVE_STRUCT_TIMESPEC

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "sys/time.h"
#include "unistd.h"
#include "myqueue.h"

#define DefNumProd 1	// Default Number of Producers
#define DefNumCons 1	// Default Number of Consumers
#define DefNumLoop 20	// Default Number of Job's loop
#define DefQueueSz 10	// Default Queue size
#define DefResults 0	// Default to export data to file

void help(int argc, char *argv[]);
void *producer(void *args);
void *consumer(void *args);

int pnum = DefNumProd;
int cnum = DefNumCons;
int lnum = DefNumLoop;
int qnum = DefQueueSz;
int resexport = DefResults;
FILE *results;

int activeprods;	// number of active producers
double totalwaittime = 0.0;	// total time waited in seconds

int main(int argc, char *argv[])
{
	srand((unsigned int)time(NULL));
	help(argc, argv);

	totalwaittime = 0.0;	// reset time

	printf("\nRunning with %d producers, %d consumers for %d loops with %d queue size.\n", pnum, cnum, lnum, qnum);

	queue *fifo;
	pthread_t *pro = (pthread_t*)malloc(pnum * sizeof(pthread_t));
	pthread_t *con = (pthread_t*)malloc(cnum * sizeof(pthread_t));

	fifo = queueInit(qnum);	// Initialize queue
	if (fifo == NULL) {
		fprintf(stderr, "main: Queue Init failed.\n");
		exit(1);
	}

	for (int i = 0; i < pnum; i++) {	// Create producers
		pthread_create(pro + i, NULL, producer, fifo);
		activeprods++;	// one more producer is active
	}
	for (int i = 0; i < cnum; i++)	// Create consumers
		pthread_create(con + i, NULL, consumer, fifo);

	printf("Producers and Consumers created. Doing jobs...\n");
	for (int i = 0; i < pnum; i++) 	// Wait all producers to end
		pthread_join(*(pro + i), NULL);
	for (int i = 0; i < cnum; i++) {// Wait all consumers to end
		pthread_join(*(con + i), NULL);
	}
	printf("DONE in average wait time %fsec!\n", totalwaittime / (pnum*lnum));	// total time over total jobs(prod*loops)

	queueDelete(fifo);
	free(con);
	free(pro);

	if (resexport) {
		results = fopen("./results.txt", "a");
		fprintf(results, "%d\t%d\t%d\t%d\t%lf\n", qnum, lnum, pnum, cnum, totalwaittime / (pnum*lnum));
		fclose(results);
	}

	return 0;
}

void help(int argc, char *argv[])
{
	if (argc > 1) {
		for (int i = 1; i < argc; i += 2) {
			if (*argv[i] == '-') {
				if (*(argv[i] + 1) == 'p')
					pnum = atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 'c')
					cnum =  atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 'q')
					qnum = atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 'o')
					resexport = atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 'l') {
					lnum = atoi(argv[i + 1]);
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
	printf("-p [Number]\t:Max number of producers (default: %i)\n", DefNumProd);
	printf("-c [Number]\t:Max number of consumers (default: %i)\n", DefNumCons);
	printf("-l [Number]\t:Max number of job's loop (default: %i)\n", DefNumLoop);
	printf("-q [Number]\t:Max size of the queue (default: %i)\n", DefQueueSz);
	printf("-o [0|1]\t:Export results to results.txt (default: %i)\n", DefResults);
}

void *producer(void *q)
{
	queue *fifo;
	int i;

	fifo = (queue *)q;

	for (i = 0; i < lnum; i++) {
		pthread_mutex_lock(fifo->mut);
		while (fifo->full) {
			//printf("producer: queue FULL.\n");
			pthread_cond_wait(fifo->notFull, fifo->mut);
		}
		if (i == lnum - 1)	// this is your last loop
			activeprods--;
		queueAdd(fifo);		// add that job to the queue
		pthread_mutex_unlock(fifo->mut);
		pthread_cond_signal(fifo->notEmpty);
	}
	return (NULL);
}

void *consumer(void *q)
{
	queue *fifo = (queue *)q;
	workFunction *d;

	while (1) {
		pthread_mutex_lock(fifo->mut);
		while (fifo->empty && activeprods) {			
			//printf("consumer: queue EMPTY.\n");
			pthread_cond_wait(fifo->notEmpty, fifo->mut);			
		}
		if (fifo->empty && !activeprods) {	// if no other job is going to be added
			pthread_mutex_unlock(fifo->mut);		// give back the key 
			pthread_cond_signal(fifo->notEmpty);	// wake other consumer
			break;
		}
		queueDel(fifo, &d);			// grab a job from queue
		funcArgs *args = (funcArgs*)d->arg;
		totalwaittime += timeval_to_sec(*args->deltime) - timeval_to_sec(*args->addtime);	// add wait time in total
		pthread_mutex_unlock(fifo->mut);		
		pthread_cond_signal(fifo->notFull);

		(d->work)(d->arg);		// execute the job
		
		freeArgs((funcArgs*)d->arg);
		free(d);
	}
	return NULL;
}
