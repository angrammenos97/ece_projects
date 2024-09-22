#include "prod_cons.h"
#include <stdio.h>
#include <stdlib.h>
#include "myqueue.h"
#include "jobs.h"

#ifdef LOGGER
	#include <pthread.h>
	#include "logger.h"
	extern logFile *glLog;  //global logger
#endif

extern queue *fifo;

void custPrint_prod_cons(char *msg);

void *producer(void *q)
{
	workFunction *job = (workFunction*)q;
	char msg[64];
    pthread_mutex_lock(fifo->mut);
	if (fifo->full) {		
		pthread_mutex_unlock(fifo->mut);
		pthread_cond_signal(fifo->notEmpty);
		sprintf(msg, "queue FULL, canceling (%p): %s.\n", job->work, job->jobid);
		custPrint_prod_cons(msg);
		return (void*)-1;
	}
	queueAdd(fifo, q);		// add that job to the queue
	pthread_mutex_unlock(fifo->mut);
	pthread_cond_signal(fifo->notEmpty);
	// pthread_mutex_lock(&activeprods_mut);
	// activeprods--;
    // pthread_mutex_unlock(&activeprods_mut);

	sprintf(msg, "Added (%p): %s\n", job->work, job->jobid);
	custPrint_prod_cons(msg);
	return NULL;
}

void *consumer(void *q)
{
	workFunction *job;
	char msg[64];

	while (1) {
		pthread_mutex_lock(fifo->mut);
		while (fifo->empty && activeprods) {
			sprintf(msg, "consumer: queue EMPTY (%d).\n", activeprods);			
			custPrint_prod_cons(msg);
			pthread_cond_wait(fifo->notEmpty, fifo->mut);			
		}
		if (fifo->empty && !activeprods) {	// if no other job is going to be added
			pthread_mutex_unlock(fifo->mut);		// give back the key 
			pthread_cond_signal(fifo->notEmpty);	// wake other consumer
			break;
		}
		queueDel(fifo, (void**)&job);			// grab a job from queue
		pthread_mutex_unlock(fifo->mut);		
		pthread_cond_signal(fifo->notFull);

		sprintf(msg, "Executing (%p): %s\n",job->work, job->jobid);
		custPrint_prod_cons(msg);
		(job->work)(job->arg);		// execute the job
		sprintf(msg, "Finished (%p): %s\n", job->work, job->jobid);
		custPrint_prod_cons(msg);
		free(job);
	}
	pthread_mutex_lock(&activecons_mut);
	activecons--;		//one more consumer dies
    pthread_mutex_unlock(&activecons_mut);
	return NULL;
}

void custPrint_prod_cons(char *msg)
{
#ifdef LOGGER
	if (glLog != NULL)
		log_write_r(glLog, msg);
	else
		printf("%s", msg);
#else
	printf("%s", msg);
#endif	
}