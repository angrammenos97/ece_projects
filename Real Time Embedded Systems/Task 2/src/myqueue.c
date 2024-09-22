#include "myqueue.h"
#include <stdlib.h>

queue *queueInit(int qsize)
{
	queue *q;
	q = (queue *)malloc(sizeof(queue));
	if (q == NULL) return (NULL);

	queuesize = qsize;
	q->empty = 1;
	q->full = 0;
	q->head = 0;
	q->tail = 0;
	q->buf = (void**)malloc(queuesize * sizeof(void*));
	q->mut = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(q->mut, NULL);
	q->notFull = (pthread_cond_t *)malloc(sizeof(pthread_cond_t));
	pthread_cond_init(q->notFull, NULL);
	q->notEmpty = (pthread_cond_t *)malloc(sizeof(pthread_cond_t));
	pthread_cond_init(q->notEmpty, NULL);

	return(q);
}

void queueDelete(queue *q)
{
	free(q->buf);
	pthread_mutex_destroy(q->mut);
	free(q->mut);
	pthread_cond_destroy(q->notFull);
	free(q->notFull);
	pthread_cond_destroy(q->notEmpty);
	free(q->notEmpty);
	free(q);
}

void queueAdd(queue *q, void *in)
{
	q->buf[q->tail] = in;
	q->tail++;
	if (q->tail == queuesize)
		q->tail = 0;
	if (q->tail == q->head)
		q->full = 1;
	q->empty = 0;	
	return;
}

void queueDel(queue *q, void **out)
{
	*out = q->buf[q->head];
	q->head++;
	if (q->head == queuesize)
		q->head = 0;
	if (q->head == q->tail)
		q->empty = 1;
	q->full = 0;
	return;
}
