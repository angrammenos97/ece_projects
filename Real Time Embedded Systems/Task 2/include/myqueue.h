#ifndef MYQUEUE_H_
#define MYQUEUE_H_

#include <pthread.h>

static int queuesize;

typedef struct queue{
	void **buf;
	long head, tail;
	int full, empty;
	pthread_mutex_t *mut;
	pthread_cond_t *notFull, *notEmpty;
} queue;

queue *queueInit(int qsize);

void queueDelete(queue *q);

void queueAdd(queue *q, void *in);

void queueDel(queue *q, void **out);

#endif // !MYQUEUE_H_
