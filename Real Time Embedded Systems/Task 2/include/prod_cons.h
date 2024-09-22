#ifndef PROD_CONS_H_
#define PROD_CONS_H_

#include <pthread.h>

extern unsigned int activeprods;
extern unsigned int activecons;
extern pthread_mutex_t activeprods_mut;
extern pthread_mutex_t activecons_mut;

typedef struct workFunction {
	void* arg;
	void* (*work)(void *);
	char jobid[32];
} workFunction;

void *producer(void *args);

void *consumer(void *args);

#endif //PROD_CONS_H_