#include "timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include "myqueue.h"
#include "prod_cons.h"
#include "jobs.h"

#ifdef LOGGER
	#include <pthread.h>
	#include "logger.h"
	extern logFile *glLog;  //global logger
#endif

extern unsigned int activeprods;
extern unsigned int activecons;
extern pthread_mutex_t activeprods_mut;
extern queue *fifo;

unsigned int eltime2us(struct timeval *sttime, struct timeval *entime);
void custPrint_timer(char *msg);

mytimer* timer_init(uint period, uint ttoexec, uint stdelay, void* stfcn, void* spfcn, void* tfcn, void* erfcn, void* udata)
{
    mytimer *t = (mytimer*)malloc(sizeof(mytimer));
    t->period = period;
    t->taskstoexecute = ttoexec;
    t->startdelay = stdelay;
    t->startfcn = stfcn;
    t->stopfcn = spfcn;
    t->timerfcn = tfcn;
    t->errorfcn = erfcn;
    t->userdata = udata;
    return t;
}

mytimerplus *timerplus_init(mytimer *t, int y, int m, int d, int h, int min, int sec)
{
    mytimerplus *tplus = (mytimerplus*)malloc(sizeof(mytimerplus));
    tplus->t = t;
    tplus->y = y;
    tplus->m = m;
    tplus->d = d;
    tplus->h = h;
    tplus->min = min;
    tplus->sec = sec;
}

void *start(void *args)
{
    mytimer *t = (mytimer*)args;    
    workFunction *job = NULL;

    int status;        //status of producer
    int corrperiod;    //corrected period
    long int threadpid = syscall(__NR_gettid);; //pid of the process
    struct timeval starttime, endtime;
    char msg[64];

    sprintf(msg, "Starting job(%li|%p)\n", threadpid, t->timerfcn);
    custPrint_timer(msg);
    if (t->startfcn != NULL)        //run start function
        (t->startfcn)(t->userdata);

    sleep(t->startdelay);   //apply start delay
    for (int i = 0; i < t->taskstoexecute; i++) {
        job = (workFunction*)malloc(sizeof(workFunction));
        job->work = t->timerfcn;
        job->arg = t->userdata;
        sprintf(job->jobid, "%d:%li", threadpid, i+1);
        gettimeofday(&starttime, NULL);
        status = (int)(producer(job));        
        if (status == -1)
            if (t->errorfcn != NULL)
                (t->errorfcn)(t->userdata);
        
        if (i == t->taskstoexecute - 1){    
	        pthread_mutex_lock(&activeprods_mut);
	        activeprods--;
            pthread_mutex_unlock(&activeprods_mut);
            pthread_cond_signal(fifo->notEmpty);
        }
        gettimeofday(&endtime, NULL);
        corrperiod = (t->period*1000) - eltime2us(&starttime, &endtime);    //correct period

	    if (corrperiod < 0){
            sprintf(msg, "WARNING(%li|%p): Needed %d more usec!\n", threadpid, t->timerfcn, -corrperiod);
            custPrint_timer(msg);
        }
        else
            usleep(corrperiod);
    }
    while (!fifo->empty || activecons){}    //wait all consumers to end
    if (t->stopfcn != NULL)                 //run stop function
        (t->stopfcn)(t->userdata);
    free(t->userdata);
    return NULL;
}

void *startat(void *arg)
{
    mytimerplus *t = (mytimerplus*)arg;
    struct tm destime;  //desired start time
    destime.tm_year = t->y - 1900;
    destime.tm_mon = t->m - 1;
    destime.tm_mday = t->d;
    destime.tm_hour = t->h;
    destime.tm_min = t->min;
    destime.tm_sec = t->sec;
    destime.tm_isdst = -1;  //unknown    
    mytimer *tmp = t->t;
    char msg[32];
    time_t alarmstart = mktime(&destime)-time(NULL);
    if (alarmstart < 0) {
        sprintf(msg, "WARNING(%u|%p): Date passed %d seconds ago!\n", getpid(), tmp->timerfcn, -alarmstart);
        custPrint_timer(msg);
    }
    else {
        sprintf(msg, "Scheduled job(%d|%p)!\n", getpid(), tmp->timerfcn);
        custPrint_timer(msg);
        sleep( mktime(&destime)-time(NULL) );
    }
    start(t->t);
    return NULL;
}

unsigned int eltime2us(struct timeval *sttime, struct timeval *entime)
{
    uint sec = entime->tv_sec - sttime->tv_sec;
    uint usec = entime->tv_usec - sttime->tv_usec;
    return (uint)((sec*1000000) + usec);
}

void custPrint_timer(char *msg)
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