#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include "myqueue.h"
#include "jobs.h"
#include "prod_cons.h"
#include "timer.h"
#include "logger.h"
#include "conf.h"

confFile *configuration;
pthread_t *prod, *cons;
unsigned int activeprods;
unsigned int activecons;
pthread_mutex_t activeprods_mut;
pthread_mutex_t activecons_mut;
queue *fifo = NULL;    //global fifo
logFile *glLog = NULL; //global logger

void closeall(int sg);
void custPrint_main(char *msg);

int main(int argc, char *argv[])
{
    srand((unsigned int)time(NULL));  
    configuration = export_conf(argc, argv);
#ifdef LOGGER
    if (configuration->uselog)
        glLog = logfile_init(configuration->logfname, configuration->livelog);
#endif  
    fifo = queueInit(configuration->queuesize);
    prod = (pthread_t *)malloc(configuration->producersnum * sizeof(pthread_t));
    cons = (pthread_t *)malloc(configuration->consumersnum * sizeof(pthread_t));
    activeprods = 0;
    activecons = 0;
    pthread_mutex_init(&activeprods_mut, NULL);
    pthread_mutex_init(&activecons_mut, NULL);
    signal(SIGINT, closeall);   //close all safely

    char tmpmsg[16], msg[256];
    sprintf(msg, "Prod = %d | Cons = %d | Qsize = %d\n", configuration->producersnum, configuration->consumersnum, configuration->queuesize);
    custPrint_main(msg);
    sprintf(msg, "Addresses of jobs:");
    for (int i = 0; i < JOBSNUMBER; i++){
        sprintf(tmpmsg, " %s=%p |", jobnames[i], jobs[i]),
        strcat(msg, tmpmsg);
    }
    strcat(msg, "\n");
    custPrint_main(msg);
    int jobID; //ID of the job to add
    mytimer *t = NULL;
    void *udata;
    for (int p = 0; p < configuration->producersnum; p++) {
        jobID = configuration->jobs[p].jobid;
        udata = (void *)calloc(1 , udatasz[jobID]);
        t = timer_init(configuration->jobs[p].period, configuration->jobs[p].numofexec, 
                        configuration->jobs[p].startdelay, stjobs[jobID], spjobs[jobID], jobs[jobID], erjobs[jobID], udata);
        if (configuration->jobs[p].usedate) {   //if complete date set
            mytimerplus *tplus = timerplus_init(t, configuration->jobs[p].year, configuration->jobs[p].month, configuration->jobs[p].date, 
                                                configuration->jobs[p].hour, configuration->jobs[p].minute, configuration->jobs[p].second);
            pthread_create((prod + p), NULL, startat, tplus);
        }
        else
            pthread_create((prod + p), NULL, start, t);
        pthread_mutex_lock(&activeprods_mut);
        activeprods++; //one more producer is active
        pthread_mutex_unlock(&activeprods_mut);
    }

    
    for (int c = 0; c < configuration->consumersnum; c++) {
        pthread_create((cons + c), NULL, consumer, NULL);
        pthread_mutex_lock(&activecons_mut);
        activecons++;
        pthread_mutex_unlock(&activecons_mut);
    }

    for (int p = 0; p < configuration->producersnum; p++)
        pthread_join(*(prod + p), NULL);
    for (int c = 0; c < configuration->consumersnum; c++)
        pthread_join(*(cons + c), NULL);

    closeall(-1);
    printf("DONE!\n");
    return 0;
}

void closeall(int sg)
{
    free(prod);
    free(cons);
    free(configuration->jobs);
    free(configuration);
    queueDelete(fifo);
    close_log(glLog);
    if (sg == SIGINT){
        printf("Terminated\n");
        exit(0);
    }
}

void custPrint_main(char *msg)
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