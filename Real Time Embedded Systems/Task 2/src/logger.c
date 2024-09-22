#include "logger.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>

struct timeval *difftimeval(struct timeval *sttime, struct timeval *edtime);
float getCpuUsage();

logFile *logfile_init(char *fname, int close)
{
    logFile *flog = (logFile*)malloc(sizeof(logFile));
    flog->fname = fname;
    flog->close = close;
    flog->logmut = (void*)malloc(sizeof(pthread_mutex_t));
    flog->starttime = (void*)malloc(sizeof(struct timeval));
    pthread_mutex_init((pthread_mutex_t*)flog->logmut, NULL);
    flog->file = (void*)fopen(flog->fname, "w");    //create a new file
    if (flog->close)
        fclose((FILE*)flog->file);
    else 
        flog->buff = (char*)calloc(1000000,sizeof(char));   //create a buffer of 1000000bytes
    char msg[64];
    time_t timenow = time(NULL);
    sprintf(msg, "Start logging at %s", asctime(localtime(&timenow)));    
    gettimeofday(flog->starttime, NULL);
    log_write(flog, msg);
    return flog;
}

void log_write(logFile *flog, char *msg)
{
    static char timestamp[32];
    struct timeval timenow, *eltime;
    gettimeofday(&timenow, NULL);
    eltime = difftimeval(flog->starttime, &timenow);
    sprintf(timestamp, "%li:%li > ", eltime->tv_sec, eltime->tv_usec);
    if (flog->close){
        flog->file = (void*)fopen(flog->fname, "a");
        fprintf(flog->file, "%s%s", timestamp, msg);
        fclose(flog->file);
    }
    else{
        //TODO flush full buff
        strcat(flog->buff, timestamp);
        strcat(flog->buff, msg);
    }
}

void log_write_r(logFile *flog, char *msg)
{
    pthread_mutex_lock((pthread_mutex_t*)flog->logmut);
    log_write(flog, msg);
    pthread_mutex_unlock((pthread_mutex_t*)flog->logmut);
}

void close_log(logFile *flog)
{
    if (flog == NULL)
        return;
    char msg[64];
    time_t timenow = time(NULL);
    sprintf(msg, "Finished logging at %s", asctime(localtime(&timenow)));
    log_write(flog, msg);
    if (!flog->close){
        flog->file = (void*)fopen(flog->fname, "a");
        printf("Logged %d characters!\n", fprintf(flog->file, flog->buff));    //flush buff
        fclose(flog->file);
    }        
    free((pthread_mutex_t*)flog->logmut);
    free((struct timeval*)flog->starttime);
    free(flog);
}

struct timeval *difftimeval(struct timeval *sttime, struct timeval *edtime)
{
    struct timeval *eltime = (struct timeval*)malloc(sizeof(struct timeval));
    eltime->tv_sec = edtime->tv_sec - sttime->tv_sec;
    if (edtime->tv_usec - sttime->tv_usec < 0){
        eltime->tv_sec--;
        eltime->tv_usec = 1000000 + edtime->tv_usec - sttime->tv_usec;
    }
    else
        eltime->tv_usec = edtime->tv_usec - sttime->tv_usec;

    return eltime;
}
