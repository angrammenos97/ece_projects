#ifndef LOGGER_H_
#define LOGGER_H_

typedef struct logFile
{
    char *fname;
    void *file; //type:FILE
    char *buff; //buffer
    int close;      //close file?
    void* logmut; //type:pthread_mutex_t
    void* starttime;    //type:timeval
}logFile;

extern logFile *glLog;  //global logger

logFile *logfile_init(char *fname, int close);

void log_write(logFile *flog, char *msg);

void log_write_r(logFile *flog, char *msg);  //thread-safe

void close_log(logFile *flog);

#endif // LOGGER_H_