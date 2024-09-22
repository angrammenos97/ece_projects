#ifndef CONF_H_
#define CONF_H_

typedef struct jobConf
{
    int jobid;
    int period;
    int numofexec;
    int startdelay;
    int usedate;
    int year;
    int month;
    int date;
    int hour;
    int minute;
    int second;
}jobConf;

typedef struct confFile
{
    int uselog;
    char *logfname;
    int livelog;
    int queuesize;
    int producersnum;
    int consumersnum;
    int jobslength;
    jobConf *jobs;
}confFile;

confFile *export_conf(int argc, char *argv[]);

#endif //CONF_H_