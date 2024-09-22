#ifndef TIMER_H_
#define TIMER_H_

typedef unsigned int uint;

typedef struct mytimer
{
    uint period;        //in msec
    uint taskstoexecute;
    uint startdelay;    //in sec
    void* (*startfcn)(void *) ;
    void* (*stopfcn)(void *) ;
    void* (*timerfcn)(void *) ;
    void* (*errorfcn)(void *) ;
    void* userdata;
}mytimer;

typedef struct mytimerplus
{
    mytimer *t;
    int y;     //year
    int m;     //month
    int d;     //date
    int h;     //hour
    int min;   //minute
    int sec;   //second
}mytimerplus;

mytimer* timer_init(uint period, uint ttoexec, uint stdelay, void* stfcn, void* spfcn, void* tfcn, void* erfcn, void* udata);

mytimerplus *timerplus_init(mytimer *t, int y, int m, int d, int h, int min, int sec);

void* start(void *t);

void* startat(void *arg);

#endif //TIMER_H_