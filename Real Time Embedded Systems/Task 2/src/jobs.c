#include "jobs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef LOGGER
	#include <pthread.h>
	#include "logger.h"
	extern logFile *glLog;  //global logger
#endif

#define PI 3.141592654

void custPrint_jobs(char *msg);
//////////PRINT JOB///////////
void* str_generator(void *args)
{
	prtArgs *prtargs = (prtArgs*)args;
	if (prtargs->charnum == NULL) {
		prtargs->charnum = (unsigned int*)malloc(sizeof(unsigned int));
		*(prtargs->charnum) = 1000;
	}
	prtargs->str = (char*)malloc(*(prtargs->charnum)*sizeof(char));
	for (int i = 0; i < *(prtargs->charnum); i++)
		*(prtargs->str+i) = (char)((rand()%94)+33);
	return NULL;
}

void* print_job(void *args)
{
	prtArgs *prtargs = (prtArgs*)args;
	FILE *tmp = fopen("tmp.txt", "w");
	fprintf(tmp, "PRINTING: %s\n", prtargs->str);
	fclose(tmp);
	return NULL;
}

void* print_done(void *args)
{
	prtArgs *prtargs = (prtArgs*)args;
	char msg[64];
	sprintf(msg,"A print job with %u characters just done!\n", *(prtargs->charnum));
	custPrint_jobs(msg);
	return NULL;
}

void* error_print(void *args)
{
	custPrint_jobs("ERROR: print_job!\n");	
	return NULL;
}
///////////SIN JOB////////////
void* sin_job(void* args)
{
	sinArgs *sinargs = (sinArgs*)args;
	for (unsigned int i = 0; i < *(sinargs->angnum); i++)
		*(sinargs->res + i) = sin(*(sinargs->angs + i));
	return NULL;
}

void* sin_printer(void *args)
{
	sinArgs *sinargs = (sinArgs*)args;
	char msg[64];
	sprintf(msg, "Computed the sin of %u angles!\n", *(sinargs->angnum));
	custPrint_jobs(msg);
	return NULL;
}

void* error_sin(void *args)
{
	custPrint_jobs("ERROR: sin_job!\n");
	return NULL;
}
///////////COS JOB////////////
void* cos_job(void *args)
{
	cosArgs *cosargs = (cosArgs*)args;
	for (unsigned int i = 0; i < *(cosargs->angnum); i++)
		*(cosargs->res + i) = cos(*(cosargs->angs + i));
	return NULL;
}

void* cos_printer(void *args)
{
	cosArgs *cosargs = (cosArgs*)args;
	char msg[64];
	sprintf(msg, "Computed the cos of %u angles!\n", *(cosargs->angnum));
	custPrint_jobs(msg);
	return NULL;
}

void* error_cos(void *args)
{
	custPrint_jobs("ERROR: cos_job!\n");
	return NULL;
}
//////////////////////////////
void* angs_generator(void *args)
{
	sinArgs *sinargs = (sinArgs*)args;
	if (sinargs->angnum == NULL) {
		sinargs->angnum = (unsigned int*)malloc(sizeof(unsigned int));
		*(sinargs->angnum) = 30000;
	}
	sinargs->angs = (double*)malloc(*(sinargs->angnum)*sizeof(double));
	sinargs->res = (double*)malloc(*(sinargs->angnum)*sizeof(double));
	for (int i = 0; i < *(sinargs->angnum); i++)
		*(sinargs->angs+i) = 2*PI*((double)(rand())/RAND_MAX);
	return NULL;
}
//////////////////////////////
void custPrint_jobs(char *msg)
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
