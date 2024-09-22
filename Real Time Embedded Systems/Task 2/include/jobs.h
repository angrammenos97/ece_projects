#ifndef JOBS_H_
#define JOBS_H_

#define JOBSNUMBER 3

//////////PRINT JOB///////////
typedef struct prtArgs {
	unsigned int *charnum;	//number of characters(default 1000)
	char *str;				//string to print
}prtArgs;

void* str_generator(void *args);

void* print_job(void *args);

void* print_done(void *args);

void* error_print(void*);
///////////SIN JOB////////////
typedef struct sinArgs {
	unsigned int *angnum;	//number of angles(default 30000)
	double *angs;	//input angles
	double *res;	//output results
}sinArgs;

void* sin_job(void* args);

void* sin_printer(void *args);

void* error_sin(void*);
///////////COS JOB////////////
typedef sinArgs cosArgs;

void* cos_job(void *args);

void* cos_printer(void *args);

void* error_cos(void*);
//////////////////////////////

void* angs_generator(void *args);

//////////////////////////////
static void* (*jobs[])(void*) = {print_job, sin_job, cos_job};             	//all 3 jobs
static void* (*stjobs[])(void*) = {str_generator, angs_generator, angs_generator};  //all start fcn
static void* (*spjobs[])(void*) = {print_done, sin_printer, cos_printer}; 	//all stop fcn
static void* (*erjobs[])(void*) = {error_print, error_sin, error_cos};    	//all error fcn
static char *jobnames[] = {"print_job", "sin_job", "cos_job"};				//
static int udatasz[] = {sizeof(prtArgs), sizeof(sinArgs), sizeof(cosArgs)};	//size of input arguments

#endif //JOBS_H_