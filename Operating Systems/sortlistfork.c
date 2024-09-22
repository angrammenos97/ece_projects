/*for (int i = 0; i < nlevels; i++){
  int child = fork();
  if (child != 0){
    sleep((rand() % 5 * getpid() % 5 * (i + 1) % 5) % 5);
    printf("Im process with pif %i, fid %i and i %i\n",getpid(), child, i);
    break;
  }
}
wait(&status);*/

/*int child = fork();
if (child == 0){
    sortAllListsFork(&numbers[1], nList - 1, nElem);
}
else{
    qsort(numbers[0], nElem, sizeof(int), cmpfunc);
    writeBinary("binary-", numbers[0], nElem);
    wait(&status);
}*/
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <sys/wait.h>

#define defNoLists 5
#define defNoElem 10

int nol = defNoLists;
int noe = defNoElem;
struct timeval startwtime, endwtime;

void help(int argc, char *argv[]);
void writeBinary(char *name, int *list, int nElem);
int cmpfunc (const void * a, const void * b);
void printListsElem(int *numbers, int nList, int nElem);

///////////////////////////////////////////
void sortAllListsFork(int **numbers, int nList, int nElem)
{
  if (nList <= 0)
    return;
  int status;
  for (int i = 0; i < nList; i++){
    int child = fork();
    if (child != 0){
      qsort(numbers[i], nElem, sizeof(int), cmpfunc);
      char name[32];
      sprintf(name, "binary-%i", i);
      writeBinary(name, numbers[i], nElem);
      waitpid(child, &status, 0);
      break;
    }
  }
  return;
}
///////////////////////////////////////////

int main(int argc, char *argv[])
{
  help(argc, argv);
  printf("Running for %i Lists with %i Elements. ", nol, noe);
  srand((unsigned int)time(NULL));

  printf("Generating random data set. ");
  gettimeofday(&startwtime, NULL);
  int **X = (int **)malloc(nol * sizeof(int*) + nol * noe * sizeof(int));
  int *ptr = (int*)(X + nol);
  for (int i = 0; i < nol; i++) {
    X[i] = (ptr + i * noe);
    for (int j = 0; j < noe; j++)
      X[i][j] = ((int)rand() % 100);
  }
  gettimeofday(&endwtime, NULL);
  double p_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
  printf("DONE in %fsec!\n", p_time);

  printListsElem(*X, nol, noe);

  printf("Sorting lists. ");
  gettimeofday(&startwtime, NULL);
  sortAllListsFork(X, nol, noe);
  gettimeofday(&endwtime, NULL);
  p_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
  printf("DONE in %fsec!\n", p_time);

  printListsElem(*X, nol, noe);

  printf("Exiting\n");
  free(X);
  return 0;
}

void help(int argc, char *argv[])
{
  if (argc > 1) {
    for (int i = 1; i < argc; i += 2) {
      if (*argv[i] == '-') {
        if (*(argv[i] + 1) == 'l')
          nol = atoi(argv[i + 1]);
        else if (*(argv[i] + 1) == 'e')
          noe = atoi(argv[i + 1]);
        else {
          help(1, argv);
          return;
        }
      }
      else {
        help(1, argv);
        return;
      }
    }
    return;
  }
}

void writeBinary(char *name, int *list, int nElem)
{
  FILE *write_ptr = fopen(name, "wb");
  fwrite(list, sizeof(int), nElem, write_ptr);
  fclose(write_ptr);
}

int cmpfunc (const void * a, const void * b)
{
   return ( *(int*)a - *(int*)b );
}

void printListsElem(int *numbers, int nList, int nElem)
{
  for (int i = 0; i < nList; i++) {
    printf("List%i: [", i+1);
    for (int j = 0; j < nElem; j++)
      printf("%i ", *(numbers + i * nElem + j));
    printf("]\n");
  }
}
