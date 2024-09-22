#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

#define BUFSIZE 128

void child( int *fd, char **cmd )
{
  close(fd[0]); // close pipe-read

  //int saved_stdout = dup(STDOUT_FILENO);  // save stdout to restore it

  if (dup2(fd[1], STDOUT_FILENO) == -1) perror("Error using dup2"); // redirect stdout to pipe
  if (dup2(fd[1], STDERR_FILENO) == -1) perror("Error using dup2"); // redirect stderr to pipe

  execvp(*cmd, cmd);    // execute commands

  printf("ERROR\n");    // in case of errors send ERROR to father process
  close(fd[1]);         // close pipe-write

  //fflush(stdout);   // clean stdout
  //if (dup2(STDOUT_FILENO, saved_stdout) == -1) perror("Error using dup2");  // restore stdout like it was
  //printf("ERROR\n");     // print in stdout ERROR
  //close(saved_stdout);    // close file which saved stdout original position

}

int main(int argc, char *argv[])
{
  if (argc == 1){
    printf("Write a command as an input.\n");
    exit(1);
  }
  int fd[2];
  int pid, status, n;
  if ( pipe(fd) < 0 ) perror("Error openning pipe");

  pid = fork();
  if (pid < 0) perror("Error doing fork");
  else if (pid == 0) {  // Child proccess
    child(fd, (argv+1));
    exit(0);
  }
  else{                 // Father proccess
    close(fd[1]); // close pipe-write
    wait(&status); // wait child to finish
    sleep(1);
    char buf[BUFSIZE];
    printf("FATHER: I haνe received from children:\n\n");
    while ( (n=read(fd[0], buf, BUFSIZE)) > 0 ){
      write(STDOUT_FILENO, buf, n); // works as printf
    }
    close(fd[0]); // close pipe-read
  }


  return 0;
}
