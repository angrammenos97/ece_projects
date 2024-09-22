# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use,default here is clang
CC = nvcc

# folders
SRCDIR = src

# define compile-time flags
CFLAGS = 
LDFLAGS = -lm

# define the source file for the library
SRC = ising

# define merged sorce code
MRGSRC = temp_main

# define the different possible executables
MAINTYPES = sequential gpu_v1 gpu_v2 gpu_v3

# define the executable file  name
MAIN = main

all : $(addprefix $(MAIN)_, $(MAINTYPES))

$(MRGSRC)_%.cu : $(SRCDIR)/$(MAIN).cu $(SRCDIR)/$(SRC)_%.c
	cat $^ > $@

$(MAIN)_% : $(MRGSRC)_%.cu 
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean :
	$(RM) $(addprefix $(MAIN)_, $(MAINTYPES)) $(MRGSRC)_*.cu output*.bin *~
