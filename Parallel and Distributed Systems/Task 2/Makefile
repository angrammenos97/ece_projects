# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use,default here is clang
CC = gcc-7
MPICC = mpicc

# setup paths
SRCDIR = src
INCDIR = inc
LIBDIR = lib

# define compile-time flags
CFLAGS = -Wall
LDFLAGS = -lm -lopenblas

# define any directories containing header files
INCLUDES = -I $(INCDIR) #-I $(NETLIB_LAPACK_ROOT)/include -L $(NETLIB_LAPACK_ROOT)/lib64 # for HPC

# define the source file for the library
SRC = knnring

# define the different possible executables
MAINTYPES = sequential mpi_s mpi_a mpi_r mpi
LIBTYPES = sequential mpi

# define the executable file  name
MAIN = main

all: $(addprefix $(MAIN)_, $(MAINTYPES))

lib: $(addprefix $(LIBDIR)/, $(addsuffix .a, $(addprefix $(SRC)_, $(LIBTYPES))))

# build main files
$(MAIN)_sequential: $(SRCDIR)/$(MAIN)_sequential.c $(LIBDIR)/$(SRC)_sequential.a
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(MAIN)_mpi: $(SRCDIR)/$(MAIN)_mpi.c $(LIBDIR)/$(SRC)_mpi.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(MAIN)_mpi%: $(SRCDIR)/$(MAIN)_mpi.c $(LIBDIR)/$(SRC)_mpi%.a
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

# build libraries
$(LIBDIR)/$(SRC)_%.a: $(SRCDIR)/$(SRC)_%.o
	mkdir -p $(LIBDIR)
	ar rcs $@ $<

$(SRCDIR)/$(SRC)_sequential.o: $(SRCDIR)/$(SRC)_sequential.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $< $(LDFLAGS)

$(SRCDIR)/$(SRC)_mpi.o: $(SRCDIR)/$(SRC)_mpi.c
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ -c $< $(LDFLAGS)

$(SRCDIR)/$(SRC)_mpi%.o: $(SRCDIR)/$(SRC)_mpi%.c
	$(MPICC) $(CFLAGS) $(INCLUDES) -o $@ -c $< $(LDFLAGS)

# remove builds and clean working folder
clean:
	$(RM) $(LIBDIR)/* $(SRCDIR)/*~ $(SRCDIR)/*.o $(INCDIR)/*~ $(addprefix $(MAIN)_, $(MAINTYPES)) *~ slurm*.out error.out
