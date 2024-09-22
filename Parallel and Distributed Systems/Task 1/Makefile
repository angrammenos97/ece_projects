# ####################################################################
#
#			   C/C++ Makefile
#
# Author: Anastasios Grammenos <avgramme@ece.auth.gr>
#
# Adapted from
#  http://www.cs.swarthmore.edu/~newhall/unixhelp/howto_makefiles.html
#
# ####################################################################
#
# 'make'        build executable file 'main'
# 'make lib'		build the libraries .a
# 'make clean'  removes all .o and executable files
#

# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use,default here is clang
CC = gcc-7

# setup paths
SRCDIR = src
INCDIR = inc
LIBDIR = lib
MATDIR = matlab

# define compile-time flags
CFLAGS = -fcilkplus -fopenmp -lpthread -Wall

# define any directories containing header files
INCLUDES = -I $(INCDIR)

# define the source file for the library
SRC = vptree

# define the different possible executables
TYPES = sequential openmp cilk pthreads

# define the executable file  name
MAIN = main

#
# The following part of the makefile is generic; it can be used to
# build any executable just by changing the definitions above
#

# call everytime
.PRECIOUS: $(LIBDIR)/%.a

all: $(addprefix $(MAIN)_, $(TYPES))

lib: $(addprefix $(LIBDIR)/, $(addsuffix .a, $(addprefix $(SRC)_, $(TYPES))))

$(MAIN)_%: $(SRCDIR)/$(MAIN).c $(LIBDIR)/$(SRC)_%.a
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ -lm

# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .cpp file) and $@: the name of the target of the rule (a .o file)

$(LIBDIR)/$(SRC)_%.a: $(SRCDIR)/$(SRC)_%.o
	mkdir -p $(LIBDIR)
	ar rcs $@ $<

# (see the gnu make manual section about automatic variables)
$(SRCDIR)/$(SRC)_%.o: $(SRCDIR)/$(SRC)_%.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $< -lm

clean:
	$(RM) $(LIBDIR)/* $(SRCDIR)/*~ $(SRCDIR)/*.o $(INCDIR)/*~ $(MATDIR)/data.m $(addprefix $(MAIN)_, $(TYPES)) *~
