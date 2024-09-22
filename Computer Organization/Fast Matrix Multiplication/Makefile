SHELL := /bin/bash

######################################################################
### EXECUTABLES

CC    = gcc
FLAGS = -O2 -Wall -std=gnu99
RM    = rm -rf
LINK  = -lm

######################################################################
### DEFINES

# DEFS += -DMAX_ITER=2
# DEFS += -DMIN_SIZE=512
# DEFS += -DMAX_SIZE=512

######################################################################
### TARGET FILES

MAIN = matrix-mult

######################################################################
### CLEAN-UP FILES

FILES_CLEAN = $(MAIN) *~

######################################################################
### COMMANDS -- ! DO NOT CHANGE BELOW !

all: $(MAIN)

debug: DEFS += -DFLAG_DEBUG
debug: $(MAIN)

$(MAIN): $(MAIN).c fast_mult.c
	$(CC) $(FLAGS) $(DEFS) $^ -o $@ -lm

clean:
	$(RM) $(FILES_CLEAN) *.o *~

.PHONY: $(MAIN)
