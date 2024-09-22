SHELL := /bin/bash

GCC	= gcc
RM	= rm -rf

MAIN	= add_half
HELPERS = helpers
OBJS    = $(HELPERS).o

all: $(MAIN)

$(MAIN): $(MAIN).c $(OBJS) half.h half_gold.h
	$(GCC) $< $(OBJS) -o $@

%.o: %.c %.h
	$(GCC) -c $<

clean:
	$(RM) $(MAIN) *~ a.out $(HELPERS).o

