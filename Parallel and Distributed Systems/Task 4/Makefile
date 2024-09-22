# define the shell to bash
SHELL	:= /bin/bash

MKDIR   := mkdir
RMDIR   := rm -rf
CC      := nvcc
BINDIR  := ./bin
OBJDIR  := ./obj
INCDIR 	:= ./include
SRCDIR  := ./src
MAINSDIR:= ./mains
SRCS    := $(wildcard $(SRCDIR)/*.cu)
OBJS    := $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(SRCS))
MAINS	:= $(patsubst $(MAINSDIR)/%.cu,$(BINDIR)/%,$(wildcard $(MAINSDIR)/*.cu))
CFLAGS  := -O0 -arch=sm_50 -cudart=shared -rdc=true -I$(INCDIR)
LDFLAGS := -lm

.PHONY: all clean

all: $(MAINS)

$(BINDIR)/%: $(OBJS) | $(BINDIR)
	$(CC) $(CFLAGS) $(LDFLAGS) $(MAINSDIR)/$(notdir $@).cu $^ -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -c $< -o $@

$(BINDIR) $(OBJDIR):
	$(MKDIR) $@

#clean everything
clean:
	$(RMDIR) $(OBJDIR) $(BINDIR)