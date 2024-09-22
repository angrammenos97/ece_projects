# define the shell to bash
SHELL	:= /bin/bash

MKDIR   := mkdir
RMDIR   := rm -rf
CC      := arm-linux-gnueabihf-gcc
BIN     := ./bin
OBJ     := ./obj
INCLUDE := ./include
SRC     := ./src
SRCS    := $(wildcard $(SRC)/*.c)
OBJS    := $(patsubst $(SRC)/%.c,$(OBJ)/%.o,$(SRCS))
MAIN	:= main
EXE     := $(BIN)/$(MAIN)
CFLAGS  := -g -D LOGGER -I$(INCLUDE)
LDFLAGS := -lm -lpthread
IP		:= 192.168.1.7
ARGS	:= conf.json

.PHONY: all rebuild build release exec debug reset clean

all: $(EXE)

$(EXE): $(OBJS) | $(BIN)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

$(OBJ)/%.o: $(SRC)/%.c | $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -c $< -o $@

$(BIN) $(OBJ):
	$(MKDIR) $@

#rebuild
rebuild: clean $(EXE)
	-scp $(EXE) root@$(IP):
#build
build: $(EXE)
	-scp $(EXE) root@$(IP):
#build release
release:
	$(eval CFLAGS := $(filter-out -g, $(CFLAGS)))
	make rebuild CFLAGS="$(CFLAGS)"
#run
exec: $(EXE)
	@scp -q $(EXE) root@$(IP):
	@ssh -n -f root@$(IP) ./$(MAIN) $(ARGS)
#debug
debug: reset $(EXE) 
	scp $(EXE) root@$(IP):
	ssh -n -f root@$(IP) ./gdbserver localhost:2000 $(MAIN) $(ARGS)
#reset server
reset: 
	-ssh root@$(IP) killall -q gdbserver
	-ssh root@$(IP) killall -q $(MAIN)
#clean everything
clean: reset
	-ssh root@$(IP) $(RM) $(MAIN)
	$(RMDIR) $(OBJ) $(BIN)