# Define the C compiler
CC=gcc

# Define the C flags
CFLAGS=-g -Wall -Werror

# Define the source files
SOURCES=game.c E9212.c

# Define the object files
OBJECTS=$(SOURCES:.c=.o)

# Define the executable
EXECUTABLE=game

# Default target
all: $(EXECUTABLE)

# Compile the object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Link the object files to create the executable
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(EXECUTABLE) -lncurses

# Clean up
clean:
	rm -f $(OBJECTS) $(EXECUTABLE)