# Compiler
CC = gcc
CFLAGS = -O2 -fopenmp 

# Executable name
TARGET = parsim

# Source and header files
SRCS = parsim-omp.c init_particles.c
HEADERS = init_particles.h
OBJS = $(SRCS:.c=.o)

# Default rule: Build the program
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) -lm

# Rule for compiling .c files into .o files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rule: Remove compiled files
clean:
	rm -f $(TARGET) $(OBJS)