# Compiler
CC = mpicc
CFLAGS = -O2 -fopenmp -g

# Executable name
TARGET = parsim

# Source and header files
SRCS = parsim-mpi.c init_particles.c cell.c
HEADERS = init_particles.h cell.h
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