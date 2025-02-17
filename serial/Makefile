# Compiler
CC = gcc-14
CFLAGS = -O2 -fopenmp

# Executable name
TARGET = parsim

# Source files
SRCS = parsim.c init_particles.c

# Default rule: Build the program
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS)

# Clean rule: Remove compiled files
clean:
	rm -f $(TARGET)