# Makefile for Dampened Wave Equation Simulation

# Compiler
MPICXX = mpicxx

# Compiler flags
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra

# Target executable
TARGET = wave_simulation

# Source files
SRC_DIR = src
SOURCES = $(SRC_DIR)/wave_simulation.cpp

# Object files
OBJ_DIR = obj
OBJECTS = $(OBJ_DIR)/wave_simulation.o

# Default target
all: $(TARGET)

# Create object directory
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Compile source file
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

# Link executable
$(TARGET): $(OBJECTS)
	$(MPICXX) $(CXXFLAGS) $^ -o $@

# Clean build files
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

# Clean all including output
cleanall: clean
	rm -rf output* *.out *.err

# Phony targets
.PHONY: all clean cleanall

# Debug build
debug: CXXFLAGS = -g -std=c++17 -Wall -Wextra -DDEBUG
debug: clean all
