# Makefile for the CUDA NPP batch image processor.
#
# Targets:
#   make        / make all   build bin/npp_processor
#   make run                 build (if needed) and run run.sh
#   make clean               remove object files and the binary

NVCC      ?= nvcc
NVCC_FLAGS = -O2 -std=c++17
INCLUDES   = -Ilib -Isrc

# NPP + CUDA core libraries.
#   nppif  - filtering functions (box filter, Canny)
#   nppig  - geometry transforms (resize)
#   nppicc - color conversion (RGB -> gray)
#   nppidei- data exchange & initialization
#   nppist - statistics & thresholding
#   nppc   - NPP core
#   culibos- CUDA library OS support
LIBS = -lnppif -lnppig -lnppicc -lnppidei -lnppist -lnppc -lculibos

SRC_DIR = src
BIN_DIR = bin
TARGET  = $(BIN_DIR)/npp_processor

# All CUDA translation units in src/.
SOURCES = $(wildcard $(SRC_DIR)/*.cu)
OBJECTS = $(SOURCES:.cu=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(OBJECTS) -o $@ $(LIBS)

# Pattern rule: compile each .cu into a .o.
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

run: all
	bash run.sh

clean:
	rm -f $(SRC_DIR)/*.o $(TARGET)
