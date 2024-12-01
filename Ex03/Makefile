CUDA_ROOT=$(CUDA_HOME)
INC=-I./inc -I$(CUDA_ROOT)/include
LIB=-L$(CUDA_ROOT)/lib64
NVCC=nvcc
NVCC_FLAGS=-O2 --compiler-options "-O2 -Wall -Wextra"

# Exercise 1
.PHONY: build1 
build1 : ./bin/task3_1
 
# Exercise 1
.PHONY: build2 
build2 : ./bin/main 
 
# Others
.PHONY: clean
clean:
	rm ./bin/*

.PHONY: rebuild
rebuild: clean build

# Exercise 3.1
./bin/task3_1: ./src/task3_1.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)

# Exercise 3.2
./bin/main: ./src/kernel.cu ./src/main.cpp
	$(NVCC) $(NVCC_FLAGS)  -o $@ $^ $(INC) $(LIB)
 
 