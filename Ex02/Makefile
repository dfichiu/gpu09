CUDA_ROOT=$(CUDA_HOME)
INC=-I./inc -I$(CUDA_ROOT)/include
LIB=-L$(CUDA_ROOT)/lib64
NVCC=nvcc
NVCC_FLAGS=-O2 --compiler-options "-O2 -Wall -Wextra"

.PHONY: build
build: ./bin/measureDataMovements

.PHONY: build1
build1: ./bin/nullKernelSync

.PHONY: build2
build2: ./bin/busyWaitKernelAsync

.PHONY: clean
clean:
	rm ./bin/*

.PHONY: rebuild
rebuild: clean build

# Exercise 2.1
./bin/nullKernelAsync: ./src/nullKernelAsync.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)

# Exercise 2.2
./bin/nullKernelSync: ./src/nullKernelSync.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)

./bin/busyWaitKernelAsync: ./src/busyWaitKernelAsync.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)

# Exercise 2.3
./bin/measureDataMovements: ./src/measureDataMovements.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)

