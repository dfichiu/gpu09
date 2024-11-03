CUDA_ROOT=$(CUDA_HOME)
INC=-I./inc -I$(CUDA_ROOT)/include
LIB=-L$(CUDA_ROOT)/lib64
NVCC=nvcc
NVCC_FLAGS=-O2 --compiler-options "-O2 -Wall -Wextra"

# Exercise 1
.PHONY: build1as
build1as: ./bin/nullKernelAsyncSingle

.PHONY: build1ss
build1ss: ./bin/nullKernelSyncSingle

.PHONY: build1a
build1a: ./bin/nullKernelAsync

.PHONY: build1s
build1s: ./bin/nullKernelSync

# Exercise 2
.PHONY: build2
build2: ./bin/busyWaitKernelAsync

# Exercise 3
.PHONY: build3
build3: ./bin/measureDataMovements

# Others
.PHONY: clean
clean:
	rm ./bin/*

.PHONY: rebuild
rebuild: clean build

# Exercise 2.1
./bin/nullKernelAsyncSingle: ./src/nullKernelAsyncSingle.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)

./bin/nullKernelSyncSingle: ./src/nullKernelSyncSingle.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)

./bin/nullKernelAsync: ./src/nullKernelAsync.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)

./bin/nullKernelSync: ./src/nullKernelSync.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)

# Exercise 2.2
./bin/busyWaitKernelAsync: ./src/busyWaitKernelAsync.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)

# Exercise 2.3
./bin/measureDataMovements: ./src/measureDataMovements.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)
