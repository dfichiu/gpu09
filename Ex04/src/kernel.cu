/******************************************************************************
 *
 *Computer Engineering Group, Heidelberg University - GPU Computing Exercise 04
 *
 *                  Group : TBD
 *
 *                   File : kernel.cu
 *
 *                Purpose : Memory Operations Benchmark
 *
 ******************************************************************************/


//
// Test Kernel
//

__global__ void 
globalMem2SharedMem
( float* d_memoryA, int N, int memSize )
{						
	extern __shared__ float s[];
	int SIZE_FLOAT = 4;
    int indx = (blockIdx.x * blockDim.x + threadIdx.x) * memSize;
    for ( int i = 0; i < memSize / SIZE_FLOAT; i++ ) {
        if ( indx + i < N / SIZE_FLOAT ) {
            s[indx + i] = d_memoryA[indx + i];
        }
	}
	__syncthreads();

}

void globalMem2SharedMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* d_memoryA, int N, int memSize ) {
	globalMem2SharedMem<<< gridSize, blockSize, shmSize >>>( d_memoryA,  N,  memSize );
}

__global__ void 
SharedMem2globalMem
( float* d_memoryA, int N, int memSize )
{
	extern __shared__ float s[];
	int SIZE_FLOAT = 4;
    int indx = (blockIdx.x * blockDim.x + threadIdx.x) * memSize;
    for ( int i = 0; i < memSize / SIZE_FLOAT; i++ ) {
        if ( indx + i < N / SIZE_FLOAT ) {
            d_memoryA[indx + i] = s[indx + i];
        }
	}
	__syncthreads();
}

void SharedMem2globalMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* d_memoryA, int N, int memSize ) {
	SharedMem2globalMem<<< gridSize, blockSize, shmSize >>>( d_memoryA,  N,  memSize );
}

__global__ void 
SharedMem2Registers
//(/*TODO Parameters*/)
( )
{
	/*TODO Kernel Code*/
}
void SharedMem2Registers_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize /* TODO Parameters*/) {
	SharedMem2Registers<<< gridSize, blockSize, shmSize >>>( /* TODO Parameters */);
}

__global__ void 
Registers2SharedMem
//(/*TODO Parameters*/)
( )
{
	// Dynamically allocate shared memory
	extern __shared__ float s[];


}
void Registers2SharedMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize /* TODO Parameters*/) {
	Registers2SharedMem<<< gridSize, blockSize, shmSize >>>( /* TODO Parameters */);
}

__global__ void 
bankConflictsRead
//(/*TODO Parameters*/)
( )
{
	/*TODO Kernel Code*/
}

void bankConflictsRead_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize /* TODO Parameters*/) {
	bankConflictsRead<<< gridSize, blockSize, shmSize >>>( /* TODO Parameters */);
}
