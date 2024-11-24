/******************************************************************************
 *
 *Computer Engineering Group, Heidelberg University - GPU Computing Exercise 04
 *
 *                  Group : 09
 *
 *                   File : kernel.cu
 *
 *                Purpose : Memory Operations Benchmark
 *
 ******************************************************************************/
#define FLOAT_SIZE 4

//
// I Global to shared
//
__global__ void 
GlobalMem2SharedMem
( float* d_memoryA, int elemMemory, int elemThread )
{	
	// Dynamically allocate shared memory.				
	extern __shared__ float s[];

	// Get shared mem. index in which the thread should start copying.
	int startSharedElemIdx = (threadIdx.x * elemThread);

	// Get elem. index in the global memory from which 
	// the current thread should start copying.
    int startGlobalElemIdx = (blockIdx.x * blockDim.x + threadIdx.x) * elemThread;

	// Iterate over the elements the thread should copy.
    for ( int i = 0; i < elemThread; i++ ) {
		// Check that we don't go over the memory's limits.
        if ( startGlobalElemIdx + i < elemMemory ) {
            s[startSharedElemIdx + i] = d_memoryA[startGlobalElemIdx + i];
        }
	}
	__syncthreads();

}
void GlobalMem2SharedMem_Wrapper(
	/* Kernel execution config param */
	dim3 gridSize,
	dim3 blockSize,
	int memorySize,
	/* Kernel arguments */
	float* d_memoryA,
	int elemMemory,
	int elemThread
) 
{
	GlobalMem2SharedMem<<< gridSize, blockSize, memorySize >>> ( d_memoryA, elemMemory, elemThread );
}

//
// II Shared to global
//
__global__ void 
SharedMem2GlobalMem
( float* d_memoryA, int elemMemory, int elemThread )
{
	// Dynamically allocate shared memory.
	extern __shared__ float s[];

	// Get shared mem. index in which the thread should start copying.
	int startSharedElemIdx = (threadIdx.x * elemThread);

	// Get elem. index in the global memory from which 
	// the current thread should start copying.
    int startGlobalElemIdx = (blockIdx.x * blockDim.x + threadIdx.x) * elemThread;

	// //
	// // Part I : Load data into shared memory from global.
	// //
	// // Iterate over the elements the thread should copy.
    // for ( int i = 0; i < elemThread; i++ ) {
	// 	// Check that we don't go over the memory's limits.
    //     if ( startGlobalElemIdx + i < elemMemory ) {
    //         s[startSharedElemIdx + i] = d_memoryA[startGlobalElemIdx + i];
    //     }
	// }

	// __syncthreads();

	//
	// Part II : Load data into **global** memory from **shared**.
	//
	// Iterate over the elements the thread should copy.
    for ( int i = 0; i < elemThread; i++ ) {
		// Check that we don't go over the memory's limits.
        if ( startGlobalElemIdx + i < elemMemory ) {
            d_memoryA[startGlobalElemIdx + i] = s[startSharedElemIdx + i];
        }
	}
	__syncthreads();
}
void SharedMem2GlobalMem_Wrapper(
	/* Kernel execution config param */
	dim3 gridSize,
	dim3 blockSize,
	int memorySize,
	/* Kernel arguments */
	float* d_memoryA,
	int elemMemory,
	int elemThread 
)
{
	SharedMem2GlobalMem<<< gridSize, blockSize, memorySize >>> ( d_memoryA, elemMemory, elemThread );
}

//
// III Shared to register
//
__global__ void 
SharedMem2Registers
( int registerMemory, int registerThread, float* outFloat )
{
	// Dynamically allocate shared memory.
	extern __shared__ float s[];

	// Get thread id.
	int tIdx = threadIdx.x;

	// Get shared mem. index from which the thread should start copying.
	int startSharedElemIdx = (tIdx * registerThread);

	// Initialize thread register.
	float tRegister = 0;

	for ( int i = 0; i < registerThread; i++ ) {
		// Check that we don't go over the memory's limits.
        if ( startSharedElemIdx + i < registerMemory ) {
            tRegister = s[startSharedElemIdx + i];
        }
	}
	
	if ( tIdx == 0 ) {
		// Save last value to avoid compiler optimizations.
		*outFloat = tRegister;
	}
}
void SharedMem2Registers_Wrapper(
	/* Kernel execution config param */
	dim3 gridSize,
	dim3 blockSize,
	int memorySize,
	/* Kernel arguments */
	int registerMemory,
	int registerThread,
	float* outFloat
) 
{
	SharedMem2Registers<<< gridSize, blockSize, memorySize >>>( registerMemory, registerThread, outFloat );
}

//
// Register to shared
//
__global__ void 
Registers2SharedMem
( int registerMemory, int registerThread, float* outFloat )
{
	// Dynamically allocate shared memory.
	extern __shared__ float s[];

	// Get thread id.
	int tIdx = threadIdx.x;

	// Get shared mem. index from which the thread should start copying.
	int startSharedElemIdx = (tIdx * registerThread);

	// Initialize thread register.
	float tRegister = 0;

	for ( int i = 0; i < registerThread; i++ ) {
		// Check that we don't go over the memory's limits.
        if ( startSharedElemIdx + i < registerMemory ) {
            s[startSharedElemIdx + i] = tRegister;
        }
	}
	
	if ( tIdx == 0 ) {
		// Save last value to avoid compiler optimizations.
		*outFloat = tRegister;
	}
}
void Registers2SharedMem_Wrapper(
	/* Kernel execution config param */
	dim3 gridSize,
	dim3 blockSize,
	int memorySize,
	/* Kernel arguments */
	int registerMemory,
	int registerThread,
	float* outFloat
) {
	Registers2SharedMem<<< gridSize, blockSize, memorySize>>> ( registerMemory, registerThread, outFloat );
}


//
// Bank conflicts
//
__global__ void 
BankConflictsRead
( int elemBankMemory, int stride, int numIterations, long* dClocks, float* outFloat )
{
	// Dynamically allocate shared memory.
	extern __shared__ float s[];

	// Get thread id.
	int tIdx = threadIdx.x;

	// Set thread address.
	int tAddr = tIdx * stride;

	// Initialize shared memory.
    if (tIdx == 0) {
        for (int i = 0; i < elemBankMemory; i++) {
			s[i] = 0;
		}
    }
    __syncthreads();

	// Initialize thread register.
	float tRegister = 0;

	long long int start = clock64();

	for (int i = 0; i < numIterations; ++i) {
		// Copy data (a float) from shared mem. address to thread's register.
		tRegister = s[tAddr & (elemBankMemory - 1)];
	}

	long long int end = clock64();

	if ( tIdx == 0 ) {
		// Save last value to avoid compiler optimizations.
		*outFloat = tRegister;
		*dClocks = (long)((end - start) / numIterations);
	}
}
void BankConflictsRead_Wrapper(
	/* Kernel execution config param */
	dim3 gridSize,
	dim3 blockSize,
	int memorySize,
	/* Kernel arguments */
	int elemBankMemory,
	int stride,
	int numIterations,
	long* dClocks,
	float* outFloat
) {
	BankConflictsRead<<< gridSize, blockSize, memorySize >>> (
		elemBankMemory,
		stride,
		numIterations, 
		dClocks,
		outFloat
	);
}

