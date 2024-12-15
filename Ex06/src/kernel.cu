/**************************************************************************************************
 *
 *       Computer Engineering Group, Heidelberg University - GPU Computing Exercise 06
 *
 *                 Gruppe : 09
 *
 *                   File : kernel.cu
 *
 *                Purpose : Reduction
 *
 **************************************************************************************************/

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>



__global__ void
initial_Kernel(int numElements, float* dataIn, float* dataOut)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	if (index < numElements)
	{
		for ( unsigned int s = 1; s < blockDim.x; s *= 2 ) {
			if ( tid % ( 2 * s ) == 0 ) {
				dataIn[index] += dataIn[index + s];
			}
			__syncthreads();
		}
		if ( tid == 0 ) {
			dataOut[blockIdx.x] = dataIn[index];
		}	
	}
}

void initial_Kernel_Wrapper(
	dim3 gridSize,
	dim3 blockSize,
	int numElements,
	float* dataIn,
	float* dataOut
)
{
	initial_Kernel<<< gridSize, blockSize>>>(numElements, dataIn, dataOut);
}


//
// Reduction_Kernel
//
__global__ void
optimized_Kernel(int numElements, float* dataIn, float* dataOut)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int bdim = blockDim.x;

	int index = bid * bdim + tid;

	// Declare shared memory.
	extern __shared__ float shMem[];

	if (index < numElements)
	{
		// Each thread loads one element from global into shared.
		shMem[tid] = dataIn[index];
		__syncthreads();

		int mapped_tid = 2 * tid;
		for ( unsigned int s = 1; s < bdim; s *= 2 ) {
			if ( mapped_tid < bdim ) {
				shMem[mapped_tid] += shMem[mapped_tid + s];
			}
			mapped_tid *= 2;
			__syncthreads();
		}
		if ( tid == 0 ) {
			dataOut[bid] = shMem[0];
		}	
	}
}

void optimized_Kernel_Wrapper(
	dim3 gridSize,
	dim3 blockSize,
	int shMemSize,
	int numElements,
	float* dataIn,
	float* dataOut
) 
{
	optimized_Kernel<<< gridSize, blockSize, shMemSize >>>(numElements, dataIn, dataOut);
}


//
// Reduction_Kernel
//
__global__ void
reduction_Kernel(int numElements, float* dataIn, float* dataOut)
{
	int index = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
	int tid = threadIdx.x;


	// Declare shared memory.
	extern __shared__ float shMem[];

	if (index < numElements)
	{
		// Each thread loads one element from global into shared.
		shMem[tid] = dataIn[index] + dataIn[index + blockDim.x];
		__syncthreads();

		// int indexedTid = 2 * tid;
		for ( unsigned int o = blockDim.x / 2; o > 0; o /= 2 ) {
			if ( tid < o ) {
				shMem[tid] += shMem[tid + o];
			}
			// indexedTid *= 2;
			__syncthreads();
		}
		if ( tid == 0 ) {
			dataOut[blockIdx.x] = shMem[0];
		}	
	}
}

void reduction_Kernel_Wrapper(
	dim3 gridSize,
	dim3 blockSize,
	int shMemSize,
	int numElements,
	float* dataIn,
	float* dataOut
) 
{
	reduction_Kernel<<< gridSize, blockSize, shMemSize >>>(numElements, dataIn, dataOut);
}

//
// Reduction Kernel using CUDA Thrust
//

void thrust_reduction_Wrapper(int numElements, float* dataIn, float* dataOut) {
	thrust::device_ptr<float> in_ptr = thrust::device_pointer_cast(dataIn);
	thrust::device_ptr<float> out_ptr = thrust::device_pointer_cast(dataOut);
	
	*out_ptr = thrust::reduce(in_ptr, in_ptr + numElements, (float) 0., thrust::plus<float>());	
}
