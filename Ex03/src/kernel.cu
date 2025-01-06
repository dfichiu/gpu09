/*************************************************************************************************
 *
 *        Computer Engineering Group, Heidelberg University - GPU Computing Exercise 03
 *
 *                           Group : TBD
 *
 *                            File : main.cu
 *
 *                         Purpose : Memory Operations Benchmark
 *
 *************************************************************************************************/

//
// Kernels
//


// N = Total memory
// memSize = Memory/thread
__global__ void 
globalMemCoalescedKernel(int* d_memoryA, int* d_memoryB, int N, int memSize){
    int SIZE_INT = 4;
    int indx = (blockIdx.x * blockDim.x + threadIdx.x) * memSize;
    for (int i = 0; i < memSize/SIZE_INT; i++) {
        if (indx + i < N/SIZE_INT ) {
            d_memoryB[indx + i] = d_memoryA[indx + i];
        }
   }
}

void 
globalMemCoalescedKernel_Wrapper(dim3 gridDim, dim3 blockDim, int* d_memoryA, int* d_memoryB ,int N, int memSize) {
	globalMemCoalescedKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/>>> (d_memoryA, d_memoryB, N, memSize);
}

__global__ void 
globalMemStrideKernel(int* d_memoryA, int* d_memoryB ,int N,  int stride)
{
    int indx = (blockIdx.x*blockDim.x + threadIdx.x) * stride  ;
 
    
    if (indx  < N/4 ) {
        d_memoryB[indx ] = d_memoryA[indx ];
    }
   
}

void 
globalMemStrideKernel_Wrapper(dim3 gridDim, dim3 blockDim, int* d_memoryA, int* d_memoryB ,int N, int stride) {
	globalMemStrideKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>(  d_memoryA,  d_memoryB , N,  stride);
}

__global__ void 
globalMemOffsetKernel(int* d_memoryA, int* d_memoryB ,int N, int offset)
{
     
    int indx = (blockIdx.x * blockDim.x + threadIdx.x);
    
    if (indx + offset< N/4 ) {
        d_memoryB[indx] = d_memoryA[indx + offset];
    }
   
}

void 
globalMemOffsetKernel_Wrapper(dim3 gridDim, dim3 blockDim, int* d_memoryA, int* d_memoryB ,int N, int offset) {
	globalMemOffsetKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>( d_memoryA,  d_memoryB , N,  offset);
}

