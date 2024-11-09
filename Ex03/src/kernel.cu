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

__global__ void 
globalMemCoalescedKernel( int d_memoryA[N] ,int d_memoryB[N], int memSize)
{
   int indx = blockIdx.x * blockDim.x + threadIdx.x;
   if (indx + memSize - 1 < N){
    for (int i=0; i<memSize; i++) {
        d_memoryB[indx+i] = d_memoryA[indx+i];
        }
   }
}

void 
globalMemCoalescedKernel_Wrapper(dim3 gridDim, dim3 blockDim, int* d_memoryA ,int* d_memoryB , int memSize ) {
	globalMemCoalescedKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>( d_memoryA ,d_memoryB,  memSize);
}

__global__ void 
globalMemStrideKernel(/*TODO Parameters*/)
{
    return  ;/*TODO Kernel Code*/
}

void 
globalMemStrideKernel_Wrapper(dim3 gridDim, dim3 blockDim /*TODO Parameters*/) {
	globalMemStrideKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>( /*TODO Parameters*/);
}

__global__ void 
globalMemOffsetKernel(/*TODO Parameters*/)
{
    return  ;/*TODO Kernel Code*/
}

void 
globalMemOffsetKernel_Wrapper(dim3 gridDim, dim3 blockDim /*TODO Parameters*/) {
	globalMemOffsetKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>( /*TODO Parameters*/);
}

