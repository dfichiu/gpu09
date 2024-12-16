#include <stdio.h>

// 一个简单的 CUDA kernel
__global__ void helloCUDA() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    // 启动 CUDA kernel
    helloCUDA<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
