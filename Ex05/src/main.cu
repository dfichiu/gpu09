/**************************************************************************************************
 *
 *       Computer Engineering Group, Heidelberg University - GPU Computing Exercise 05
 *
 *                                 Group : 09
 *
 *                                  File : main.cu
 *
 *                               Purpose : Naive Matrix Multiplication
 *
 *************************************************************************************************/

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chCommandLine.h>
#include <chTimer.hpp>
#include <cuda_runtime.h>

#include "mmult_cpu.h"

const static int DEFAULT_MATRIX_WIDTH  = 1024;
const static int DEFAULT_BLOCK_DIM     =   32;

//
// Function Prototypes
//
void printHelp(char * /*programName*/);

//
// matMul_Kernel
//
__global__
void
matMul_Kernel( int matrixWidth, float* matrixA, float* matrixB, float* matrixC )
{
    int elementIdx = blockIdx.x * blockDim.x + threadIdx.x;  // column
    int elementIdy = blockIdx.y * blockDim.y + threadIdx.y;  // row

    float Pvalue = 0; // intermediate result
    float Melement, Nelement;

    int elementId = elementIdy * matrixWidth + elementIdx;
    
    if ( elementIdx < matrixWidth && elementIdy < matrixWidth) {
        Pvalue = 0;
        for ( int k = 0; k < matrixWidth; ++k ) {
                Melement = matrixA[elementIdy * matrixWidth + k];
                Nelement = matrixB[k * matrixWidth + elementIdx];
                Pvalue += Melement * Nelement;
            }
        matrixC[elementId] = Pvalue;
    }
}


__global__
void
shMatMul_Kernel( int matrixWidth, int TILEWIDTH, float* matrixA, float* matrixB, float* matrixC )
{
    extern __shared__ float sh_Mem[];
    float *Ads = &( sh_Mem[0] );
    float *Bds = &( sh_Mem[TILEWIDTH * TILEWIDTH] );

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    int elementIdx = bx * TILEWIDTH + tx;  // column
    int elementIdy = by * TILEWIDTH + ty;  // row

    int elementId = elementIdy * matrixWidth + elementIdx;
    float Pvalue = 0;

    if ( elementIdx < matrixWidth && elementIdy < matrixWidth ) {
        for ( int m = 0; m < matrixWidth / TILEWIDTH; ++m ) { // loop over tiles
            // Collaborative loading of Md and Nd tiles into shared memory.
            Ads[ty * TILEWIDTH + tx] = matrixA[elementIdy * matrixWidth + ( m * TILEWIDTH + tx )];
            Bds[ty * TILEWIDTH + tx] = matrixB[elementIdx + ( m * TILEWIDTH + ty ) * matrixWidth];

            __syncthreads();

            for ( int k = 0; k < TILEWIDTH; ++k )
                Pvalue += Ads[ty * TILEWIDTH + k] * Bds[k * TILEWIDTH + tx];

            __syncthreads();
        }
        matrixC[elementId] = Pvalue;
    }
}


//
// Main
//
int
main(int argc, char * argv[])
{
    //
    // Show Help
    //
    bool showHelp = chCommandLineGetBool("h", argc, argv);
    if (!showHelp) {
        showHelp = chCommandLineGetBool("help", argc, argv);
    }

    if (showHelp) {
        printHelp(argv[0]);
        exit(0);
    }

    // std::cout << "***" << std::endl
    //           << "*** Starting ..." << std::endl
    //           << "***" << std::endl;

    ChTimer memCpyH2DTimer, memCpyD2HTimer;
    ChTimer kernelTimer;
    ChTimer CPUTimer;
    //
    // Allocate Memory
    //
    int matrixWidth = 0;
    chCommandLineGet<int>(&matrixWidth, "s", argc, argv);
    chCommandLineGet<int>(&matrixWidth, "size", argc, argv);
    matrixWidth = matrixWidth != 0 ? matrixWidth : DEFAULT_MATRIX_WIDTH;

    int matrixSize = matrixWidth * matrixWidth;

    //
    // Host Memory
    //
    bool pinnedMemory = chCommandLineGetBool("p", argc, argv);
    if (!pinnedMemory) {
        pinnedMemory = chCommandLineGetBool("pinned-memory",argc,argv);
    }

    float* h_matrixA = NULL;
    float* h_matrixB = NULL;
    float* h_matrixC = NULL;
    if (!pinnedMemory) {
        // Pageable
        h_matrixA = static_cast<float*>(malloc(
                        static_cast<size_t>(matrixSize * sizeof(*h_matrixA))));
        h_matrixB = static_cast<float*>(malloc(
                        static_cast<size_t>(matrixSize * sizeof(*h_matrixB))));
        h_matrixC = static_cast<float*>(calloc(
                        static_cast<size_t>(matrixSize), sizeof *h_matrixC));

    } else {
        // Pinned
        cudaMallocHost(&h_matrixA, static_cast<size_t>(matrixSize * sizeof(*h_matrixA)));
        cudaMallocHost(&h_matrixB, static_cast<size_t>(matrixSize * sizeof(*h_matrixB)));
        cudaMallocHost(&h_matrixC, static_cast<size_t>(matrixSize * sizeof(*h_matrixC)));
        memset ( h_matrixC, 0, matrixSize * sizeof(*h_matrixC) );
    }

    //
    // Device Memory
    //
    float* d_matrixA = NULL;
    float* d_matrixB = NULL;
    float* d_matrixC = NULL;
    cudaMalloc(&d_matrixA, static_cast<size_t>(matrixSize * sizeof(*d_matrixA)));
    cudaMalloc(&d_matrixB, static_cast<size_t>(matrixSize * sizeof(*d_matrixB)));
    cudaMalloc(&d_matrixC, static_cast<size_t>(matrixSize * sizeof(*d_matrixC)));

    //
    // Check Pointers
    //
    if (h_matrixA == NULL || h_matrixB == NULL || h_matrixC == NULL ||
        d_matrixA == NULL || d_matrixB == NULL || d_matrixC == NULL )
    {
        std::cout << "\033[31m***" << std::endl
                  << "*** Error - Allocation of Memory failed!!!" << std::endl
                  << "***\033[0m" << std::endl;
        exit(-1);
    }

    //
    // Init Matrices
    //
    for (int i = 0; i < matrixSize; i++) {
        int x = i % matrixWidth;
        int y = i / matrixWidth;
        h_matrixA[i] = static_cast<float>(x * y);
        h_matrixB[i] = static_cast<float>(x + y);
    }

    //
    // Copy Data to the Device
    //
    memCpyH2DTimer.start();

    cudaMemcpy(d_matrixA, h_matrixA, static_cast<size_t>(matrixSize * sizeof(*d_matrixA)), 
            cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixB, h_matrixB, static_cast<size_t>(matrixSize * sizeof(*d_matrixB)), 
            cudaMemcpyHostToDevice);

    memCpyH2DTimer.stop();

    //
    // Get Kernel Launch Parameters
    //
    int blockSize = 0,
        gridSize = 0;

    // Block Dimension / Threads per Block
    chCommandLineGet<int>(&blockSize,"t", argc, argv);
    chCommandLineGet<int>(&blockSize,"threads-per-block", argc, argv);
    blockSize = blockSize != 0 ? blockSize : DEFAULT_BLOCK_DIM;

    if (blockSize > 32) {
        std::cout << "\033[31m***" << std::endl
                  << "*** Error - The number of threads per block is too big" << std::endl
                  << "***\033[0m" << std::endl;
        exit(-1);
    }

    gridSize = ceil(static_cast<float>(matrixWidth) / static_cast<float>(blockSize));

    dim3 grid_dim = dim3(gridSize, gridSize, 1);
    dim3 block_dim = dim3(blockSize, blockSize, 1);

    // std::cout << "***" << std::endl
    //           << "*** Grid Dim:  " << grid_dim.x << "x" << grid_dim.y << "x" << grid_dim.z 
    //                   << std::endl
    //           << "*** Block Dim: " << block_dim.x << "x" << block_dim.y << "x" << block_dim.z 
    //                   << std::endl
    //           << "***" << std::endl;

    // TODO Calculate shared mem size
    int TILEWIDTH = blockSize;
    // Note: Holds the tiles from both input matrices.
    int sharedMemSize = 2 * TILEWIDTH * TILEWIDTH * sizeof(float);

    // std::cout << "*** MatrixWidth:"<< matrixWidth << std::endl;
    // if (chCommandLineGetBool("shared", argc, argv)) {
    //     std::cout << "*** Tilewidth:"<<TILEWIDTH << std::endl;
    // }
    kernelTimer.start();

    //
    // Launch Kernel
    //
    if (chCommandLineGetBool("shared", argc, argv)) {
        // std::cout <<"*** shared"<< std::endl;
        shMatMul_Kernel<<<grid_dim, block_dim, sharedMemSize>>> ( matrixWidth, TILEWIDTH, d_matrixA, d_matrixB, d_matrixC );
    } else {
        // std::cout <<"*** nonshared"<< std::endl;
        matMul_Kernel<<<grid_dim, block_dim>>> ( matrixWidth, d_matrixA, d_matrixB, d_matrixC );
    }

    //
    // Synchronize
    //
    cudaDeviceSynchronize();

    //
    // Check for Errors
    //
    cudaError_t cudaError = cudaGetLastError();
    if ( cudaError != cudaSuccess ) {
        std::cout << "\033[31m***" << std::endl
                  << "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
                    << std::endl
                  << "***\033[0m" << std::endl;

        return -1;
    }

    kernelTimer.stop();

    //
    // Copy Back Data
    //
    memCpyD2HTimer.start();

    cudaMemcpy(h_matrixC, d_matrixC, static_cast<size_t>(matrixSize * sizeof(*d_matrixC)), 
            cudaMemcpyDeviceToHost);

    memCpyD2HTimer.stop();

    //
    // Check Result
    //
    bool dontCheckResult = chCommandLineGetBool("c", argc, argv);
    if (!dontCheckResult) {
        dontCheckResult = chCommandLineGetBool("no-check", argc, argv);
    }

    if (!dontCheckResult) {
        float* h_matrixD = static_cast<float*>(
                calloc(static_cast<size_t>(matrixSize), sizeof(*h_matrixD)));

        CPUTimer.start();

        MatrixMulOnHostBlocked(h_matrixA, h_matrixB, h_matrixD, 
                static_cast<long>(matrixWidth), 32);
        CPUTimer.stop();

        bool resultOk = MatrixCompare(h_matrixC, h_matrixD, 
                static_cast<long>(matrixWidth));

        if (!resultOk) {
            std::cout << "\033[31m***" << std::endl
                      << "*** Error - The two matrices are different!!!" << std::endl
                      << "***\033[0m" << std::endl;

            exit(-1);
        }

        free(h_matrixD);
    }

    //
    // Print Meassurement Results
    //
    // Print header without bandwith.
    if (chCommandLineGetBool("print-header", argc, argv))
        std::cout << "type,matrix_width,grid_dim,block_dim,t_copy_to_device,t_copy_from_device,t_mm" << std::endl;
    
    // Print measurements
    std::cout << chCommandLineGetBool("shared", argc, argv) << ','
              << matrixWidth << ','
              << gridSize << ','
              << blockSize << ','
              << 1e3 * memCpyH2DTimer.getTime() << ','
              << 1e3 * memCpyD2HTimer.getTime() << ','
              << 1e3 * kernelTimer.getTime() << ','
              << std::endl;

    
    // std::cout << "***" << std::endl
    //           << "*** Results:" << std::endl
    //           << "***    Matrix Size: " << matrixSize << std::endl
    //           << "***    Time to Copy to Device: " << 1e3 * memCpyH2DTimer.getTime()
    //             << " ms" << std::endl
    //           << "***    Copy Bandwidth: " 
    //             << 1e-9 * memCpyH2DTimer.getBandwidth(2 * matrixSize * sizeof(*h_matrixA))
    //             << " GB/s" << std::endl
    //           << "***    Time to Copy from Device: " << 1e3 * memCpyD2HTimer.getTime()
    //             << " ms" << std::endl
    //           << "***    Copy Bandwidth: " 
    //             << 1e-9 * memCpyD2HTimer.getBandwidth(matrixSize * sizeof(*h_matrixA))
    //             << " GB/s" << std::endl
    //           << "***    Time for Matrix Multiplication: " << 1e3 * kernelTimer.getTime()
    //               << " ms" << std::endl
    //           << "***" << std::endl;
    
    if (!dontCheckResult) {
         std::cout << "***  CPU time needed for the same Problem : "<< 1e3 * CPUTimer.getTime()
                << " ms" << std::endl;


    }
    if (chCommandLineGetBool("print-matrix", argc, argv) 
       && matrixWidth <= 16) {
        printOutMatrix(h_matrixC, matrixWidth);
    }

    // Free Memory
    if (!pinnedMemory) {
        free(h_matrixA);
        free(h_matrixB);
        free(h_matrixC);
    } else {
        cudaFreeHost(h_matrixA);
        cudaFreeHost(h_matrixB);
        cudaFreeHost(h_matrixC);
    }
    cudaFree(d_matrixA);
    cudaFree(d_matrixB);
    cudaFree(d_matrixC);

    return 0;
}

void
printHelp(char * programName)
{
    std::cout << "Help:" << std::endl
              << "  Usage: " << std::endl
              << "  " << programName << " [-p] [-s <matrix_size>] [-t <threads_per_block>]" 
                << std::endl
              << "                 [-g <blocks_per_grid] [-c] [--print-matrix]" 
                << std::endl
              << "" << std::endl
              << "  -p|--pinned-memory" << std::endl
              << "  Use pinned Memory instead of pageable memory" << std::endl
              << "" << std::endl
              << "  -s <matrix_size>|--size <matix_size>" << std::endl
              << "  The width of the Matrix" << std::endl
              << "" << std::endl
              << "  -t <threads_per_block>|--threads-per-block <threads_per_block>" 
                << std::endl
              << "  The number of threads per block" << std::endl
              << "" << std::endl
              << "  -c|--no-checking" << std::endl
              << "  Do not check the result of the matrix multiplication" << std::endl
              << "" << std::endl
              << "  --print-matrix" << std::endl
              << "  Print the output matrix (only recommended for small matrices)" << std::endl
              << std::endl;
}

