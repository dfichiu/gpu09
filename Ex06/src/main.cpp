/**************************************************************************************************
 *
 *       Computer Engineering Group, Heidelberg University - GPU Computing Exercise 06
 *
 *                 Gruppe : 09
 *
 *                   File : main.cpp
 *
 *                Purpose : Reduction
 *
 **************************************************************************************************/

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <chCommandLine.h>
#include <chTimer.hpp>
#include <cuda_runtime.h>

const static int DEFAULT_MATRIX_SIZE = 1024;
const static int DEFAULT_BLOCK_DIM   = 128;

//
// Function Prototypes
//
void printHelp(char *);


extern void initial_Kernel_Wrapper(
	dim3 gridSize,
	dim3 blockSize, 
	int numElements,
	float* dataIn,
	float* dataOut
);

extern void optimized_Kernel_Wrapper(
	dim3 gridSize,
	dim3 blockSize,
	int shMemSize,
	int numElements,
	float* dataIn,
	float* dataOut);

extern void reduction_Kernel_Wrapper(
	dim3 gridSize,
	dim3 blockSize,
	int shMemSize,
	int numElements,
	float* dataIn,
	float* dataOut);

extern void thrust_reduction_Wrapper(int numElements, float* dataIn, float* dataOut);

//
// Main
//
int
main(int argc, char * argv[])
{
	bool showHelp = chCommandLineGetBool("h", argc, argv);
	if (!showHelp)
	{
		showHelp = chCommandLineGetBool("help", argc, argv);
	}

	if (showHelp)
	{
		printHelp(argv[0]);
		exit(0);
	}

	std::cout << "***" << std::endl
			  << "*** Starting ..." << std::endl
			  << "***" << std::endl;

	ChTimer memCpyH2DTimer, memCpyD2HTimer;
	ChTimer kernelTimer;

	//
	// Allocate Memory
	//
	int numElements = 0;
	chCommandLineGet<int>(&numElements, "s", argc, argv);
	chCommandLineGet<int>(&numElements, "size", argc, argv);
	numElements = numElements != 0 ?
			numElements : DEFAULT_MATRIX_SIZE;
	//
	// Host Memory
	//
	bool pinnedMemory = chCommandLineGetBool("p", argc, argv);
	if (!pinnedMemory)
	{
		pinnedMemory = chCommandLineGetBool("pinned-memory", argc, argv);
	}

	float* h_dataIn = NULL;
	float* h_dataOut = NULL;
	if (!pinnedMemory)
	{
		// Pageable
		h_dataIn = static_cast<float*>
				(malloc(static_cast<size_t>(numElements * sizeof(*h_dataIn))));
		h_dataOut = static_cast<float*>
				(malloc(static_cast<size_t>(sizeof(*h_dataOut))));
	}
	else
	{
		// Pinned
		cudaMallocHost(&h_dataIn, 
				static_cast<size_t>(numElements * sizeof(*h_dataIn)));
		cudaMallocHost(&h_dataOut, 
				static_cast<size_t>(sizeof(*h_dataOut)));
	}
	// Init host data
	*h_dataOut = 0;
	for (int i=0; i<numElements; i++) {
		h_dataIn[i] = 1.;
	}

	// Device Memory
	float* d_dataIn = NULL;
	float* d_dataOut = NULL;
	cudaMalloc(&d_dataIn, 
			static_cast<size_t>(numElements * sizeof(*d_dataIn)));
	cudaMalloc(&d_dataOut, 
			static_cast<size_t>(sizeof(*d_dataOut)));

	if (h_dataIn == NULL || h_dataOut == NULL ||
		d_dataIn == NULL || d_dataOut == NULL)
	{
		std::cout << "\033[31m***" << std::endl
		          << "*** Error - Memory allocation failed" << std::endl
		          << "***\033[0m" << std::endl;

		exit(-1);
	}

	//
	// Copy Data to the Device
	//
	memCpyH2DTimer.start();

	cudaMemcpy(d_dataIn, h_dataIn, 
			static_cast<size_t>(numElements * sizeof(*d_dataIn)), 
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_dataOut, h_dataOut, 
			static_cast<size_t>(sizeof(*d_dataOut)), 
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
	blockSize = blockSize != 0 ?
			blockSize : DEFAULT_BLOCK_DIM;

	if (blockSize > 1024)
	{
		std::cout << "\033[31m***" << std::endl
		          << "*** Error - The number of threads per block is too big" << std::endl
		          << "***\033[0m" << std::endl;

		exit(-1);
	}

	gridSize = ceil(static_cast<float>(numElements) / static_cast<float>(blockSize));
	// std::cout << "** GRID DIM" << gridSize << std::endl;/
	// std::cout << "** BLOCK DIM" << blockSize << std::endl;
	dim3 grid_dim = dim3(gridSize);
	dim3 block_dim = dim3(blockSize);

	kernelTimer.start();
	
	if (chCommandLineGetBool("thrust", argc, argv)) {
		thrust_reduction_Wrapper(numElements, d_dataIn, d_dataOut);
	} else if (chCommandLineGetBool("init", argc, argv)) {
		std::cout<< "*** Initial kernel "<< std::endl;
		initial_Kernel_Wrapper(grid_dim, block_dim, numElements, d_dataIn, d_dataOut);
		initial_Kernel_Wrapper(1, grid_dim, gridSize, d_dataOut, d_dataOut);
	} else if (chCommandLineGetBool("optimized", argc, argv)) {
		std::cout<< "*** Optimized kernel"<< std::endl;
		optimized_Kernel_Wrapper(
			grid_dim,
			block_dim,
			blockSize * sizeof(float),
			numElements,
			d_dataIn,
			d_dataOut
		);
		optimized_Kernel_Wrapper(
			1,
			grid_dim,
			gridSize * sizeof(float),
			gridSize, 
			d_dataOut, 
			d_dataOut
		);
	} else {
		std::cout<< "*** Reduction kernel"<< std::endl;
		reduction_Kernel_Wrapper(
			grid_dim,
			block_dim,
			blockSize * sizeof(float),
			numElements,
			d_dataIn,
			d_dataOut
		);
		reduction_Kernel_Wrapper(
			1,
			grid_dim,
			gridSize * sizeof(float),
			gridSize, 
			d_dataOut, 
			d_dataOut
		);
	}

	// Synchronize
	cudaDeviceSynchronize();

	// Check for Errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess)
	{
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

	cudaMemcpy(h_dataOut, d_dataOut, 
			static_cast<size_t>(sizeof(*d_dataOut)), 
			cudaMemcpyDeviceToHost);

	memCpyD2HTimer.stop();


	
	ChTimer CPUTimer;
	
	float* arr = new float[numElements];
	for (int i = 0; i < numElements; ++i)
		arr[i] = 1.0;
	int result = 0;
	 
	CPUTimer.start();
	if (chCommandLineGetBool("cpu", argc, argv)) {
		for (int i = 0; i < numElements; i++) {
  			result += arr[i];
		} 
	}
	CPUTimer.stop();
	delete arr;

	
	std::cout << "***RESULTAT CPU:" << result<<std::endl; 
	// Print Meassurement Results
	std::cout << "***" << std::endl
			  << "*** Results:" << std::endl
			  << "***    h_dataOut: " << *h_dataOut << std::endl
			  << "***    Num Elements: " << numElements << std::endl
			  << "***    Time to Copy to Device: " << 1e3 * memCpyH2DTimer.getTime()
			  	<< " ms" << std::endl
			  << "***    Copy Bandwidth: " 
			  	<< 1e-9 * memCpyH2DTimer.getBandwidth(numElements * sizeof(*h_dataIn))
			  	<< " GB/s" << std::endl
			  << "***    Time to Copy from Device: " << 1e3 * memCpyD2HTimer.getTime()
			  	<< " ms" << std::endl
			  << "***    Copy Bandwidth: " 
			  	<< 1e-9 * memCpyD2HTimer.getBandwidth(sizeof(*h_dataOut))
				<< " GB/s" << std::endl
			  << "***    Time for GPU Reduction: " << 1e3 * kernelTimer.getTime()
			    << " ms" << std::endl
			  << "***    GPU Reduction Bandwidth: " 
			  	<< 1e-9 * kernelTimer.getBandwidth(numElements * sizeof(float))
				<< " GB/s" << std::endl
			  << "***" << std::endl;

	if (chCommandLineGetBool("cpu", argc, argv)) {
		std::cout << "***    Time for CPU Reduction: " << 1e3 * CPUTimer.getTime()
				    << " ms" << std::endl
				  << "***    CPU Reduction Bandwidth: " 
			  	    << 1e-9 * CPUTimer.getBandwidth(numElements * sizeof(float))
				    << " GB/s" << std::endl;
	}

	// Free Memory
	if (!pinnedMemory)
	{
		free(h_dataIn);
		free(h_dataOut);
	}
	else
	{
		cudaFreeHost(h_dataIn);
		cudaFreeHost(h_dataOut);
	}
	cudaFree(d_dataIn);
	cudaFree(d_dataOut);
	
	return 0;
}

void
printHelp(char * argv)
{
	std::cout << "Help:" << std::endl
			  << "  Usage: " << std::endl
			  << "  " << argv << " [-p] [--thrust] [-s <num-elements>] [-t <threads_per_block>]"
			  	<< std::endl
			  << "" << std::endl
			  << "  -p|--pinned-memory" << std::endl
			  << "	Use pinned Memory instead of pageable memory" << std::endl
			  << "" << std::endl
			  << "  --thrust" << std::endl
			  << "	Use CUDA Thrust reduction instead of custom kernel" << std::endl
			  << "" << std::endl
			  << "  -s <num-elements>|--size <num-elements>" << std::endl
			  << "	The size of the Matrix" << std::endl
			  << "" << std::endl
			  << "  -t <threads_per_block>|--threads-per-block <threads_per_block>" 
			  	<< std::endl
			  << "	The number of threads per block" << std::endl
			  << "" << std::endl;
}
