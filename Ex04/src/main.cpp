/******************************************************************************
 *
 *Computer Engineering Group, Heidelberg University - GPU Computing Exercise 04
 *
 *                  Group : 09
 *
 *                   File : main.cpp
 *
 *                Purpose : Memory Operations Benchmark
 *
 ******************************************************************************/

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <chCommandLine.h>
#include <chTimer.hpp>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cmath>

const static int DEFAULT_MEM_SIZE       = 10 * 1024 * 1024; // 10 MB
const static int DEFAULT_NUM_ITERATIONS =         1000;
const static int DEFAULT_block_size     =          128;
const static int DEFAULT_grid_size      =           1;
const static int FLOAT_SIZE             =           4; // 4B

//
// Function Prototypes
//
void printHelp(char *);

//
// Kernel Wrappers
//
extern void GlobalMem2SharedMem_Wrapper(
	/* Kernel execution config param */
	dim3 gridSize,
	dim3 blockSize,
	int memorySize,
	/* Kernel arguments */
	float* d_memory,
	int elemMemory,
	int elemThread
);
extern void SharedMem2GlobalMem_Wrapper(
	/* Kernel execution config param */
	dim3 gridSize,
	dim3 blockSize,
	int memorySize,
	/* Kernel arguments */
	float* d_memory,
	int elemMemory,
	int elemThread
);
extern void SharedMem2Registers_Wrapper(
	/* Kernel execution config param */
	dim3 gridSize,
	dim3 blockSize,
	int memorySize,
	/* Kernel arguments */
	int registerMemory,
	int threadMemory,
	float* outFloat
);
extern void Registers2SharedMem_Wrapper(
	/* Kernel execution config param */
	dim3 gridSize,
	dim3 blockSize,
	int memorySize,
	/* Kernel arguments */
	int registerMemory,
	int threadMemory,
	float* outFloat
);
extern void BankConflictsRead_Wrapper(
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
);


//
// Main
//
int
main ( int argc, char * argv[] )
{
	// Show Help
	bool optShowHelp = chCommandLineGetBool("h", argc, argv);
	if ( !optShowHelp )
		optShowHelp = chCommandLineGetBool("help", argc, argv);

	if ( optShowHelp ) {
		printHelp ( argv[0] );
		exit (0);
	}

	// std::cout << "***" << std::endl
	//           << "*** Starting ..." << std::endl
	// 		  << "***" << std::endl;

	ChTimer kernelTimer;

	//
	// Get kernel launch parameters and configuration
	//
	int optNumIterations = 0,
		optBlockSize = 0,
		optGridSize = 0;

	// Number of Iterations
	chCommandLineGet<int> ( &optNumIterations, "i", argc, argv );
	chCommandLineGet<int> ( &optNumIterations, "iterations", argc, argv );
	optNumIterations = ( optNumIterations != 0 ) ? optNumIterations : DEFAULT_NUM_ITERATIONS;

	// Block Dimension / Threads per Block
	chCommandLineGet <int> ( &optBlockSize,"t", argc, argv );
	chCommandLineGet <int> ( &optBlockSize,"threads-per-block", argc, argv );
	optBlockSize = optBlockSize != 0 ? optBlockSize : DEFAULT_block_size;

	if ( optBlockSize > 1024 ) {
		std::cout << "\033[31m***" << std::endl
		          << "*** Error - The number of threads per block is too big" << std::endl
		          << "***\033[0m" << std::endl;

		exit(-1);
	}

	// Grid Dimension
	chCommandLineGet <int> ( &optGridSize, "g", argc, argv );
	chCommandLineGet <int> ( &optGridSize, "grid-dim", argc, argv );
	optGridSize = optGridSize != 0 ? optGridSize : DEFAULT_grid_size;
	
	dim3 grid_size = dim3 ( optGridSize );
	dim3 block_size = dim3 ( optBlockSize );
	
	int optModulo = 32*1024; // modulo in access pattern for conflict test
	chCommandLineGet <int> ( &optModulo, "mod", argc, argv );

	// Stride
	int optStride = 1;
	chCommandLineGet <int> ( &optStride, "stride", argc, argv );

	// Memory size
	int optMemorySize = 0;
	
	chCommandLineGet <int> ( &optMemorySize, "s", argc, argv );
	chCommandLineGet <int> ( &optMemorySize, "size", argc, argv );
	optMemorySize = optMemorySize != 0 ? optMemorySize : DEFAULT_MEM_SIZE;
	

	//
	// Device Memory
	//
	float* d_memory = NULL;
	// Note: The allocated memory is not cleared.
	cudaMalloc ( &d_memory, static_cast <size_t> ( optMemorySize ) ); // optMemorySize is **in bytes**

	float *outFloat = NULL;  // dummy variable to prevent compiler optimizations
	cudaMalloc ( &outFloat, static_cast <float> ( sizeof ( float ) ) );
		
	long hClocks = 0;
	long *dClocks = NULL;
	cudaMalloc ( &dClocks, sizeof ( long ) );
	
	if ( d_memory == NULL || dClocks == NULL )
	{
		std::cout << "\033[31m***" << std::endl
		          << "*** Error - Memory allocation failed" << std::endl
		          << "***\033[0m" << std::endl;

		exit (-1);
	}

	//
	// Part I: Shared memory & Global memory view => Entire resource (i.e., optMemorySize)
	// is split between thread blocks.
	// 
	// The (total) no. of (data) elements the memory can hold.
	int elemMemory = std::floor(optMemorySize / float(FLOAT_SIZE));

	// Derive shared memory size **in bytes**.
	int memorySize = elemMemory * FLOAT_SIZE;

	// The (total) no. of (data) elements each thread should copy.
	int elemThread = std::ceil(elemMemory / float(optBlockSize * optGridSize));

	//
	// Part II: Shared memory & Register => Entire resource (i.e., optMemorySize) is owned
	// by a single thread block.
	/*
		The shared memory is 48KB (per thread block).
		Every thread block has 64k 32=bit (4-byte) registers.
		A float has 4 bytes.
	*/
	int registerMemory = std::floor(optMemorySize / float(FLOAT_SIZE));

	int registerThread = std::ceil(registerMemory / float(optBlockSize));

	//
	// Part III: Bank conflicts
	//
	int elemBankMemory = 32 * 1024 / FLOAT_SIZE;
	int bankMemorySize = 32 * 1024;

	//
	// Tests
	//
	// Start timer.
	kernelTimer.start();
	// for ( int i = 0; i < optNumIterations; i++ )
	{
		//
		// Launch Kernel
		//
		// std::cout
		// 	<< "Starting kernel: "
		// 	<< grid_size.x << "x" << block_size.x << " threads, "
		// 	<< optMemorySize << "B shared memory, "
		// 	<< optNumIterations << " iterations"
		// 	<< std::endl;

		if ( chCommandLineGetBool ( "global2shared", argc, argv ) )
		{
			GlobalMem2SharedMem_Wrapper(
				/* Kernel execution config param */
				grid_size,
				block_size,
				memorySize,  // Shared mem. size **in bytes**
				/* Kernel arguments */
				d_memory,  // Pointer to device mem.
				elemMemory,  // Total no. of elements in the memory
				elemThread  // No. of elements each thread should copy
			);
		}
		else if ( chCommandLineGetBool ( "shared2global", argc, argv ) )
		{
			SharedMem2GlobalMem_Wrapper(
				/* Kernel execution config param */
				grid_size,
				block_size,
				memorySize,
				/* Kernel arguments */
				d_memory,
				elemMemory,
				elemThread
			);
		}
		else if ( chCommandLineGetBool ( "shared2register", argc, argv ) )
		{
			SharedMem2Registers_Wrapper(
				/* Kernel execution config param */
				grid_size,
				block_size,
				memorySize,
				/* Kernel arguments */
				registerMemory,
				registerThread,
				outFloat
			);
		}
		else if ( chCommandLineGetBool ( "register2shared", argc, argv ) )
		{
			Registers2SharedMem_Wrapper( 
				/* Kernel execution config param */
				grid_size,
				block_size,
				memorySize,
				/* Kernel arguments */
				registerMemory,
				registerThread,
				outFloat
			);
		}
		else if ( chCommandLineGetBool ( "shared2register_conflict", argc, argv ) )
		{
			BankConflictsRead_Wrapper(
				/* Kernel execution config param */
				1, 
				block_size, 
				bankMemorySize,
				/* Kernel arguments */
				elemBankMemory,
				optStride,
				optNumIterations,
				dClocks,
				outFloat
			);
		}
	}

	// Mandatory synchronize after all kernel launches
	cudaDeviceSynchronize();
	// Stop timer.
	kernelTimer.stop();

	cudaError_t cudaError = cudaGetLastError();
	if ( cudaError != cudaSuccess )
	{
		std::cout << "\033[31m***" << std::endl
	    	      << "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
				  << std::endl
		          << "***\033[0m" << std::endl;

		return -1;
	}

	//
	// Print Measurement Results
	//
	if ( chCommandLineGetBool ( "global2shared", argc, argv ) ) {
		std::cout << "Copy global->shared, size=" << std::setw(10) << optMemorySize << ", gDim=" << std::setw(5) << grid_size.x << ", bDim=" << std::setw(5) << block_size.x;
		//std::cout << ", time=" << kernelTimer.getTime(optNumIterations) << 
		std::cout.precision ( 2 );
		std::cout << ", bw=" << std::fixed << std::setw(6) << ( optMemorySize * grid_size.x ) / kernelTimer.getTime(optNumIterations) / (1E09) << "GB/s" << std::endl;
		}   
	
	if ( chCommandLineGetBool ( "shared2global", argc, argv ) ) {
		std::cout << "Copy shared->global, size=" << std::setw(10) << optMemorySize << ", gDim=" << std::setw(5) << grid_size.x << ", bDim=" << std::setw(5) << block_size.x;
		//std::cout << ", time=" << kernelTimer.getTime(optNumIterations) << 
		std::cout.precision ( 2 );
		std::cout << ", bw=" << std::fixed << std::setw(6) << ( optMemorySize * grid_size.x ) / kernelTimer.getTime(optNumIterations) / (1E09) << "GB/s" << std::endl;
		//std::cout << ", " << std::fixed << std::setw(6) << ( optMemorySize * grid_size.x ) / kernelTimer.getTime(optNumIterations) / (1E09) ;
	
	}

	if ( chCommandLineGetBool ( "shared2register", argc, argv ) ) {
		//std::cout << "Copy shared->register, size=" << std::setw(10) << optMemorySize << ", gDim=" << std::setw(5) << grid_size.x << ", bDim=" << std::setw(5) << block_size.x;
		//std::cout << ", time=" << kernelTimer.getTime(optNumIterations) << 
		std::cout.precision ( 2 );
		//std::cout << ", bw=" << std::fixed << std::setw(6) << ( optMemorySize * grid_size.x ) / kernelTimer.getTime(optNumIterations) / (1E09) << "GB/s" << std::endl;
	    std::cout << ", " << std::fixed << std::setw(6) << ( optMemorySize * grid_size.x ) / kernelTimer.getTime(optNumIterations) / (1E09);
	}
	
	if ( chCommandLineGetBool ( "register2shared", argc, argv ) ) {
		//std::cout << "Copy register->shared, size=" << std::setw(10) << optMemorySize << ", gDim=" << std::setw(5) << grid_size.x << ", bDim=" << std::setw(5) << block_size.x;
		//std::cout << ", time=" << kernelTimer.getTime(optNumIterations) << 
		std::cout.precision ( 2 );
		//std::cout << ", bw=" << std::fixed << std::setw(6) << ( optMemorySize * grid_size.x ) / kernelTimer.getTime(optNumIterations) / (1E09) << "GB/s" << std::endl;
	    std::cout << ", " << std::fixed << std::setw(6) << ( optMemorySize * grid_size.x ) / kernelTimer.getTime(optNumIterations) / (1E09);
	
	}	

	if ( chCommandLineGetBool ( "shared2register_conflict", argc, argv ) ) {
		if ( chCommandLineGetBool ( "shared2register_conflict", argc, argv ) ) {
			cudaError_t error = cudaMemcpy ( &hClocks, dClocks, sizeof ( long ), cudaMemcpyDeviceToHost );
			if ( error != cudaSuccess) {
				fprintf ( stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString ( error ) );
				return 1;
			}
		}	

		std::cout << "Shared memory bank conflict test, size=" << std::setw(10) << 32 * 1024 << ", gDim=" << std::setw(5) << grid_size.x << ", bDim=" << std::setw(5) << block_size.x;
		std::cout << ", stride=" << std::setw(6) << optStride << ", modulo=" << std::setw(6) << optModulo;
		std::cout << ", clocks=" << std::setw(10) << hClocks << std::endl;
	}
	
	return 0;
}

void
printHelp(char * programName)
{
	std::cout 	
		<< "Usage: " << std::endl
		<< "  " << programName << " [-p] [-s <memory_size>] [-i <num_iterations>]" << std::endl
		<< "                [-t <threads_per_block>] [-g <blocks_per_grid]" << std::endl
		<< "                [-stride <stride>] [-offset <offset>]" << std::endl
		<< "  --{global2shared,shared2global,shared2register,register2shared,shared2register_conflict}" << std::endl
		<< "    Run kernel analyzing shared memory performance" << std::endl
		<< "  -s <memory_size>|--size <memory_size>" << std::endl
		<< "    The amount of memory to allcate" << std::endl
		<< "  -t <threads_per_block>|--threads-per-block <threads_per_block>" << std::endl
		<< "    The number of threads per block" << std::endl
		<< "  -g <blocks_per_grid>|--grid-dim <blocks_per_grid>" << std::endl
		<< "     The number of blocks per grid" << std::endl
		<< "  -i <num_iterations>|--iterations <num_iterations>" << std::endl
		<< "     The number of iterations to launch the kernel" << std::endl
		<< "  -stride <stride>" << std::endl
		<< "     Stride parameter for global-stride test. Not that size parameter is ignored then." << std::endl
		<< "  -offset <offset>" << std::endl
		<< "     Offset parameter for global-offset test. Not that size parameter is ignored then." << std::endl
		<< "" << std::endl;
}
