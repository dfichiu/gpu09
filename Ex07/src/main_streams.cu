/******************************************************************************
 *
 *           XXXII Heidelberg Physics Graduate Days - GPU Computing
 *
 *                 Group : 09
 *
 *                   File : main_stream.cu
 *
 *                Purpose : n-Body Computation
 *
 ******************************************************************************/

#include <cmath>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <chCommandLine.h>
#include <chTimer.hpp>
#include <cstdio>
#include <iomanip>

const static int DEFAULT_NUM_ELEMENTS = 524288;
const static int DEFAULT_NUM_ITERATIONS = 5;
const static int DEFAULT_BLOCK_SIZE = 1024;
const static int DEFAULT_GRID_SIZE = 64;

const static float TIMESTEP = 1e-4;//1e-6;	  // s
const static float GAMMA = 6.673e-11; // (Nm^2)/(kg^2)


//
// Coalesced AOS (Arrays of structures) packed in a structure for access easiness.
//
struct Body_t
{
	float4 posMass;	 /* x = x */
					 /* y = y */
					 /* z = z */
					 /* w = Mass */
	float3 velocity; /* x = v_x*/
					 /* y = v_y */
					 /* z= v_z */

	Body_t() : posMass(make_float4(0, 0, 0, 0)), velocity(make_float3(0, 0, 0)) {}
};


//
// Function Prototypes
//
void printHelp(char *);
void printElement(float4 *, float3 *, int, int);

//
// Device Functions
//

//
// Calculate the distance between two points.
//
__device__ float
getDistance(float4 a, float4 b)
{
	float dx = b.x - a.x;
	float dy = b.y - a.y;
	float dz = b.z - a.z;

	return sqrt(dx * dx + dy * dy + dz * dz);
}

//
// Calculate the force between two bodies.
//
__device__ void
bodyBodyInteraction(float4 bodyA, float4 bodyB, float3 &force)
{
	float softeningFactor = 0.01;
	float distance = getDistance(bodyA, bodyB);

	if (distance == 0)
		return;

	float distanceSquared = distance * distance;
	distanceSquared += softeningFactor;

	float invSquaredDistance = rsqrtf(distanceSquared);
	float invCubeDistance = invSquaredDistance * invSquaredDistance * invSquaredDistance;

	float constant = GAMMA * bodyA.w * bodyB.w * invCubeDistance;

	force.x = constant * (bodyB.x - bodyA.x);
	force.y = constant * (bodyB.y - bodyA.y);
	force.z = constant * (bodyB.z - bodyA.z);
}

//
// Calculate the new velocity of one particle.
//
__device__ void
calculateVelocity(float mass, float3 &currentVelocity, float3 force)
{
	currentVelocity.x += TIMESTEP * force.x / mass;
    currentVelocity.y += TIMESTEP * force.y / mass;
    currentVelocity.z += TIMESTEP * force.z / mass;
}


//
// Use packed SOA and shared memory to optimize load and store operations.
//
__global__ void
sharedNbody_Kernel(
    int segmentSize,
    float4 *posMasses_current,
    float3 *velocities_current,
    float4 *posMasses_other
)
{
	// Initialize shared memory for the positions and masses of others.
	extern __shared__ float4 sharedPosMass[];

	// Id of the body element associated with the thread.
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	float4 elementPosMass;
	float3 elementForce;
	float3 elementVelocity;

	if (elementId < 2 * segmentSize) {
		elementPosMass = posMasses_current[elementId];  // element processes by the current thread
		elementVelocity = velocities_current[elementId];

		for (int i = 0; i < 2 * segmentSize; i += blockDim.x)
		{
			// Load elements into shared memory.
			sharedPosMass[threadIdx.x] = posMasses_other[i + threadIdx.x];

			// Synchronize threads: read-after-write.
			__syncthreads();

			for (int j = 0; j < blockDim.x; j++)
			{
				elementForce = make_float3(0, 0, 0);

				// Compute body-body interactions (i.e, forces) between
				// the thread-body and the loaded bodies.
				bodyBodyInteraction(elementPosMass, sharedPosMass[j], elementForce);

				// Compute the thread-body velocity.
				calculateVelocity(elementPosMass.w, elementVelocity, elementForce);
			}

			// Synchronize threads: write-after-read.
			__syncthreads(); 
		}
        velocities_current[elementId] = elementVelocity;
	}
}

//
// n-Body Kernel for packed SOA to update the position.
// Needed to prevent write-after-read-hazards.
//
__global__ void
packedSOAUpdatePosition_Kernel(int segmentSize, float4 *posMasses, float3 *velocities)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	if (elementId < segmentSize)
	{
		posMasses[elementId].x += TIMESTEP * velocities[elementId].x;
		posMasses[elementId].y += TIMESTEP * velocities[elementId].y;
		posMasses[elementId].z += TIMESTEP * velocities[elementId].z;
	}
}

//
// Main
//
int main(int argc, char *argv[])
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
	// Set command line arguments.
	//
	bool silent = chCommandLineGetBool("silent", argc, argv);
	// Get number of iterations.
	int numIterations = DEFAULT_NUM_ITERATIONS;

	//
	// Allocate host memory for the particles.
	//
	// Use default number of elements.
	int numElements = DEFAULT_NUM_ELEMENTS;

	//
	// Host Memory
	//
	bool pinnedMemory = chCommandLineGetBool("p", argc, argv);
	if (!pinnedMemory)
	{
		pinnedMemory = chCommandLineGetBool("pinned-memory", argc, argv);
	}

	// Allocate host memory.
	// Packed SOA
	float4 *h_posMasses;
	float3 *h_velocities;

	if (!pinnedMemory)
	{
		// Pageable
		h_posMasses = static_cast<float4 *>(malloc(static_cast<size_t>(numElements * sizeof(*h_posMasses))));
		h_velocities = static_cast<float3 *>(malloc(static_cast<size_t>(numElements * sizeof(*h_velocities))));
	}
	else
	{
		// Pinned
		cudaMallocHost(&h_posMasses, static_cast<size_t>(numElements * sizeof(*h_posMasses)));
		cudaMallocHost(&h_velocities, static_cast<size_t>(numElements * sizeof(*h_velocities)));
	}

	srand(0); // Always use the same random numbers.

	// Initialize particles: packed SOA.
	for (int i = 0; i < numElements; i++)
	{
		h_posMasses[i].x = 1e-8 * static_cast<float>(rand()); // Modify the random values to
		h_posMasses[i].y = 1e-8 * static_cast<float>(rand()); // increase the position changes
		h_posMasses[i].z = 1e-8 * static_cast<float>(rand()); // and the velocity.
		h_posMasses[i].w = 1e4 * static_cast<float>(rand());

		h_velocities[i].x = 0.0f;
		h_velocities[i].y = 0.0f;
		h_velocities[i].z = 0.0f;
	}


    //
    // Intialize grid and block size.
    //
    int blockSize = DEFAULT_BLOCK_SIZE;
    int gridSize = DEFAULT_GRID_SIZE;

	if (blockSize > 1024)
	{
		std::cout << "\033[31m***" << std::endl
				  << "*** Error - The number of threads per block is too big" << std::endl
				  << "***\033[0m" << std::endl;

		exit(-1);
	}

    dim3 block_dim = dim3(blockSize);
    dim3 grid_dim = dim3(gridSize);


	if (!silent)
    {
		std::cout << "***" << std::endl;
		std::cout << "*** Grid: " << gridSize << std::endl;
		std::cout << "*** Block: " << blockSize << std::endl;
		std::cout << "***" << std::endl;
	}


	//
	// Initialize streams.
	//
	// Declare streams.
	cudaStream_t stream0, stream1;

	// Create streams.
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);

	//
	// Initialize per-stream device memory.
	//
	// Device memory for stream0.
	float4 *d_posMasses_current0, *d_posMasses_other0;
	float3 *d_velocities_current0;

	// Device memory for stream1.
	float4 *d_posMasses_current1, *d_posMasses_other1;
	float3 *d_velocities_current1;

	int segmentSize = pow(2, 16);  // two arrays (current & other)/kernel

	// Allocate memory for stream0.
	cudaMalloc(&d_posMasses_current0, static_cast<size_t>(segmentSize * sizeof(*d_posMasses_current0)));
    cudaMalloc(&d_posMasses_other0, static_cast<size_t>(segmentSize * sizeof(*d_posMasses_other0)));
	cudaMalloc(&d_velocities_current0, static_cast<size_t>(segmentSize * sizeof(*d_velocities_current0)));

	// Allocate memory for stream1.
	cudaMalloc(&d_posMasses_current1, static_cast<size_t>(segmentSize * sizeof(*d_posMasses_current1)));
    cudaMalloc(&d_posMasses_other1, static_cast<size_t>(segmentSize * sizeof(*d_posMasses_other1)));
	cudaMalloc(&d_velocities_current1, static_cast<size_t>(segmentSize * sizeof(*d_velocities_current1)));

	// Check for memory (device & host) allocation errors.
	if (
        // Host
		(h_posMasses == NULL) ||
		(h_velocities == NULL) ||
		//
        // Device
		//
		// Stream 0
		(d_posMasses_current0 == NULL) ||
		(d_velocities_current0 == NULL) ||
        (d_posMasses_other0 == NULL) ||
		// Stream 1
		(d_posMasses_current1 == NULL) ||
		(d_velocities_current1 == NULL) ||
        (d_posMasses_other1 == NULL)
	)
	{
		std::cout << "\033[31m***" << std::endl
				  << "*** Error - Memory allocation failed" << std::endl
				  << "***\033[0m" << std::endl;

		exit(-1);
	}

	kernelTimer.start();
	for (int it = 0; it < numIterations; it++)
	{
        std::cout << "*** Iteration " << it + 1 << " ***" << std::endl;
        for (int i = 0; i < numElements; i += 2 * segmentSize)
        {
            // std::cout << "current last element: " << i << std::endl;

            // Copy per stream **current** positions and masses.
            cudaMemcpyAsync(
                d_posMasses_current0,
                h_posMasses + i,  // beginning of memory to copy
                static_cast<size_t>(segmentSize * sizeof(*d_posMasses_current0)),  // how much to copy
                cudaMemcpyHostToDevice,
                stream0
            );
            cudaMemcpyAsync(
                d_posMasses_current1,
                h_posMasses + i + segmentSize,  // beginning of memory to copy
                static_cast<size_t>(segmentSize * sizeof(*d_posMasses_current1)),  // how much to copy
                cudaMemcpyHostToDevice,
                stream1
            );

            // Copy per stream **current** velocities.
            cudaMemcpyAsync(
                d_velocities_current0,
                h_velocities + i,  // beginning of memory to copy
                static_cast<size_t>(segmentSize * sizeof(*d_velocities_current0)),  // how much to copy
                cudaMemcpyHostToDevice,
                stream0
            );
            cudaMemcpyAsync(
                d_velocities_current1,
                h_velocities + i + segmentSize,  // beginning of memory to copy
                static_cast<size_t>(segmentSize * sizeof(*d_velocities_current1)),  // how much to copy
                cudaMemcpyHostToDevice,
                stream1
            );

            for (int j = 0; j < numElements; j += segmentSize)
            {
                // std::cout << "other last element: " << j << std::endl;

                // Copy per stream **current** positions and masses.
                cudaMemcpyAsync(
                    d_posMasses_other0,
                    h_posMasses + j,  // beginning of memory to copy
                    static_cast<size_t>(segmentSize * sizeof(*d_posMasses_other0)),  // how much to copy
                    cudaMemcpyHostToDevice,
                    stream0
                );
                cudaMemcpyAsync(
                    d_posMasses_current1,
                    h_posMasses + j,  // beginning of memory to copy
                    static_cast<size_t>(segmentSize * sizeof(*d_posMasses_other1)),  // how much to copy
                    cudaMemcpyHostToDevice,
                    stream1
                );

                // Start kernels for velocity computations.
                sharedNbody_Kernel<<<
                    grid_dim,
                    block_dim,
                    blockSize * sizeof(float4),  // shared memory for die positions and masses
                    stream0
                >>>(segmentSize, d_posMasses_current0, d_velocities_current0, d_posMasses_other0);
                sharedNbody_Kernel<<<
                    grid_dim,
                    block_dim,
                    blockSize * sizeof(float4),  // shared memory for die positions and masses
                    stream1
                >>>(segmentSize, d_posMasses_current1, d_velocities_current1, d_posMasses_other1);
            }

            // Update positions.
            packedSOAUpdatePosition_Kernel<<<
                grid_dim,
                block_dim,
                0,  // declare shared memory size
                stream0
            >>>(segmentSize,  d_posMasses_current0,  d_velocities_current0);
            packedSOAUpdatePosition_Kernel<<<
                grid_dim,
                block_dim,
                0,  // declare shared memory size
                stream1
            >>>(segmentSize,  d_posMasses_current1,  d_velocities_current1);


            // Copy new positions (and old masses) and velocities from device to host.
            cudaMemcpyAsync(
                h_posMasses + i,
                d_posMasses_current0,
                static_cast<size_t>(segmentSize * sizeof(*d_posMasses_current0)),
                cudaMemcpyDeviceToHost,
                stream0
            );
            cudaMemcpyAsync(
                h_posMasses + i + segmentSize,
                d_posMasses_current1,
                static_cast<size_t>(segmentSize * sizeof(*d_posMasses_current1)), 
                cudaMemcpyDeviceToHost,
                stream1
            );

            // Copy per stream **current** velocities.
            cudaMemcpyAsync(
                h_velocities + i,
                d_velocities_current0,
                static_cast<size_t>(segmentSize * sizeof(*d_velocities_current0)),
                cudaMemcpyDeviceToHost,
                stream0
            );
            cudaMemcpyAsync(
                h_velocities + i + segmentSize,
                d_velocities_current1,
                static_cast<size_t>(segmentSize * sizeof(*d_velocities_current1)),
                cudaMemcpyDeviceToHost,
                stream1
            );
        }

		if (!silent)
		{
			printElement(h_posMasses, h_velocities, 0, it + 1);
			printElement(h_posMasses, h_velocities, 1, it + 1);
			printElement(h_posMasses, h_velocities, 2, it + 1);

			printElement(h_posMasses, h_velocities, 400000, it + 1);
			printElement(h_posMasses, h_velocities, 458751, it + 1);
			printElement(h_posMasses, h_velocities, 500001, it + 1);
			printElement(h_posMasses, h_velocities, 500002, it + 1);
		}
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


	// Free host memory.
	if (!pinnedMemory)
	{
		free(h_posMasses);
        free(h_velocities);
	}
	else
	{
		cudaFreeHost(h_posMasses);
        cudaFreeHost(h_velocities);
	}

	// Free device memory.
	// cudaFree(d_particles);
	cudaFree(d_posMasses_current0);
	cudaFree(d_posMasses_other0);
	cudaFree(d_velocities_current0);

	// Device memory for stream1.
	cudaFree(d_posMasses_current1);
	cudaFree(d_posMasses_other1);
	cudaFree(d_velocities_current1);

	// Print Meassurement Results
	std::cout << "***" << std::endl
			  << "*** Results:" << std::endl
			  << "***    Num Elements: " << numElements << std::endl
			  << "***    Num Iterations: " << numIterations << std::endl
			  << "***    Threads per block: " << blockSize << std::endl
			  << "***    Time to Copy to Device: " << 1e3 * memCpyH2DTimer.getTime()
			  << " ms" << std::endl
			  << "***    Copy Bandwidth: "
			  << 1e-9 * memCpyH2DTimer.getBandwidth(numElements * (sizeof(h_posMasses) + sizeof(h_velocities)))
			  << " GB/s" << std::endl
			  << "***    Time to Copy from Device: " << 1e3 * memCpyD2HTimer.getTime()
			  << " ms" << std::endl
			  << "***    Copy Bandwidth: "
			  << 1e-9 * memCpyD2HTimer.getBandwidth(numElements * (sizeof(h_posMasses) + sizeof(h_velocities)))
			  << " GB/s" << std::endl
			  << "***    Time for n-Body Computation: " << 1e3 * kernelTimer.getTime()
			  << " ms" << std::endl
			  << "***" << std::endl;

	return 0;
}

void printHelp(char *argv)
{
	std::cout << "Help:" << std::endl
			  << "  Usage: " << std::endl
			  << "  " << argv << " [-p] [-s <num-elements>] [-t <threads_per_block>]"
			  << std::endl
			  << "" << std::endl
			  << "  -p|--pinned-memory" << std::endl
			  << "    Use pinned Memory instead of pageable memory" << std::endl
			  << "" << std::endl
			  << "  -s <num-elements>|--size <num-elements>" << std::endl
			  << "    Number of elements (particles)" << std::endl
			  << "" << std::endl
			  << "  -i <num-iterations>|--num-iterations <num-iterations>" << std::endl
			  << "    Number of iterations" << std::endl
			  << "" << std::endl
			  << "  -t <threads_per_block>|--threads-per-block <threads_per_block>"
			  << std::endl
			  << "    The number of threads per block" << std::endl
			  << "" << std::endl
			  << "  --silent"
			  << std::endl
			  << "    Suppress print output during iterations (useful for benchmarking)" << std::endl
			  << "" << std::endl;
}

//
// Print one element
//
void printElement(
    float4 *posMasses,
    float3 *velocities,
    int elementId,
    int iteration
)
{
	float4 posMass = posMasses[elementId];
	float3 velocity = velocities[elementId];

	std::cout << "***" << std::endl
			  << "*** Printing element " << elementId << " in iteration " << iteration << std::endl
			  << "***" << std::endl
			  << "*** Position: <"
			  << std::setw(11) << std::setprecision(9) << posMass.x << "|"
			  << std::setw(11) << std::setprecision(9) << posMass.y << "|"
			  << std::setw(11) << std::setprecision(9) << posMass.z << "> [m]" << std::endl
			  << "*** Velocity: <"
			  << std::setw(11) << std::setprecision(9) << velocity.x << "|"
			  << std::setw(11) << std::setprecision(9) << velocity.y << "|"
			  << std::setw(11) << std::setprecision(9) << velocity.z << "> [m/s]" << std::endl
			  << "*** Mass: <"
			  << std::setw(11) << std::setprecision(9) << posMass.w << "> [kg]" << std::endl
			  << "***" << std::endl;
}
