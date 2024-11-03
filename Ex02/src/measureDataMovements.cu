/*
 *
 * nullKernelAsync.cu
 *
 * Microbenchmark for throughput of asynchronous kernel launch.
 *
 * Build with: nvcc -I ../chLib <options> nullKernelAsync.cu
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions 
 * are met: 
 *
 * 1. Redistributions of source code must retain the above copyright 
 *    notice, this list of conditions and the following disclaimer. 
 * 2. Redistributions in binary form must reproduce the above copyright 
 *    notice, this list of conditions and the following disclaimer in 
 *    the documentation and/or other materials provided with the 
 *    distribution. 
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <stdio.h>

#include "chTimer.h"

__global__
void
NullKernel()
{
}


void
transferTest(int transfer_N,int CIter){
    char *pageable_array = 0;
    char *device_array = 0;
    
    printf( "Current size %d\n", transfer_N );
    //printf( "Current iters %d\n", CIter  / transfer_N );
    pageable_array = (char*)malloc(transfer_N );
    cudaMalloc((void**)&device_array, transfer_N);
    
    printf( "Measuring D2H " ); fflush( stdout );
    
    chTimerTimestamp start, stop;
    chTimerGetTime( &start );
    for ( int i = 0; i < CIter ; i++ ) {

        // Transfer data from device to host
        cudaMemcpy(  pageable_array,device_array,  transfer_N * sizeof(char), cudaMemcpyDeviceToHost );
        
    }
    chTimerGetTime( &stop );
     {
        double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
        double usPerLaunch = microseconds / (float) CIter;

        printf( "%.2f us\n", usPerLaunch );
    }

   
   
   printf( "Measuring H2D " ); fflush( stdout );
   
    chTimerGetTime( &start );
    for ( int i = 0; i < CIter ; i++ ) {
                
        
        // Transfer data from host to device
        cudaMemcpy(   device_array,pageable_array, transfer_N * sizeof(char) , cudaMemcpyHostToDevice );
        
    }
    chTimerGetTime( &stop );    
     {
        double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
        double usPerLaunch = microseconds / (float) CIter;

        printf( "%.2f us\n", usPerLaunch );
    }

    // Free device buffer
    cudaFree( device_array ); 

    free( pageable_array );
};


int
main()
{
    
    
    
    const int cIterations = 1000000;
   
    // initializing errors.
    cudaError_t err = cudaGetLastError();
   
    int kb = 1<<10;
    int mb = 1<<20;
    int gb = 1<<30;
   
    int num_bytes_pageable = kb;
    int num_bytes_pinned =  kb;
    
    char *pageable_array = 0;
 
    char *pinned_array = 0;

    printf( "Measuring malloc... " ); fflush( stdout );
    chTimerTimestamp start, stop;
    chTimerGetTime( &start );
    for ( int i = 0; i < cIterations; i++ ) {
                // pointers to host & device arrays
       
        
        // malloc a host array
        pageable_array = (char*)malloc(num_bytes_pageable );
        
    }
    chTimerGetTime( &stop );
     {
        double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
        double usPerLaunch = microseconds / (float) cIterations;

        printf( "%.2f us\n", usPerLaunch );
    }
    

    err = cudaGetLastError();
    if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    else{
        printf("Allocating pageable memory worked\n");
    }


    printf( "Measuring CudaMallocHost " ); fflush( stdout );
  
    chTimerGetTime( &start );
    for ( int i = 0; i < cIterations; i++ ) {
                // pointers to host & device arrays
       
        
        
        cudaMallocHost((void**)&pinned_array, num_bytes_pinned);
        
    }
    chTimerGetTime( &stop );
     {
        double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
        double usPerLaunch = microseconds / (float) cIterations;

        printf( "%.2f us\n", usPerLaunch );
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    else{
        printf("Allocating pinned memory worked\n");
    }
    
    //Testing different transfer sizes, the bigger the data the less cycles we do.
    transferTest(      kb,      mb);
    transferTest(5  *  kb,200 * kb);
    transferTest(10*   kb,100 * kb);
    transferTest(50*   kb,20 *  kb);
    transferTest(100*  kb,10 *  kb);
    transferTest(500*  kb,2 *   kb);
    transferTest(      mb,      kb);
    transferTest(10*   mb,      100);
    transferTest(100*  mb,      10);
    transferTest(      gb,      1);
    //transferTest(gb,cIterations);


    free ( pageable_array );
    
    cudaFreeHost( pinned_array );  

    cudaDeviceSynchronize();
    // initializing errors.
    err = cudaGetLastError();
    if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    else{
        printf("Freeing memory worked\n");
    }

    return 0;
}


