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
transferTest(long transfer_N,int CIter){
    char *pageable_array = 0;
    char *device_array1 = 0;
    char *device_array2 = 0;
    char *pinned_array = 0;
    double gb = 1<<30;


    printf( "Current size %d\n", transfer_N );
    //printf( "Current iters %d\n", CIter  / transfer_N );
    pageable_array = (char*)malloc(transfer_N );
    cudaMalloc((void**)&device_array1, transfer_N);
    cudaMalloc((void**)&device_array2, transfer_N);
    cudaMallocHost((void**)&pinned_array, transfer_N);
    printf( "Measuring D2H " ); fflush( stdout );
    
    chTimerTimestamp start, stop ;
    chTimerGetTime( &start );
    for ( int i = 0; i < CIter ; i++ ) {

        // Transfer data from device to host
        cudaMemcpy(  pageable_array,device_array1,  transfer_N * sizeof(char), cudaMemcpyDeviceToHost );
        
    }
    chTimerGetTime( &stop );
    {
        double bandwith = chTimerBandwidth( &start, &stop,( CIter * transfer_N )/ gb);
        printf( "%.2f GB/s \n", bandwith  );
    }

   
   
   printf( "Measuring H2D " ); fflush( stdout );
   
    chTimerGetTime( &start );
    for ( int i = 0; i < CIter ; i++ ) {
                
        
        // Transfer data from host to device
        cudaMemcpy(   device_array1,pageable_array, transfer_N * sizeof(char) , cudaMemcpyHostToDevice );
        
    }
    chTimerGetTime( &stop );    
    {
        double bandwith = chTimerBandwidth( &start, &stop,( CIter * transfer_N )/ gb);
        printf( "%.2f GB/s \n", bandwith  );
    }
    printf( "Measuring D2H_pinned " ); fflush( stdout );
    
   
    chTimerGetTime( &start );
    for ( int i = 0; i < CIter ; i++ ) {

        // Transfer data from device to host
        cudaMemcpy(  pinned_array,device_array1,  transfer_N * sizeof(char), cudaMemcpyDeviceToHost );
        
    }
    chTimerGetTime( &stop );
    {
        double bandwith = chTimerBandwidth( &start, &stop,( CIter * transfer_N )/ gb);
        printf( "%.2f GB/s \n", bandwith  );
    }

   
   
   printf( "Measuring H2D_pinned " ); fflush( stdout );
   
    chTimerGetTime( &start );
    for ( int i = 0; i < CIter ; i++ ) {
                
        
        // Transfer data from host to device
        cudaMemcpy(   device_array1, pinned_array, transfer_N * sizeof(char) , cudaMemcpyHostToDevice );
        
    }
    chTimerGetTime( &stop );    
    {
        double bandwith = chTimerBandwidth( &start, &stop,( CIter * transfer_N )/ gb);
        printf( "%.2f GB/s \n", bandwith  );
    }
    printf( "Measuring D2D " ); fflush( stdout );
   
     chTimerGetTime( &start );
    for ( int i = 0; i < CIter ; i++ ) {
                
       
        // Transfer data from host to device
        cudaMemcpy(   device_array1,device_array2, transfer_N * sizeof(char) , cudaMemcpyDeviceToDevice );
        cudaDeviceSynchronize();
        
    }
    chTimerGetTime( &stop );
    {
        double bandwith = chTimerBandwidth( &start, &stop,( CIter * transfer_N )/ gb);
        printf( "%.2f GB/s \n", bandwith  );
    }

    // Free device buffer
    cudaFree( device_array1 ); 
    cudaFree( device_array2 ); 
    cudaFreeHost( pinned_array );  
    free( pageable_array );
};


int
main()
{
    
    
    
    //const int cIterations = 10000;
   
    // initializing errors.
    cudaError_t err = cudaGetLastError();
   
    long kb = 1<<10;
    long mb = 1<<20;
    long gb = 1<<30;
   
    int num_bytes_pageable = mb;
    //int num_bytes_pinned =  mb;
    
    //char *pageable_array = 0;
 
    //char *pinned_array = 0;

   
    //chTimerTimestamp start, stop;
    
    
    //Testing different transfer sizes, the bigger the data the less cycles we do.
    transferTest(      kb,      mb);
    transferTest(5  *  kb,200 * kb);
    transferTest(10*   kb,100 * kb);
    transferTest(50*   kb,20 *  kb);
    transferTest(100*  kb,10 *  kb);
    transferTest(500*  kb,2 *   kb);
    transferTest(      mb,      kb);
    transferTest(10*   mb,      kb);
    transferTest(100*  mb,      kb);
    transferTest(      gb,      100);
    //transferTest(gb,cIterations);


   

    cudaDeviceSynchronize();
    // initializing errors.
    err = cudaGetLastError();
    if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    else{
        printf("Everything worked\n");
    }

    return 0;
}


