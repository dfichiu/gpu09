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

int
main()
{
    const int cIterations = 1000000;

    int numBlocks = 16384;
    int threadsPerBlock = 1024;

    chTimerTimestamp start, stop;

    printf( "numBlocks,threadsPerBlock,usPerLaunch\n" ); fflush( stdout );

    while ( numBlocks > 0) {
        threadsPerBlock = 1024;
        while ( threadsPerBlock > 1 ) {
            chTimerGetTime( &start );
            for ( int i = 0; i < cIterations; i++ ) {
                NullKernel<<<numBlocks,threadsPerBlock>>>();
                cudaDeviceSynchronize();
            }
            chTimerGetTime( &stop );

            {
                double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
                double usPerLaunch = microseconds / (float) cIterations;

                printf( "%d,%d,%.2f us\n", numBlocks, threadsPerBlock, usPerLaunch );
                fflush( stdout );
            }

            threadsPerBlock = threadsPerBlock / 2;
        }
        
        numBlocks = numBlocks - 1024;
    }
    
    return 0;
}
