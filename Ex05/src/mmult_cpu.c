#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>

#define bool int
#define false 0
#define true -1

long calcTime ( struct timeval* tp1, struct timeval* tp2 )
{
	//calculate time passed measured with gettimeofday
	//returns total usecs
	long usecs;

	usecs = (tp2->tv_sec - tp1->tv_sec) * 1E6 + tp2->tv_usec - tp1->tv_usec;

	return usecs;
}


void printTime ( char *label, struct timeval *tp1, struct timeval *tp2, long bytes, long mflops )
{
	//calculate and print out passed time measured with gettimeofday
	long usecs;

	usecs = calcTime (tp1, tp2);
	if ( bytes != 0 )
		printf ( "%s: %4ld usecs passed (%.2lf MB/s)\n", label, usecs, (double) bytes/usecs );
	else if ( mflops != 0 )
		printf ( "%s: %4ld usecs passed (%ld MFLOP, %.2lf MFLOP/s)\n", label, usecs, mflops, (double) mflops/usecs*1E06 );
	else
		printf ( "%s: %4ld usecs passed\n", label, usecs );
	fflush (stdout);
}

bool MatrixCompare ( float* P, float* Q, long  matWidth)
{
	long i;

	for ( i = 0; i < matWidth * matWidth; i++ ) {
		//if ( P[i] != Q[i] )
		// Holger 09.04.2014 floating point calculations might have small errors depending on the operation order
		if ( fabs ( ( P[i]-Q[i] ) / ( P[i]+Q[i] ) ) > 1E-05 )
			return false;
	}
	return true;
}

void MatrixMulOnHost(float* M, float* N, float* P, long matWidth)
{  
	long i, j, k;
	
	for ( i = 0; i < matWidth; ++i) {
		for ( j = 0; j < matWidth; ++j) {
			float sum = 0;
			for ( k = 0; k < matWidth; ++k) {
				float a = M[i * matWidth + k];
				float b = N[k * matWidth + j];
				//printf ("P[%ld][%ld] += M[%ld][%ld] * N[%ld][%ld]\n", j, i, k, i, j, k );
				sum += a * b;
			}
			P[i * matWidth + j] = sum;
		}
	}
}

long findMin ( long a, long b )
{
	if ( a <= b )
		return a;
	else
		return b;
}

void MatrixMulOnHostBlocked(float* M, float* N, float* P, long matWidth, long blockSize)
{
	long ii, jj, kk, i, j, k;
	float temp;

	//printf ("matWidth = %ld, blockSize = %ld\n", matWidth, blockSize );
	for (ii = 0; ii < matWidth; ii += blockSize) {
		for (jj = 0; jj < matWidth; jj += blockSize) {
			for (kk = 0; kk < matWidth; kk += blockSize) {
				for (i = ii; i < findMin(ii+blockSize, matWidth); i++) {
					for (j = jj; j < findMin(jj+blockSize, matWidth); j++) {
						temp = 0;
						for (k = kk; k < findMin(kk+blockSize, matWidth); k++) {
							//if ( j == 1 && i == 2 ) {
							//printf ("P[%ld][%ld] += M[%ld][%ld] * N[%ld][%ld]\n", j, i, k, i, j, k );
							//printf ("P[%ld][%ld] += %.2f * %.2f\n", j, i,  M[i * matWidth + k], N[k * matWidth + j] );
							//}
							temp += M[i * matWidth + k] * N[k * matWidth + j];
						}
						P[ i * matWidth + j] += temp;
						//printf ("P[%ld][%ld]=%.2f\n", j, i, P[ i * matWidth + j] );
					}
				}
			}
		}
	}  
}



void printOutMatrix (float *matrix, int width) {
	int i;
	for (i = 0; i < width*width; i++) {
		printf ("%4.2f\t", matrix[i%width + (i/width) * width]);
		if ((i+1) % width == 0) printf ("\n");
		}
	printf ("\n");
}
