#include <iostream>
#include <iomanip>
#include <vector>
#include <chCommandLine.h>
#include <chTimer.hpp>
 

// Function to initialize matrix A with custom values
void initializeMatrixA(std::vector<std::vector<int>>& matrix, int size) {

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = i+j; // Fill the matrix with incremental values
        }
    }
}

// Function to initialize matrix B with custom values
void initializeMatrixB(std::vector<std::vector<int>>& matrix, int size) {

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = i*j; // Fill the matrix with incremental values
        }
    }
}

// Function to add matrix A and matrix B into matrix C
void multMatrices(const std::vector<std::vector<int>>& A,
                 const std::vector<std::vector<int>>& B,
                 std::vector<std::vector<int>>& C, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
					C[i][j] += A[i][k]*B[k][j]; // Element-wise addition of matrices
			}
		}
    }
}

// Function to print a matrix
void printMatrix(const std::vector<std::vector<int>>& matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int main( int argc, char * argv[] ) {
    int size = 3; // Define the size of the square matrices (e.g., 3x3)
    
	ChTimer CPUTimer;
	chCommandLineGet <int> ( &size, "n", argc, argv );
    // Create 3 matrices A, B, and C
    std::vector<std::vector<int>> A(size, std::vector<int>(size));
    std::vector<std::vector<int>> B(size, std::vector<int>(size));
    std::vector<std::vector<int>> C(size, std::vector<int>(size));

    // Initialize matrices A and B
    initializeMatrixA(A, size);
    initializeMatrixB(B, size);
	CPUTimer.start();
    // Add matrices A and B to get matrix C
    multMatrices(A, B, C, size);
	CPUTimer.stop();
    // Print matrices  C if small enough
     
	if (size<6){
 		std::cout << "Matrix A:" << std::endl;
    	printMatrix(A, size);
		std::cout << "Matrix B:" << std::endl;
    	printMatrix(B, size);
		
		std::cout << "Matrix C (A * B):" << std::endl;
    	printMatrix(C, size);
	}
	double Gflops = (double)size*(double)size*2.0*(double)size/(1E09);
 
	 

    std::cout << "N=" << size << ", Time: "<< std::fixed  << std::setw(6)  << CPUTimer.getTime(1)  << "s,     " <<   Gflops/CPUTimer.getTime(1)  << "GFLOP/s" <<std::endl;
	 

    return 0;
}
