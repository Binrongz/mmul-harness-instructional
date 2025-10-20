#ifdef LIKWID_PERFMON
#include <likwid.h>
#endif

#include <string.h>  // for memcpy

const char* dgemm_desc = "Blocked dgemm with copy optimization and OpenMP.";

/*
 * Function: square_dgemm_blocked
 * Input:
 *   - n: matrix dimension (n x n)
 *   - block_size: size of cache-friendly blocks
 *   - A, B: input matrices (row-major, size n*n)
 * Output:
 *   - C: result matrix C = C + A*B (modified in place)
 */
void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C) 
{
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_START("MMUL_Region");
#endif

    // Use OpenMP parallel region with thread-local storage
    #pragma omp parallel
    {
        // Allocate thread-local block buffers
        double* blockA = new double[block_size * block_size];
        double* blockB = new double[block_size * block_size];
        double* blockC = new double[block_size * block_size];
        
        // Parallelize the outer loop over blocks
        #pragma omp for
        for (int i0 = 0; i0 < n; i0 += block_size) {
            for (int j0 = 0; j0 < n; j0 += block_size) {
                
                // Calculate actual block dimensions (handle edge cases)
                int block_rows = (i0 + block_size > n) ? n - i0 : block_size;
                int block_cols = (j0 + block_size > n) ? n - j0 : block_size;
                
                // Copy C block to buffer using memcpy
                for (int i = 0; i < block_rows; i++) {
                    memcpy(&blockC[i * block_size], 
                           &C[(i0 + i) * n + j0], 
                           block_cols * sizeof(double));
                }
                
                // Loop over k blocks
                for (int k0 = 0; k0 < n; k0 += block_size) {
                    int block_k = (k0 + block_size > n) ? n - k0 : block_size;
                    
                    // Copy A block to buffer using memcpy
                    for (int i = 0; i < block_rows; i++) {
                        memcpy(&blockA[i * block_size], 
                               &A[(i0 + i) * n + k0], 
                               block_k * sizeof(double));
                    }
                    
                    // Copy B block to buffer using memcpy
                    for (int k = 0; k < block_k; k++) {
                        memcpy(&blockB[k * block_size], 
                               &B[(k0 + k) * n + j0], 
                               block_cols * sizeof(double));
                    }
                    
                    // Perform matrix multiplication on sub-blocks
                    for (int ii = 0; ii < block_rows; ii++) {
                        for (int jj = 0; jj < block_cols; jj++) {
                            for (int kk = 0; kk < block_k; kk++) {
                                blockC[ii * block_size + jj] += 
                                    blockA[ii * block_size + kk] * blockB[kk * block_size + jj];
                            }
                        }
                    }
                }
                
                // Write the result back to original matrix C using memcpy
                for (int i = 0; i < block_rows; i++) {
                    memcpy(&C[(i0 + i) * n + j0], 
                           &blockC[i * block_size], 
                           block_cols * sizeof(double));
                }
            }
        }
        
        // Free thread-local buffers
        delete[] blockA;
        delete[] blockB;
        delete[] blockC;
    }

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_STOP("MMUL_Region");
#endif
}