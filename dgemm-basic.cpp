#ifdef LIKWID_PERFMON
#include <likwid.h>
#endif

const char* dgemm_desc = "Basic implementation with OpenMP, three-loop dgemm.";

/*
 * Function: square_dgemm
 * Input: 
 *   - n: matrix dimension (n x n)
 *   - A: pointer to matrix A (row-major, size n*n)
 *   - B: pointer to matrix B (row-major, size n*n)
 * Output:
 *   - C: pointer to result matrix C = C + A*B (modified in place)
 */
void square_dgemm(int n, double* A, double* B, double* C) 
{
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_START("MMUL_Region");
#endif

    // Parallelize the outer loop
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i*n + j] += A[i*n + k] * B[k*n + j];
            }
        }
    }

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_STOP("MMUL_Region");
#endif
}