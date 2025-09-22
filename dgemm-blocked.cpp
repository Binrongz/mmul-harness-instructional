const char* dgemm_desc = "Blocked dgemm.";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in row-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C) 
{
   // Iterate over the block coordinates
   for (int i0 = 0; i0 < n; i0 += block_size) {
      for (int j0 = 0; j0 < n; j0 += block_size) {
         for (int k0 = 0; k0 < n; k0 += block_size) {
            // Do a triple loop inside a block
            for (int i = i0; i < i0 + block_size; i++) {
               for (int j = j0; j < j0 + block_size; j++) {
                  for (int k = k0; k < k0 + block_size; k++) {
                     C[i*n + j] += A[i*n + k] * B[k*n + j];
                  }
               }
            }
         }
      }
   }
}
