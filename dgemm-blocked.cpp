const char* dgemm_desc = "Blocked dgemm.";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in row-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C) 
{
   // Allocating block buffers
   double* blockA = new double[block_size * block_size];
   double* blockB = new double[block_size * block_size];
   double* blockC = new double[block_size * block_size];
    
   // Outer loop: Iterates over the block
   for (int i0 = 0; i0 < n; i0 += block_size) {
      for (int j0 = 0; j0 < n; j0 += block_size) {

         int block_rows = (i0 + block_size > n) ? n - i0 : block_size;
         int block_cols = (j0 + block_size > n) ? n - j0 : block_size;
            
         // Copy C blocks to the buffer
         for (int i = 0; i < block_rows; i++) {
            for (int j = 0; j < block_cols; j++) {
               blockC[i * block_size + j] = C[(i0 + i) * n + (j0 + j)];
            }
         }
            
         for (int k0 = 0; k0 < n; k0 += block_size) {
            int block_k = (k0 + block_size > n) ? n - k0 : block_size;
                
            // Copy A blocks to the buffer
            for (int i = 0; i < block_size; i++) {
               for (int k = 0; k < block_size; k++) {
                  blockA[i * block_size + k] = A[(i0 + i) * n + (k0 + k)];
               }
            }
                
            // Copy B blocks to the buffer
            for (int k = 0; k < block_size; k++) {
               for (int j = 0; j < block_size; j++) {
                  blockB[k * block_size + j] = B[(k0 + k) * n + (j0 + j)];
               }
            }
                
            // Execute matrix multiplication on sub-blocks of the matrix
            for (int ii = 0; ii < block_size; ii++) {
               for (int jj = 0; jj < block_size; jj++) {
                  for (int kk = 0; kk < block_size; kk++) {
                     blockC[ii * block_size + jj] += blockA[ii * block_size + kk] * blockB[kk * block_size + jj];
                  }
               }
            }
         }
            
         // Write the result back to the original matrix C
         for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
               C[(i0 + i) * n + (j0 + j)] = blockC[i * block_size + j];
            }
         }
      }
   }
    
   // Free buffer
   delete[] blockA;
   delete[] blockB;
   delete[] blockC;
}