#include<mkl.h>
#include<iostream>
#include <random>
#include <iterator>
#include <cstdint>

int main() {
  
  int m, n, k, b, bn;
  m = 6;
  k = 6;
  n = 6;
  b = 2;
  bn = 5;

  float* A_dense = new float[m * k] {1.0, 0.0, 6.0, 7.0, 0.0, 0.0, 2.0, 1.0, 8.0, 2.0, 0.0, 0.0, 6.0, 8.0, 1.0, 4.0, 0.0, 0.0, 7.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,};

  float* A = new float[b * b * bn] {1.0, 0.0, 2.0, 1.0, 6.0, 7.0, 8.0, 2.0, 1.0, 4.0, 5.0, 1.0, 4.0, 3.0, 0.0, 0.0, 7.0, 2.0, 0.0, 0.0};
  MKL_INT* A_col = new MKL_INT[bn] {0, 1, 1, 1, 2};
  MKL_INT* A_pointB = new MKL_INT[3] {0, 2, 3};
  MKL_INT* A_pointE = new MKL_INT[3] {2, 3, 5};
  
  printf ("A dense matrix A: \n");
  for (int i = 0; i < std::min(m, 6); i++) {
    for (int j = 0; j < std::min(k, 6); j++) {
      std::cout << A_dense[j + i * k] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  // mkl_sparse_s_create_bsr ( sparse_matrix_t *A , const sparse_index_base_t indexing , const sparse_layout_t block_layout , const MKL_INT rows , const MKL_INT cols , const MKL_INT block_size , MKL_INT *rows_start , MKL_INT *rows_end , MKL_INT *col_indx , float *values )
  
  sparse_matrix_t* A_handle;
  auto sparse_status = mkl_sparse_s_create_bsr(A_handle, 
			  SPARSE_INDEX_BASE_ZERO, 
			  SPARSE_LAYOUT_ROW_MAJOR,
			  m,
			  k,
			  2,
			  A_pointB,
			  A_pointE,
			  A_col,
			  A);

  if (sparse_status != SPARSE_STATUS_SUCCESS) {
    std::cerr << "Created a sparse matrix" << std::endl;
  }
  
  // std::cout << "Deallocating matrices..." << std::endl;
  // delete[] A;
  // delete[] B;
  // delete[] C;
  return 0;
}
