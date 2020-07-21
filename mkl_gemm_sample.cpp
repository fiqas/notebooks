#include<mkl.h>
#include<iostream>
#include <random>
#include <iterator>

float get_random() {
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
    return dis(e);
}

int main() {

  std::cout << "hoho" << std::endl;
  
  int m, n, k;
  m = 1024;
  k = 512;
  n = 1024;

  float* A = new float[m * k];
  float* B = new float[k * n];
  float* C = new float[m * n];

  float alpha = 1.0;
  float beta = 1.0;

  for (int i = 0; i < m * k; i++) {
    A[i] = get_random();
  }
  
  printf ("Top left corner of matrix A: \n");
  for (int i = 0; i < std::min(m, 6); i++) {
    for (int j = 0; j < std::min(k, 6); j++) {
      std::cout << A[j + i * k] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  for (int i = 0; i < k * n; i++)
    B[i] = get_random();

  printf ("Top left corner of matrix B: \n");
  for (int i = 0; i < std::min(k, 6); i++) {
    for (int j = 0; j < std::min(n, 6); j++) {
      std::cout << B[j + i * n] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  for (int i = 0; i < m * n; i++)
    C[i] = 0;
  
  printf ("Top left corner of matrix C before multiply: \n");
  for (int i = 0; i < std::min(m, 6); i++) {
    for (int j = 0; j < std::min(n, 6); j++) {
      std::cout << C[j + i * n] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Size of A is " << m * k << " " << *(&A + 1) - A << std::endl;
  std::cout << "Size of B is " << k * n << " " << *(&B + 1) - B << std::endl;
  std::cout << "Size of C is " << m * n << " " << *(&C + 1) - C << std::endl;

  auto s_initial = dsecnd();
  int LOOP_COUNT = 20;
  for (int r = 0; r < LOOP_COUNT; r++) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
	      m, n, k, alpha, A, k, B, n, beta, C, n);
  }
  auto s_elapsed = (dsecnd() - s_initial) / LOOP_COUNT;
  printf ("\n Computations completed.\n\n");
  printf ("== at %.5f milliseconds == \n\n", (s_elapsed * 1000));
  
  printf ("Top left corner of matrix C after multiply: \n");
  for (int i = 0; i < std::min(m, 6); i++) {
    for (int j = 0; j < std::min(n, 6); j++) {
      std::cout << C[j + i * n] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  
  std::cout << "Deallocating matrices..." << std::endl;
  delete[] A;
  delete[] B;
  delete[] C;
  return 0;
}
