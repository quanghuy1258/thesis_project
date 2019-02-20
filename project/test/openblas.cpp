#include <memory>

#include "gtest/gtest.h"
#include <cblas.h>

TEST(OpenBLAS, ExampleTest) {
  std::srand(std::time(0));
  int M = std::rand() % 20 + 1;
  int K = std::rand() % 20 + 1;
  int N = std::rand() % 20 + 1;
  std::unique_ptr<double[]> A(new double[M * K]);
  std::unique_ptr<double[]> B(new double[K * N]);
  std::unique_ptr<double[]> C(new double[M * N]);
  int count = 1;
  for (int i = 0; i < M * K; i++)
    A[i] = 1.0 / (count++);
  for (int i = 0; i < K * N; i++)
    B[i] = 1.0 / (count++);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A.get(),
              M, B.get(), K, 0.0, C.get(), M);
  for (int i = 0; i < M * N; i++) {
    int r = i % M;
    int c = i / M;
    double val = 0.0;
    for (int j = 0; j < K; j++) {
      val += A[j * M + r] * B[c * K + j];
    }
    EXPECT_TRUE(std::abs(C[i] - val) < 1e-10);
  }
}

