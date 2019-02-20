#include <cstdlib>
#include <ctime>
#include <memory>

#include <benchmark/benchmark.h>
#include <cblas.h>

static void BM_OpenBLAS_Matmul(benchmark::State &state) {
  std::srand(std::time(0));
  int M = state.range(0);
  int K = state.range(1);
  int N = state.range(2);
  std::unique_ptr<double[]> A(new double[M * K]);
  std::unique_ptr<double[]> B(new double[K * N]);
  std::unique_ptr<double[]> C(new double[M * N]);
  for (auto _ : state) {
    state.PauseTiming();
    for (int i = 0; i < M * K; i++)
      A[i] = 1.0 / std::rand();
    for (int i = 0; i < K * N; i++)
      B[i] = 1.0 / std::rand();
    state.ResumeTiming();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0,
                A.get(), M, B.get(), K, 0.0, C.get(), M);
  }
}
BENCHMARK(BM_OpenBLAS_Matmul)->Args({100, 150, 200})->Args({1000, 1500, 2000});
