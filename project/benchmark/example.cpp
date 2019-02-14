#include <benchmark/benchmark.h>

static void BM_Example(benchmark::State &state) {
  for (auto _ : state) {
    int x = 1 + 1;
  }
}
BENCHMARK(BM_Example);
