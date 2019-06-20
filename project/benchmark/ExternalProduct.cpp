#include <benchmark/benchmark.h>

#include "thesis/batched_fft.h"
#include "thesis/decomposition.h"
#include "thesis/memory_management.h"
#include "thesis/trgsw_cipher.h"
#include "thesis/trgsw_function.h"
#include "thesis/trlwe_cipher.h"
#include "thesis/trlwe_function.h"

using namespace thesis;

#define M_PI 3.14159265358979323846

#ifdef USING_32BIT

static void BM_ExternalProduct(benchmark::State &state) {
  // Prepare for benchmark
  //   Set parameters
  const int N = 1024;                              // see TFHE
  const int k = 1;                                 // see TFHE
  const int l = 2;                                 // see TFHE
  const int Bgbit = 10;                            // see TFHE
  const double sd = (9e-9) / std::sqrt(2. / M_PI); // see TFHE
  //   Set number of TRLWE
  const int number_trlwe = state.range(0);
  //   Generate key
  BatchedFFT fft_key(N, 2, k);
  TorusInteger *s =
      (TorusInteger *)MemoryManagement::mallocMM(N * k * sizeof(TorusInteger));
  TrlweFunction::genkey(s, N, k);
  TrlweFunction::keyToFFT(s, N, k, &fft_key);
  //   Generate 1 TRGSW cipher
  TrgswCipher trgsw(N, k, l, Bgbit, sd, sd * sd);
  for (int i = 0; i < trgsw._kpl; i++)
    // For simplicity, trgsw = TrgswCipher(msg = 0)
    TrlweFunction::createSample(&fft_key, i & 1, trgsw.get_trlwe_data(i),
                                trgsw._N, trgsw._k, trgsw._sdError);
  //   Generate TRLWE ciphers
  std::vector<TrlweCipher *> trlwe_list(number_trlwe);
  for (int i = 0; i < number_trlwe; i++) {
    trlwe_list[i] = new TrlweCipher(N, k, sd, sd * sd);
    // For simplicity, trlwe_list[i] = TrlweCipher(msg = 0)
    TrlweFunction::createSample(&fft_key, i & 1, trlwe_list[i]);
  }
  //   Malloc decomposition vector
  TorusInteger *decomp_vec = (TorusInteger *)MemoryManagement::mallocMM(
      N * (k + 1) * l * sizeof(TorusInteger));
  //   Prepare fft for decomposition
  BatchedFFT fft_decomp(N, k + 1, (k + 1) * l);
  //   Wait for all
  fft_key.waitAllOut();
  // Benchmark
  for (auto _ : state) {
    // Set input | trgsw
    for (int i = 0; i <= k; i++) {
      for (int j = 0; j < trgsw._kpl; j++)
        fft_decomp.setInp(trgsw.get_pol_data(j, i), i, j);
    }
    for (int it = 0; it < number_trlwe; it++) {
      // Decomposition
      Decomposition::onlyDecomp(trlwe_list[it], &trgsw, decomp_vec);
      // Multiplication
      for (int i = 0; i < trgsw._kpl; i++)
        fft_decomp.setInp(decomp_vec + N * i, i);
      for (int i = 0; i <= k; i++) {
        for (int j = 0; j < trgsw._kpl; j++)
          fft_decomp.setMul(i, j);
      }
      // Get result
      trlwe_list[it]->clear_trlwe_data();
      for (int i = 0; i <= k; i++)
        fft_decomp.addAllOut(trlwe_list[it]->get_pol_data(i), i);
    }
    // Wait for all
    fft_decomp.waitAllOut();
  }
  // Clean for benchmark
  //   Free decomposition vector
  MemoryManagement::freeMM(decomp_vec);
  decomp_vec = nullptr;
  //   Free TRLWE ciphers
  for (int i = 0; i < number_trlwe; i++) {
    delete trlwe_list[i];
    trlwe_list[i] = nullptr;
  }
  //   Free key
  MemoryManagement::freeMM(s);
  s = nullptr;
}
BENCHMARK(BM_ExternalProduct)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(50)
    ->Arg(100)
    ->Arg(150)
    ->Arg(200);

#endif
