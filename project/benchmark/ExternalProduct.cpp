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

static void BM_ExternalProduct_TRLWEs(benchmark::State &state) {
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
BENCHMARK(BM_ExternalProduct_TRLWEs)
    ->Arg(1)
    ->Arg(4)
    ->Arg(16)
    ->Arg(50)
    ->Arg(100)
    ->Arg(150);

static void BM_ExternalProduct_TRGSWs(benchmark::State &state) {
  // Prepare for benchmark
  //   Set parameters
  const int N = 1024;                              // see TFHE
  const int k = 1;                                 // see TFHE
  const int l = 2;                                 // see TFHE
  const int Bgbit = 10;                            // see TFHE
  const double sd = (9e-9) / std::sqrt(2. / M_PI); // see TFHE
  //   Set number of TRGSW
  const int number_trgsw = state.range(0);
  //   Generate key
  BatchedFFT fft_key(N, 2, k);
  TorusInteger *s =
      (TorusInteger *)MemoryManagement::mallocMM(N * k * sizeof(TorusInteger));
  TrlweFunction::genkey(s, N, k);
  TrlweFunction::keyToFFT(s, N, k, &fft_key);
  //   Generate TRGSW ciphers
  std::vector<TrgswCipher *> trgsw_list(number_trgsw);
  for (int i = 0; i < number_trgsw; i++) {
    trgsw_list[i] = new TrgswCipher(N, k, l, Bgbit, sd, sd * sd);
    for (int j = 0; j < trgsw_list[i]->_kpl; j++)
      // For simplicity, trgsw_list[i] = TrgswCipher(msg = 0)
      TrlweFunction::createSample(
          &fft_key, j & 1, trgsw_list[i]->get_trlwe_data(j), trgsw_list[i]->_N,
          trgsw_list[i]->_k, trgsw_list[i]->_sdError);
  }
  //   Generate 1 TRLWE cipher
  TrlweCipher trlwe(N, k, sd, sd * sd);
  //     For simplicity, trlwe = TrlweCipher(msg = 0)
  TrlweFunction::createSample(&fft_key, 1, &trlwe);
  //   Malloc decomposition vector
  TorusInteger *decomp_vec = (TorusInteger *)MemoryManagement::mallocMM(
      N * (k + 1) * l * sizeof(TorusInteger));
  //   Prepare fft for decomposition
  BatchedFFT fft_decomp(N, k + 1, (k + 1) * l);
  //   Wait for all
  fft_key.waitAllOut();
  // Benchmark
  for (auto _ : state) {
    for (int it = 0; it < number_trgsw; it++) {
      // Set input | trgsw
      for (int i = 0; i <= k; i++) {
        for (int j = 0; j < trgsw_list[it]->_kpl; j++)
          fft_decomp.setInp(trgsw_list[it]->get_pol_data(j, i), i, j);
      }
      if (it == 0) {
        // Decomposition
        Decomposition::onlyDecomp(&trlwe, trgsw_list[0], decomp_vec);
        // Set input | trlwe
        for (int i = 0; i < trgsw_list[0]->_kpl; i++)
          fft_decomp.setInp(decomp_vec + N * i, i);
      }
      // Multiplication
      for (int i = 0; i <= k; i++) {
        for (int j = 0; j < trgsw_list[it]->_kpl; j++)
          fft_decomp.setMul(i, j);
      }
      // Get result
      trlwe.clear_trlwe_data();
      for (int i = 0; i <= k; i++)
        fft_decomp.addAllOut(trlwe.get_pol_data(i), i);
    }
    // Wait for all
    fft_decomp.waitAllOut();
  }
  // Clean for benchmark
  //   Free decomposition vector
  MemoryManagement::freeMM(decomp_vec);
  decomp_vec = nullptr;
  //   Free TRGSW ciphers
  for (int i = 0; i < number_trgsw; i++) {
    delete trgsw_list[i];
    trgsw_list[i] = nullptr;
  }
  //   Free key
  MemoryManagement::freeMM(s);
  s = nullptr;
}
BENCHMARK(BM_ExternalProduct_TRGSWs)
    ->Arg(1)
    ->Arg(4)
    ->Arg(16)
    ->Arg(50)
    ->Arg(100)
    ->Arg(150);
#endif
