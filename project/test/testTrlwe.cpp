#include "gtest/gtest.h"

#include "thesis/batched_fft.h"
#include "thesis/declarations.h"
#include "thesis/load_lib.h"
#include "thesis/memory_management.h"
#include "thesis/profiling_timer.h"
#include "thesis/stream.h"
#include "thesis/thread_management.h"
#include "thesis/trlwe_cipher.h"
#include "thesis/trlwe_function.h"

using namespace thesis;

TEST(Thesis, TrlweEncryptDecrypt) {
  std::srand(std::time(nullptr));
  const int N = 1024;
  const int k = 1;
  const double sd = std::sqrt(2. / CONST_PI) * pow(2., -15);
  const int numberTests = 100;
  const int parallel = ThreadManagement::getNumberThreadsInPool();
  std::vector<void *> streams(parallel);
  for (int i = 0; i < parallel; i++)
    streams[i] = Stream::createS();
  std::vector<TrlweCipher *> ciphers(numberTests);
  BatchedFFT fft(N, parallel, k);
  TorusInteger *s =
      (TorusInteger *)MemoryManagement::mallocMM(N * k * sizeof(TorusInteger));
  TorusInteger *dPlain = (TorusInteger *)MemoryManagement::mallocMM(
      N * sizeof(TorusInteger) * numberTests);
  double *dError =
      (double *)MemoryManagement::mallocMM(N * sizeof(double) * numberTests);
  std::vector<TorusInteger> oriPlain(N * numberTests),
      calPlain(N * numberTests);
  std::vector<double> error(N * numberTests);
  TrlweFunction::genkey(s, N, k);
  TrlweFunction::keyToFFT(s, N, k, &fft);
  for (int i = 0; i < numberTests; i++) {
    ciphers[i] = new TrlweCipher(N, k, sd, sd * sd);
    TrlweFunction::createSample(&fft, i % parallel, ciphers[i]);
    for (int j = 0; j < N; j++)
      oriPlain[N * i + j] = std::rand() & 1;
  }
  MemoryManagement::memcpyMM_h2d(dPlain, oriPlain.data(),
                                 N * sizeof(TorusInteger) * numberTests);
  DECLARE_TIMING(EncDec);
  START_TIMING(EncDec);
  for (int i = 0; i < numberTests; i++)
    TrlweFunction::putPlain(ciphers[i], dPlain + N * i, streams[i % parallel]);
  for (int i = 0; i < numberTests; i++) {
    Stream::synchronizeS(streams[i % parallel]);
    TrlweFunction::getPlain(&fft, i % parallel, ciphers[i], dPlain + N * i);
  }
  for (int i = 0; i < numberTests; i++) {
    fft.waitOut(i % parallel);
    TrlweFunction::roundPlain(dPlain + N * i, dError + N * i, N,
                              streams[i % parallel]);
  }
  for (int i = 0; i < parallel; i++)
    Stream::synchronizeS(streams[i]);
  STOP_TIMING(EncDec);
  MemoryManagement::memcpyMM_d2h(calPlain.data(), dPlain,
                                 N * sizeof(TorusInteger) * numberTests,
                                 streams[0]);
  MemoryManagement::memcpyMM_d2h(error.data(), dError,
                                 N * sizeof(double) * numberTests, streams[1]);
  Stream::synchronizeS(streams[0]);
  Stream::synchronizeS(streams[1]);
  MemoryManagement::freeMM(dError);
  MemoryManagement::freeMM(dPlain);
  MemoryManagement::freeMM(s);
  for (int i = 0; i < numberTests; i++)
    delete ciphers[i];
  for (int i = 0; i < parallel; i++)
    Stream::destroyS(streams[i]);
  double sumError = 0;
  for (int i = 0; i < N * numberTests; i++) {
    ASSERT_TRUE(oriPlain[i] == calPlain[i]);
    sumError += error[i];
  }
  std::cout << "Avg error = " << sumError / (N * numberTests) << std::endl;
  PRINT_TIMING(EncDec);
}

TEST(Thesis, TrlweRotate) {
  std::srand(std::time(nullptr));
  const int deg = std::rand();
  const int N = 1024;
  const int k = 1;
  const double sd = std::sqrt(2. / CONST_PI) * pow(2., -15);
  const int numberTests = 100;
  const int parallel = ThreadManagement::getNumberThreadsInPool();
  std::vector<void *> streams(parallel);
  for (int i = 0; i < parallel; i++)
    streams[i] = Stream::createS();
  std::vector<TrlweCipher *> ciphers(numberTests * 2);
  BatchedFFT fft(N, parallel, k);
  TorusInteger *s =
      (TorusInteger *)MemoryManagement::mallocMM(N * k * sizeof(TorusInteger));
  TorusInteger *dPlain = (TorusInteger *)MemoryManagement::mallocMM(
      N * sizeof(TorusInteger) * numberTests * 2);
  double *dError = (double *)MemoryManagement::mallocMM(N * sizeof(double) *
                                                        numberTests * 2);
  std::vector<TorusInteger> oriPlain(N * numberTests * 2),
      calPlain(N * numberTests * 2);
  std::vector<double> error(N * numberTests * 2);
  TrlweFunction::genkey(s, N, k);
  TrlweFunction::keyToFFT(s, N, k, &fft);
  for (int i = 0; i < numberTests; i++) {
    ciphers[i] = new TrlweCipher(N, k, sd, sd * sd);
    ciphers[i + numberTests] = new TrlweCipher(N, k, sd, sd * sd);
    TrlweFunction::createSample(&fft, i % parallel, ciphers[i]);
    for (int j = 0; j < N; j++)
      oriPlain[N * i + j] = std::rand() & 1;
    int Nx2 = N * 2;
    int temp_deg = (deg % Nx2 + Nx2) % Nx2;
    for (int j = 0; j < N; j++) {
      if (j >= temp_deg)
        oriPlain[N * (i + numberTests) + j] = oriPlain[N * i + j - temp_deg];
      else if (j + N >= temp_deg)
        oriPlain[N * (i + numberTests) + j] =
            oriPlain[N * i + j + N - temp_deg];
      else
        oriPlain[N * (i + numberTests) + j] =
            oriPlain[N * i + j + N * 2 - temp_deg];
    }
  }
  MemoryManagement::memcpyMM_h2d(dPlain, oriPlain.data(),
                                 N * sizeof(TorusInteger) * numberTests);
  DECLARE_TIMING(EncDecRotate);
  START_TIMING(EncDecRotate);
  for (int i = 0; i < numberTests; i++) {
    TrlweFunction::putPlain(ciphers[i], dPlain + N * i, streams[i % parallel]);
    TrlweFunction::rotate(ciphers[i + numberTests], ciphers[i], deg,
                          streams[i % parallel]);
  }
  for (int i = 0; i < numberTests * 2; i++) {
    Stream::synchronizeS(streams[i % parallel]);
    TrlweFunction::getPlain(&fft, i % parallel, ciphers[i], dPlain + N * i);
  }
  for (int i = 0; i < numberTests * 2; i++) {
    fft.waitOut(i % parallel);
    TrlweFunction::roundPlain(dPlain + N * i, dError + N * i, N,
                              streams[i % parallel]);
  }
  for (int i = 0; i < parallel; i++)
    Stream::synchronizeS(streams[i]);
  STOP_TIMING(EncDecRotate);
  MemoryManagement::memcpyMM_d2h(calPlain.data(), dPlain,
                                 N * sizeof(TorusInteger) * numberTests * 2,
                                 streams[0]);
  MemoryManagement::memcpyMM_d2h(
      error.data(), dError, N * sizeof(double) * numberTests * 2, streams[1]);
  Stream::synchronizeS(streams[0]);
  Stream::synchronizeS(streams[1]);
  MemoryManagement::freeMM(dError);
  MemoryManagement::freeMM(dPlain);
  MemoryManagement::freeMM(s);
  for (int i = 0; i < numberTests * 2; i++)
    delete ciphers[i];
  for (int i = 0; i < parallel; i++)
    Stream::destroyS(streams[i]);
  double sumError = 0;
  for (int i = 0; i < N * numberTests * 2; i++) {
    ASSERT_TRUE(oriPlain[i] == calPlain[i]);
    sumError += error[i];
  }
  std::cout << "Avg error = " << sumError / (N * numberTests * 2) << std::endl;
  PRINT_TIMING(EncDecRotate);
}
