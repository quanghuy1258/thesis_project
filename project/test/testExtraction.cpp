#include "gtest/gtest.h"

#include "thesis/batched_fft.h"
#include "thesis/declarations.h"
#include "thesis/extraction.h"
#include "thesis/load_lib.h"
#include "thesis/memory_management.h"
#include "thesis/profiling_timer.h"
#include "thesis/stream.h"
#include "thesis/thread_management.h"
#include "thesis/tlwe_cipher.h"
#include "thesis/tlwe_function.h"
#include "thesis/trlwe_cipher.h"
#include "thesis/trlwe_function.h"

using namespace thesis;

TEST(Thesis, Extraction) {
  std::srand(std::time(nullptr));
  const int N = 1024;
  const int k = 1;
  const double sd = std::sqrt(2. / CONST_PI) * pow(2., -15);
  const int numberTests = 100;
  const int parallel = ThreadManagement::getNumberThreadsInPool();
  BatchedFFT fft(N, parallel, k);
  TorusInteger *s =
      (TorusInteger *)MemoryManagement::mallocMM(N * k * sizeof(TorusInteger));
  TrlweFunction::genkey(s, N, k);
  TrlweFunction::keyToFFT(s, N, k, &fft);
  std::vector<void *> streams(parallel);
  for (int i = 0; i < parallel; i++)
    streams[i] = Stream::createS();
  TorusInteger *dPlain = (TorusInteger *)MemoryManagement::mallocMM(
      N * sizeof(TorusInteger) * numberTests);
  double *dError =
      (double *)MemoryManagement::mallocMM(sizeof(double) * numberTests);
  std::vector<TorusInteger> oriPlain(N * numberTests), calPlain(numberTests);
  std::vector<TrlweCipher *> oriCiphers(numberTests);
  std::vector<TlweCipher *> calCiphers(numberTests);
  std::vector<int> randDeg(numberTests);
  std::vector<double> errors(numberTests);
  for (int i = 0; i < numberTests; i++) {
    oriCiphers[i] = new TrlweCipher(N, k, sd, sd * sd);
    calCiphers[i] = new TlweCipher(N * k, sd, sd * sd);
    TrlweFunction::createSample(&fft, i % parallel, oriCiphers[i]);
    for (int j = 0; j < N; j++)
      oriPlain[N * i + j] = std::rand() & 1;
    randDeg[i] = std::abs(std::rand()) % N;
  }
  MemoryManagement::memcpyMM_h2d(dPlain, oriPlain.data(),
                                 N * sizeof(TorusInteger) * numberTests);
  fft.waitAllOut();
  DECLARE_TIMING(Extraction);
  START_TIMING(Extraction);
  for (int i = 0; i < numberTests; i++) {
    TrlweFunction::putPlain(oriCiphers[i], dPlain + N * i,
                            streams[i % parallel]);
    Extraction::extract(oriCiphers[i], randDeg[i], calCiphers[i],
                        streams[i % parallel]);
    TlweFunction::decrypt(s, calCiphers[i], dPlain + i, dError + i,
                          streams[i % parallel]);
  }
  for (int i = 0; i < parallel; i++)
    Stream::synchronizeS(streams[i]);
  STOP_TIMING(Extraction);
  MemoryManagement::memcpyMM_d2h(
      calPlain.data(), dPlain, numberTests * sizeof(TorusInteger), streams[0]);
  MemoryManagement::memcpyMM_d2h(errors.data(), dError,
                                 numberTests * sizeof(double), streams[1]);
  Stream::synchronizeS(streams[0]);
  Stream::synchronizeS(streams[1]);
  for (int i = 0; i < numberTests; i++) {
    delete calCiphers[i];
    delete oriCiphers[i];
  }
  MemoryManagement::freeMM(dError);
  MemoryManagement::freeMM(dPlain);
  for (int i = 0; i < parallel; i++)
    Stream::destroyS(streams[i]);
  MemoryManagement::freeMM(s);
  double sumError = 0;
  for (int i = 0; i < numberTests; i++) {
    ASSERT_TRUE(oriPlain[N * i + randDeg[i]] == calPlain[i]);
    sumError += errors[i];
  }
  std::cout << "Avg error = " << sumError / numberTests << std::endl;
  PRINT_TIMING(Extraction);
}
