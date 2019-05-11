#include "gtest/gtest.h"

#include "thesis/batched_fft.h"
#include "thesis/declarations.h"
#include "thesis/load_lib.h"
#include "thesis/memory_management.h"
#include "thesis/profiling_timer.h"
#include "thesis/stream.h"
#include "thesis/thread_management.h"
#include "thesis/trgsw_cipher.h"
#include "thesis/trgsw_function.h"
#include "thesis/trlwe_cipher.h"
#include "thesis/trlwe_function.h"

using namespace thesis;

TEST(Thesis, TrgswEncryptDecrypt) {
  std::srand(std::time(nullptr));
  const int N = 1024;
  const int k = 1;
  const int l = 3;
  const int Bgbit = 10;
  const double sd = std::sqrt(2. / CONST_PI) * pow(2., -30);
  const int msgSize = 13;
  const int numberTests = 100;
  const int parallel = ThreadManagement::getNumberThreadsInPool();
  BatchedFFT fft(N, parallel, k);
  TorusInteger *s =
      (TorusInteger *)MemoryManagement::mallocMM(N * k * sizeof(TorusInteger));
  // >>> Generate TRLWE key -> Generate TRGSW key
  TrlweFunction::genkey(s, N, k);
  TrlweFunction::keyToFFT(s, N, k, &fft);
  // <<<
  std::vector<void *> streams(parallel);
  for (int i = 0; i < parallel; i++)
    streams[i] = Stream::createS();
  TorusInteger *dPlain = (TorusInteger *)MemoryManagement::mallocMM(
      N * sizeof(TorusInteger) * numberTests);
  std::vector<TorusInteger> oriPlain(N * numberTests),
      calPlain(N * numberTests);
  std::vector<TrgswCipher *> ciphers(numberTests);
  for (int i = 0; i < numberTests; i++) {
    ciphers[i] = new TrgswCipher(N, k, l, Bgbit, sd, sd * sd);
    for (int j = 0; j < (k + 1) * l; j++) {
      // >>> Create TRLWE samples for each row of TRGSW samples
      TrlweFunction::createSample(&fft, ((k + 1) * l * i + j) % parallel,
                                  ciphers[i]->get_trlwe_data(j), ciphers[i]->_N,
                                  ciphers[i]->_k, ciphers[i]->_sdError);
      // <<<
    }
    // >>> Create random plaintexts
    const int mask = (1 << msgSize) - 1;
    for (int j = 0; j < N; j++)
      oriPlain[N * i + j] = std::rand() & mask;
    // <<<
  }
  // >>> Move plaintexts form host to device
  MemoryManagement::memcpyMM_h2d(dPlain, oriPlain.data(),
                                 N * sizeof(TorusInteger) * numberTests);
  // <<<
  fft.waitAllOut();
  DECLARE_TIMING(EncDec);
  START_TIMING(EncDec);
  // >>> Put plaintexts to TRGSW samples -> TRGSW ciphers
  for (int i = 0; i < numberTests; i++)
    TrgswFunction::addMuGadget(dPlain + N * i, ciphers[i],
                               streams[i % parallel]);
  // <<<
  // >>> Decrypt: Get plaintexts with error
  for (int i = 0; i < numberTests; i++) {
    Stream::synchronizeS(streams[i % parallel]);
    for (int j = 0; j < l; j++)
      TrlweFunction::getPlain(&fft, (l * i + j) % parallel,
                              ciphers[i]->get_trlwe_data(k * l + j),
                              ciphers[i]->_N, ciphers[i]->_k,
                              ciphers[i]->get_pol_data(k * l + j, k));
  }
  // <<<
  // >>> Clean buffer (we will place calculated plaintexts here)
  MemoryManagement::memsetMM(dPlain, 0, N * sizeof(TorusInteger) * numberTests);
  // <<<
  fft.waitAllOut();
  // >>> Decrypt last l rows of each TRGSW ciphers: PartDecrypt
  for (int i = 0; i < numberTests; i++) {
    for (int j = 0; j < l; j++)
      TrgswFunction::partDecrypt(
          ciphers[i], ciphers[i]->get_pol_data(k * l + j, k), j, msgSize,
          dPlain + N * i, streams[i % parallel]);
  }
  // <<<
  // >>> Remove error and round the plaintexts
  for (int i = 0; i < numberTests; i++)
    TrgswFunction::finalDecrypt(dPlain + N * i, N, msgSize,
                                streams[i % parallel]);
  // <<<
  for (int i = 0; i < parallel; i++)
    Stream::synchronizeS(streams[i]);
  // >>> Move plaintexts from device to host
  STOP_TIMING(EncDec);
  MemoryManagement::memcpyMM_d2h(calPlain.data(), dPlain,
                                 N * sizeof(TorusInteger) * numberTests);
  // <<<
  for (int i = 0; i < numberTests; i++)
    delete ciphers[i];
  MemoryManagement::freeMM(dPlain);
  for (int i = 0; i < parallel; i++)
    Stream::destroyS(streams[i]);
  MemoryManagement::freeMM(s);
  for (int i = 0; i < N * numberTests; i++)
    ASSERT_TRUE(oriPlain[i] == calPlain[i]);
  PRINT_TIMING(EncDec);
}
