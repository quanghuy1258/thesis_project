#include "gtest/gtest.h"

#include "thesis/declarations.h"
#include "thesis/load_lib.h"
#include "thesis/memory_management.h"
#include "thesis/profiling_timer.h"
#include "thesis/stream.h"
#include "thesis/thread_management.h"
#include "thesis/tlwe_cipher.h"
#include "thesis/tlwe_function.h"

using namespace thesis;

TEST(Thesis, TlweEncryptDecrypt) {
  std::srand(std::time(nullptr));
  const int n = 500;
  const double sd = std::sqrt(2. / CONST_PI) * pow(2., -15);
  const int numberTests = 100;
  const int parallel = ThreadManagement::getNumberThreadsInPool();
  std::vector<void *> streams(parallel);
  std::vector<TlweCipher *> ciphers(numberTests);
  std::vector<TorusInteger> oriPlains(numberTests), calPlains(numberTests);
  std::vector<double> errors(numberTests);
  TorusInteger *s =
      (TorusInteger *)MemoryManagement::mallocMM(n * sizeof(TorusInteger));
  TorusInteger *dCalPlains = (TorusInteger *)MemoryManagement::mallocMM(
      numberTests * sizeof(TorusInteger));
  double *dErrors =
      (double *)MemoryManagement::mallocMM(numberTests * sizeof(double));
  for (int i = 0; i < parallel; i++)
    streams[i] = Stream::createS();
  TlweFunction::genkey(s, n);
  DECLARE_TIMING(EncDec);
  START_TIMING(EncDec);
  for (int i = 0; i < numberTests; i++) {
    ciphers[i] = new TlweCipher(n, sd, sd * sd);
    oriPlains[i] = std::rand() & 1;
    TlweFunction::encrypt(s, oriPlains[i], ciphers[i], streams[i % parallel]);
    TlweFunction::decrypt(s, ciphers[i], dCalPlains + i, dErrors + i,
                          streams[i % parallel]);
  }
  STOP_TIMING(EncDec);
  for (int i = 0; i < parallel; i++)
    Stream::synchronizeS(streams[i]);
  MemoryManagement::memcpyMM_d2h(calPlains.data(), dCalPlains,
                                 numberTests * sizeof(TorusInteger),
                                 streams[0]);
  MemoryManagement::memcpyMM_d2h(errors.data(), dErrors,
                                 numberTests * sizeof(double), streams[1]);
  Stream::synchronizeS(streams[0]);
  Stream::synchronizeS(streams[1]);
  for (int i = 0; i < numberTests; i++)
    delete ciphers[i];
  for (int i = 0; i < parallel; i++)
    Stream::destroyS(streams[i]);
  MemoryManagement::freeMM(dErrors);
  MemoryManagement::freeMM(dCalPlains);
  MemoryManagement::freeMM(s);
  double sumError = 0;
  for (int i = 0; i < numberTests; i++) {
    ASSERT_TRUE(oriPlains[i] == calPlains[i]);
    sumError += errors[i];
  }
  std::cout << "Avg error = " << sumError / numberTests << std::endl;
  PRINT_TIMING(EncDec);
}
