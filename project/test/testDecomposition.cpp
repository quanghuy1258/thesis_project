#include "gtest/gtest.h"

#include "thesis/batched_fft.h"
#include "thesis/declarations.h"
#include "thesis/decomposition.h"
#include "thesis/load_lib.h"
#include "thesis/memory_management.h"
#include "thesis/stream.h"
#include "thesis/thread_management.h"
#include "thesis/trgsw_cipher.h"
#include "thesis/trgsw_function.h"
#include "thesis/trlwe_cipher.h"
#include "thesis/trlwe_function.h"

using namespace thesis;

TEST(Thesis, Decomposition) {
  std::srand(std::time(nullptr));
  const int N = 1024;
  const int k = 1;
  const int l = 3;
  const int Bgbit = 10;
  const double sd = std::sqrt(2. / CONST_PI) * pow(2., -30);
  const int numberTests = 100;
  const int parallel = ThreadManagement::getNumberThreadsInPool();
  BatchedFFT fft(N, parallel, k);
  TorusInteger *s =
      (TorusInteger *)MemoryManagement::mallocMM(N * k * sizeof(TorusInteger));
  // >>> Generate key
  TrlweFunction::genkey(s, N, k);
  TrlweFunction::keyToFFT(s, N, k, &fft);
  // <<<
  TorusInteger *dPlain = (TorusInteger *)MemoryManagement::mallocMM(
      N * sizeof(TorusInteger) * numberTests);
  double *dError =
      (double *)MemoryManagement::mallocMM(N * sizeof(double) * numberTests);
  std::vector<TorusInteger> oriPlain(N * numberTests),
      calPlain(N * numberTests);
  std::vector<double> error(N * numberTests);
  std::vector<int> mulArg(numberTests);
  std::vector<void *> streams(parallel);
  for (int i = 0; i < parallel; i++)
    streams[i] = Stream::createS();
  std::vector<TrlweCipher *> ciphers(numberTests);
  // >>> Prepare ciphertexts
  for (int i = 0; i < numberTests; i++) {
    ciphers[i] = new TrlweCipher(N, k, sd, sd * sd);
    TrlweFunction::createSample(&fft, i % parallel, ciphers[i]);
    for (int j = 0; j < N; j++)
      oriPlain[N * i + j] = std::rand() & 1;
    mulArg[i] = std::rand() & 3;
  }
  MemoryManagement::memcpyMM_h2d(dPlain, oriPlain.data(),
                                 N * sizeof(TorusInteger) * numberTests);
  for (int i = 0; i < numberTests; i++)
    TrlweFunction::putPlain(ciphers[i], dPlain + N * i, streams[i % parallel]);
  // <<<
  // >>> Prepare multiplication
  BatchedFFT decomp(N, 4 * (k + 1), (k + 1) * l);
  TrgswCipher *mul[4];
  for (int i = 0; i < 4; i++) {
    mul[i] = new TrgswCipher(N, k, l, Bgbit, sd, sd * sd);
    for (int j = 0; j < (k + 1) * l; j++)
      TrlweFunction::createSample(&fft, ((k + 1) * l * i + j) % parallel,
                                  mul[i]->get_trlwe_data(j), mul[i]->_N,
                                  mul[i]->_k, mul[i]->_sdError);
  }
  fft.waitAllOut();
  for (int i = 0; i < 4; i++)
    TrgswFunction::addMuGadget(i, mul[i], streams[i % parallel]);
  for (int a = 0; a < 4; a++) {
    Stream::synchronizeS(streams[a % parallel]);
    for (int b = 0; b <= k; b++) {
      for (int c = 0; c < (k + 1) * l; c++)
        decomp.setInp(mul[a]->get_pol_data(c, b), a * (k + 1) + b, c);
    }
  }
  // <<<
  // >>> Multiplication
  TorusInteger *dDecomp = (TorusInteger *)MemoryManagement::mallocMM(
      N * (k + 1) * l * sizeof(TorusInteger));
  for (int a = 0; a < numberTests; a++) {
    Stream::synchronizeS(streams[a % parallel]);
    Decomposition::onlyDecomp(ciphers[a], mul[mulArg[a]], dDecomp);
    for (int b = 0; b < (k + 1) * l; b++)
      decomp.setInp(dDecomp + N * b, b);
    for (int b = 0; b <= k; b++) {
      for (int c = 0; c < (k + 1) * l; c++)
        decomp.setMul(mulArg[a] * (k + 1) + b, c);
    }
    ciphers[a]->clear_trlwe_data();
    for (int b = 0; b <= k; b++)
      decomp.addAllOut(ciphers[a]->get_pol_data(b), mulArg[a] * (k + 1) + b);
  }
  decomp.waitAllOut();
  // <<<
  // >>> Decrypt
  for (int i = 0; i < numberTests; i++) {
    Stream::synchronizeS(streams[i % parallel]);
    TrlweFunction::getPlain(&fft, i % parallel, ciphers[i], dPlain + N * i);
  }
  for (int i = 0; i < numberTests; i++) {
    fft.waitOut(i % parallel);
    TrlweFunction::roundPlain(dPlain + N * i, dError + N * i, N,
                              streams[i % parallel]);
  }
  // <<<
  for (int i = 0; i < parallel; i++)
    Stream::synchronizeS(streams[i]);
  // >>> Move calculated plaintexts and error from device to host
  MemoryManagement::memcpyMM_d2h(calPlain.data(), dPlain,
                                 N * sizeof(TorusInteger) * numberTests,
                                 streams[0]);
  MemoryManagement::memcpyMM_d2h(error.data(), dError,
                                 N * sizeof(double) * numberTests, streams[1]);
  Stream::synchronizeS(streams[0]);
  Stream::synchronizeS(streams[1]);
  // <<<
  MemoryManagement::freeMM(dDecomp);
  for (int i = 0; i < 4; i++)
    delete mul[i];
  for (int i = 0; i < numberTests; i++)
    delete ciphers[i];
  for (int i = 0; i < parallel; i++)
    Stream::destroyS(streams[i]);
  MemoryManagement::freeMM(dError);
  MemoryManagement::freeMM(dPlain);
  MemoryManagement::freeMM(s);
  double sumError = 0;
  for (int i = 0; i < N * numberTests; i++) {
    ASSERT_TRUE(oriPlain[i] * (mulArg[i / N] & 1) == calPlain[i]);
    sumError += error[i];
  }
  std::cout << "Avg error = " << sumError / (N * numberTests) << std::endl;
}

TEST(Thesis, DecompositionForBlindRotate) {
  std::srand(std::time(nullptr));
  const int N = 1024;
  const int k = 1;
  const int l = 3;
  const int Bgbit = 10;
  const double sd = std::sqrt(2. / CONST_PI) * pow(2., -30);
  const int numberTests = 100;
  const int parallel = ThreadManagement::getNumberThreadsInPool();
  BatchedFFT fft(N, parallel, k);
  TorusInteger *s =
      (TorusInteger *)MemoryManagement::mallocMM(N * k * sizeof(TorusInteger));
  // >>> Generate key
  TrlweFunction::genkey(s, N, k);
  TrlweFunction::keyToFFT(s, N, k, &fft);
  // <<<
  TorusInteger *dPlain = (TorusInteger *)MemoryManagement::mallocMM(
      N * sizeof(TorusInteger) * numberTests);
  double *dError =
      (double *)MemoryManagement::mallocMM(N * sizeof(double) * numberTests);
  std::vector<TorusInteger> oriPlain(N * numberTests),
      calPlain(N * numberTests);
  std::vector<double> error(N * numberTests);
  std::vector<int> mulArg(numberTests);
  std::vector<int> degArg(numberTests);
  std::vector<void *> streams(parallel);
  for (int i = 0; i < parallel; i++)
    streams[i] = Stream::createS();
  std::vector<TrlweCipher *> ciphers(numberTests);
  // >>> Prepare ciphertexts
  for (int i = 0; i < numberTests; i++) {
    ciphers[i] = new TrlweCipher(N, k, sd, sd * sd);
    TrlweFunction::createSample(&fft, i % parallel, ciphers[i]);
    for (int j = 0; j < N; j++)
      oriPlain[N * i + j] = std::rand() & 1;
    mulArg[i] = std::rand() & 3;
    degArg[i] = std::abs(std::rand()) % (2 * N);
  }
  MemoryManagement::memcpyMM_h2d(dPlain, oriPlain.data(),
                                 N * sizeof(TorusInteger) * numberTests);
  for (int i = 0; i < numberTests; i++)
    TrlweFunction::putPlain(ciphers[i], dPlain + N * i, streams[i % parallel]);
  // <<<
  // >>> Prepare multiplication
  BatchedFFT decomp(N, 4 * (k + 1), (k + 1) * l);
  TrgswCipher *mul[4];
  for (int i = 0; i < 4; i++) {
    mul[i] = new TrgswCipher(N, k, l, Bgbit, sd, sd * sd);
    for (int j = 0; j < (k + 1) * l; j++)
      TrlweFunction::createSample(&fft, ((k + 1) * l * i + j) % parallel,
                                  mul[i]->get_trlwe_data(j), mul[i]->_N,
                                  mul[i]->_k, mul[i]->_sdError);
  }
  fft.waitAllOut();
  for (int i = 0; i < 4; i++)
    TrgswFunction::addMuGadget(i, mul[i], streams[i % parallel]);
  for (int a = 0; a < 4; a++) {
    Stream::synchronizeS(streams[a % parallel]);
    for (int b = 0; b <= k; b++) {
      for (int c = 0; c < (k + 1) * l; c++)
        decomp.setInp(mul[a]->get_pol_data(c, b), a * (k + 1) + b, c);
    }
  }
  // <<<
  // >>> Multiplication + BlindRotate
  TorusInteger *dDecomp = (TorusInteger *)MemoryManagement::mallocMM(
      N * (k + 1) * l * sizeof(TorusInteger));
  for (int a = 0; a < numberTests; a++) {
    Stream::synchronizeS(streams[a % parallel]);
    Decomposition::forBlindRotate(ciphers[a], mul[mulArg[a]], degArg[a],
                                  dDecomp);
    for (int b = 0; b < (k + 1) * l; b++)
      decomp.setInp(dDecomp + N * b, b);
    for (int b = 0; b <= k; b++) {
      for (int c = 0; c < (k + 1) * l; c++)
        decomp.setMul(mulArg[a] * (k + 1) + b, c);
    }
    for (int b = 0; b <= k; b++)
      decomp.addAllOut(ciphers[a]->get_pol_data(b), mulArg[a] * (k + 1) + b);
  }
  decomp.waitAllOut();
  // <<<
  // >>> Decrypt
  for (int i = 0; i < numberTests; i++) {
    Stream::synchronizeS(streams[i % parallel]);
    TrlweFunction::getPlain(&fft, i % parallel, ciphers[i], dPlain + N * i);
  }
  for (int i = 0; i < numberTests; i++) {
    fft.waitOut(i % parallel);
    TrlweFunction::roundPlain(dPlain + N * i, dError + N * i, N,
                              streams[i % parallel]);
  }
  // <<<
  for (int i = 0; i < parallel; i++)
    Stream::synchronizeS(streams[i]);
  // >>> Move calculated plaintexts and error from device to host
  MemoryManagement::memcpyMM_d2h(calPlain.data(), dPlain,
                                 N * sizeof(TorusInteger) * numberTests,
                                 streams[0]);
  MemoryManagement::memcpyMM_d2h(error.data(), dError,
                                 N * sizeof(double) * numberTests, streams[1]);
  Stream::synchronizeS(streams[0]);
  Stream::synchronizeS(streams[1]);
  // <<<
  MemoryManagement::freeMM(dDecomp);
  for (int i = 0; i < 4; i++)
    delete mul[i];
  for (int i = 0; i < numberTests; i++)
    delete ciphers[i];
  for (int i = 0; i < parallel; i++)
    Stream::destroyS(streams[i]);
  MemoryManagement::freeMM(dError);
  MemoryManagement::freeMM(dPlain);
  MemoryManagement::freeMM(s);
  double sumError = 0;
  for (int i = 0; i < numberTests; i++) {
    for (int j = 0; j < N; j++) {
      int trueVal, falseVal;
      if (j >= degArg[i])
        trueVal = oriPlain[N * i + j - degArg[i]];
      else if (j + N >= degArg[i])
        trueVal = oriPlain[N * i + j + N - degArg[i]];
      else
        trueVal = oriPlain[N * i + j + N * 2 - degArg[i]];
      falseVal = oriPlain[N * i + j];
      ASSERT_TRUE(((mulArg[i] & 1) ? trueVal : falseVal) ==
                  calPlain[N * i + j]);
      sumError += error[N * i + j];
    }
  }
  std::cout << "Avg error = " << sumError / (N * numberTests) << std::endl;
}
