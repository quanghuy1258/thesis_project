#include "gtest/gtest.h"

#include "thesis/batched_fft.h"
#include "thesis/declarations.h"
#include "thesis/load_lib.h"
#include "thesis/memory_management.h"
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
}
/*
TEST(Thesis, TrlweExtractAllToTlwe) {
  std::srand(std::time(nullptr));
  thesis::Trlwe trlweObj;
  thesis::Tlwe tlweObj;
  std::vector<double> errors;
  std::vector<bool> expectedPlaintexts;

  trlweObj.clear_s();
  trlweObj.clear_ciphertexts();
  trlweObj.clear_plaintexts();

  trlweObj.generate_s();
  std::vector<thesis::PolynomialBinary> x;
  int numberTests = 100;
  x.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    x[i].resize(trlweObj.get_N());
    for (int j = 0; j < trlweObj.get_N(); j++) {
      x[i][j] = std::rand() & 1;
    }
    trlweObj.addPlaintext(x[i]);
  }
  trlweObj.encryptAll();
  trlweObj.clear_plaintexts();
  trlweObj.tlweExtractAll(tlweObj);
  tlweObj.decryptAll();
  std::cout << "Number of plaintexts: " << tlweObj.get_plaintexts().size()
            << std::endl;
  expectedPlaintexts.resize(numberTests * trlweObj.get_N());
  for (int i = 0; i < numberTests; i++) {
    for (int j = 0; j < trlweObj.get_N(); j++) {
      ASSERT_TRUE(x[i][j] ==
                  tlweObj.get_plaintexts()[i * trlweObj.get_N() + j]);
      expectedPlaintexts[i * trlweObj.get_N() + j] = x[i][j];
    }
  }
  tlweObj.getAllErrorsForDebugging(errors, expectedPlaintexts);
  for (int i = 0; i < numberTests * trlweObj.get_N(); i++) {
    ASSERT_TRUE(errors[i] < 0.25);
  }
}

TEST(Thesis, TrlweExtractToTlwe) {
  std::srand(std::time(nullptr));
  thesis::Trlwe trlweObj;
  thesis::Tlwe tlweObj;
  std::vector<double> errors;
  std::vector<bool> expectedPlaintexts;

  trlweObj.clear_s();
  trlweObj.clear_ciphertexts();
  trlweObj.clear_plaintexts();
  trlweObj.generate_s();

  std::vector<thesis::PolynomialBinary> x;
  int numberTests = 100;
  x.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    x[i].resize(trlweObj.get_N());
    for (int j = 0; j < trlweObj.get_N(); j++) {
      x[i][j] = std::rand() & 1;
    }
    trlweObj.addPlaintext(x[i]);
  }
  trlweObj.encryptAll();
  trlweObj.clear_plaintexts();
  std::vector<int> ps(numberTests), cipherIDs(numberTests);
  for (int i = 0; i < numberTests; i++) {
    ps[i] = std::rand() % trlweObj.get_N();
    cipherIDs[i] = std::rand() % numberTests;
  }
  trlweObj.tlweExtract(tlweObj, ps, cipherIDs);
  tlweObj.decryptAll();
  std::cout << "Number of plaintexts: " << tlweObj.get_plaintexts().size()
            << std::endl;
  expectedPlaintexts.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    ASSERT_TRUE(x[cipherIDs[i]][ps[i]] == tlweObj.get_plaintexts()[i]);
    expectedPlaintexts[i] = x[cipherIDs[i]][ps[i]];
  }
  tlweObj.getAllErrorsForDebugging(errors, expectedPlaintexts);
  for (int i = 0; i < numberTests; i++) {
    ASSERT_TRUE(errors[i] < 0.25);
  }
}
*/
