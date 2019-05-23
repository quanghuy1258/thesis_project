#include "gtest/gtest.h"

#include "mpc_application.h"
#include "thesis/batched_fft.h"
#include "thesis/memory_management.h"
#include "thesis/profiling_timer.h"
#include "thesis/torus_utility.h"
#include "thesis/trgsw_function.h"
#include "thesis/trlwe_function.h"

using namespace thesis;

const int numParty = 3;
const int N = 1024;
const int m = 6;
const int l = 30;
const double sdFresh = 1e-9;

bool is_file_exist(const char *fileName);
void save_data(const char *fileName, void *buffer, int sz);
void load_data(const char *fileName, void *buffer, int sz);

bool test_genkey(void *priv_key, void *pub_key);
bool genkey();

bool test_pre_expand(void *priv_key, void *pub_key, void *pre_expand);
bool pre_expand();

bool test_encrypt(bool msg, void *priv_key, void *pub_key, void *mainCipher,
                  void *randCipher, void *random);
bool encrypt();

bool expand_partDec();

bool test_operator();

bool reduce();

TEST(Mpc, Full) {
  ASSERT_TRUE(genkey());
  ASSERT_TRUE(pre_expand());
  ASSERT_TRUE(encrypt());
  ASSERT_TRUE(expand_partDec());
  ASSERT_TRUE(test_operator());
  ASSERT_TRUE(reduce());
}

bool is_file_exist(const char *fileName) {
  std::ifstream f(fileName, std::ifstream::binary);
  bool chk = f.good();
  f.close();
  return chk;
}
void save_data(const char *fileName, void *buffer, int sz) {
  std::ofstream f(fileName, std::ifstream::binary);
  f.write((char *)buffer, sz);
  f.close();
}
void load_data(const char *fileName, void *buffer, int sz) {
  std::ifstream f(fileName, std::ifstream::binary);
  f.read((char *)buffer, sz);
  f.close();
}
bool test_genkey(void *priv_key, void *pub_key) {
  // Private key
  TorusInteger *priv =
      (TorusInteger *)MemoryManagement::mallocMM(N * sizeof(TorusInteger));
  MemoryManagement::memcpyMM_h2d(priv, priv_key, N * sizeof(TorusInteger));
  // Public key
  TorusInteger *pub = (TorusInteger *)MemoryManagement::mallocMM(
      m * N * 2 * sizeof(TorusInteger));
  MemoryManagement::memcpyMM_h2d(pub, pub_key,
                                 m * N * 2 * sizeof(TorusInteger));
  // Plaintext
  TorusInteger *plain =
      (TorusInteger *)MemoryManagement::mallocMM(m * N * sizeof(TorusInteger));
  MemoryManagement::memsetMM(plain, 0, m * N * sizeof(TorusInteger));
  // Error
  double *err = (double *)MemoryManagement::mallocMM(m * N * sizeof(double));
  MemoryManagement::memsetMM(err, 0, m * N * sizeof(double));
  // FFT
  BatchedFFT fft(N, 2, 1);
  fft.setInp(priv, 0);
  // Get plain
  for (int i = 0; i < m; i++)
    TrlweFunction::getPlain(&fft, i & 1, pub + N * 2 * i, N, 1, plain + N * i);
  fft.waitAllOut();
  // Round plain
  for (int i = 0; i < m; i++)
    TrlweFunction::roundPlain(plain + N * i, err + N * i, N);
  // Check plain + error
  TorusInteger *hPlain = new TorusInteger[m * N];
  double *hErr = new double[m * N];
  MemoryManagement::memcpyMM_d2h(hPlain, plain, m * N * sizeof(TorusInteger));
  MemoryManagement::memcpyMM_d2h(hErr, err, m * N * sizeof(double));
  double avgErr = 0;
  bool chk = true;
  for (int i = 0; i < m * N; i++) {
    if (hPlain[i] != 0)
      chk = false;
    if (hErr[i] > 0.125)
      chk = false;
    avgErr += hErr[i];
  }
  std::cout << avgErr / (m * N) << std::endl;
  // Free all
  MemoryManagement::freeMM(priv);
  MemoryManagement::freeMM(pub);
  MemoryManagement::freeMM(plain);
  MemoryManagement::freeMM(err);
  delete[] hPlain;
  delete[] hErr;
  return chk;
}
bool genkey() {
  // Create parties
  MpcApplication party_0(numParty, 0, N, m, l, sdFresh);
  MpcApplication party_1(numParty, 1, N, m, l, sdFresh);
  MpcApplication party_2(numParty, 2, N, m, l, sdFresh);
  // Create private keys
  party_0.createPrivkey();
  party_1.createPrivkey();
  party_2.createPrivkey();
  // Create public keys
  party_0.createPubkey();
  party_1.createPubkey();
  party_2.createPubkey();
  // Export & test keys
  {
    void *privKey = std::malloc(party_0.getSizePrivkey());
    void *pubKey = std::malloc(party_0.getSizePubkey());
    party_0.exportPrivkey(privKey);
    party_0.exportPubkey(pubKey);
    save_data("PrivKey_0_3", privKey, party_0.getSizePrivkey());
    save_data("PubKey_0_3", pubKey, party_0.getSizePubkey());
    bool testKey = test_genkey(privKey, pubKey);
    std::free(privKey);
    std::free(pubKey);
    if (!testKey)
      return false;
  }
  {
    void *privKey = std::malloc(party_1.getSizePrivkey());
    void *pubKey = std::malloc(party_1.getSizePubkey());
    party_1.exportPrivkey(privKey);
    party_1.exportPubkey(pubKey);
    save_data("PrivKey_1_3", privKey, party_1.getSizePrivkey());
    save_data("PubKey_1_3", pubKey, party_1.getSizePubkey());
    bool testKey = test_genkey(privKey, pubKey);
    std::free(privKey);
    std::free(pubKey);
    if (!testKey)
      return false;
  }
  {
    void *privKey = std::malloc(party_2.getSizePrivkey());
    void *pubKey = std::malloc(party_2.getSizePubkey());
    party_2.exportPrivkey(privKey);
    party_2.exportPubkey(pubKey);
    save_data("PrivKey_2_3", privKey, party_2.getSizePrivkey());
    save_data("PubKey_2_3", pubKey, party_2.getSizePubkey());
    bool testKey = test_genkey(privKey, pubKey);
    std::free(privKey);
    std::free(pubKey);
    if (!testKey)
      return false;
  }
  return true;
}
bool test_pre_expand(void *priv_key, void *pub_key, void *pre_expand) {
  // Private key
  TorusInteger *priv =
      (TorusInteger *)MemoryManagement::mallocMM(N * sizeof(TorusInteger));
  MemoryManagement::memcpyMM_h2d(priv, priv_key, N * sizeof(TorusInteger));
  // Public key
  TorusInteger *pub = (TorusInteger *)MemoryManagement::mallocMM(
      m * N * 2 * sizeof(TorusInteger));
  MemoryManagement::memcpyMM_h2d(pub, pub_key,
                                 m * N * 2 * sizeof(TorusInteger));
  // Pre expand
  TorusInteger *expand =
      (TorusInteger *)MemoryManagement::mallocMM(m * N * sizeof(TorusInteger));
  MemoryManagement::memcpyMM_h2d(expand, pre_expand,
                                 m * N * sizeof(TorusInteger));
  // FFT
  BatchedFFT fft(N, 2, 1);
  fft.setInp(priv, 0);
  // Calculate
  for (int i = 0; i < m; i++) {
    fft.setInp(pub + N * 2 * i, i & 1, 0);
    fft.setMul(i & 1, 0);
    TorusUtility::addVector(expand + N * i, pub + N * (2 * i + 1), N);
    fft.subAllOut(expand + N * i, i & 1);
  }
  fft.waitAllOut();
  // Get error
  TorusInteger *error = new TorusInteger[m * N];
  MemoryManagement::memcpyMM_d2h(error, expand, m * N * sizeof(TorusInteger));
  double avgErr = 0;
  bool chk = true;
  for (int i = 0; i < m * N; i++) {
    double e = std::abs(error[i] / std::pow(2, 8 * sizeof(TorusInteger)));
    if (e > 0.125)
      chk = false;
    avgErr += e;
  }
  std::cout << avgErr / (m * N) << std::endl;
  // Free all
  MemoryManagement::freeMM(priv);
  MemoryManagement::freeMM(pub);
  MemoryManagement::freeMM(expand);
  delete[] error;
  return chk;
}
bool pre_expand() {
  // Create parties
  MpcApplication party_0(numParty, 0, N, m, l, sdFresh);
  MpcApplication party_1(numParty, 1, N, m, l, sdFresh);
  MpcApplication party_2(numParty, 2, N, m, l, sdFresh);
  // Import private keys, create pre expand and test
  {
    bool chk = true;
    void *privKey = std::malloc(party_0.getSizePrivkey());
    void *pubKey = std::malloc(party_0.getSizePubkey());
    void *preExpand = std::malloc(party_0.getSizePreExpand());
    // >>> Import private key
    load_data("PrivKey_0_3", privKey, party_0.getSizePrivkey());
    party_0.importPrivkey(privKey);
    // <<<
    load_data("PubKey_1_3", pubKey, party_0.getSizePubkey());
    party_0.preExpand(pubKey, preExpand);
    save_data("PreExpand_0_1_3", preExpand, party_0.getSizePreExpand());
    chk = test_pre_expand(privKey, pubKey, preExpand) && chk;
    load_data("PubKey_2_3", pubKey, party_0.getSizePubkey());
    party_0.preExpand(pubKey, preExpand);
    save_data("PreExpand_0_2_3", preExpand, party_0.getSizePreExpand());
    chk = test_pre_expand(privKey, pubKey, preExpand) && chk;
    std::free(pubKey);
    std::free(privKey);
    std::free(preExpand);
    if (!chk)
      return false;
  }
  {
    bool chk = true;
    void *privKey = std::malloc(party_1.getSizePrivkey());
    void *pubKey = std::malloc(party_1.getSizePubkey());
    void *preExpand = std::malloc(party_1.getSizePreExpand());
    // >>> Import private key
    load_data("PrivKey_1_3", privKey, party_1.getSizePrivkey());
    party_1.importPrivkey(privKey);
    // <<<
    load_data("PubKey_2_3", pubKey, party_1.getSizePubkey());
    party_1.preExpand(pubKey, preExpand);
    save_data("PreExpand_1_2_3", preExpand, party_1.getSizePreExpand());
    chk = test_pre_expand(privKey, pubKey, preExpand) && chk;
    load_data("PubKey_0_3", pubKey, party_1.getSizePubkey());
    party_1.preExpand(pubKey, preExpand);
    save_data("PreExpand_1_0_3", preExpand, party_1.getSizePreExpand());
    chk = test_pre_expand(privKey, pubKey, preExpand) && chk;
    std::free(pubKey);
    std::free(privKey);
    std::free(preExpand);
    if (!chk)
      return false;
  }
  {
    bool chk = true;
    void *privKey = std::malloc(party_2.getSizePrivkey());
    void *pubKey = std::malloc(party_2.getSizePubkey());
    void *preExpand = std::malloc(party_2.getSizePreExpand());
    // >>> Import private key
    load_data("PrivKey_2_3", privKey, party_2.getSizePrivkey());
    party_2.importPrivkey(privKey);
    // <<<
    load_data("PubKey_0_3", pubKey, party_2.getSizePubkey());
    party_2.preExpand(pubKey, preExpand);
    save_data("PreExpand_2_0_3", preExpand, party_2.getSizePreExpand());
    chk = test_pre_expand(privKey, pubKey, preExpand) && chk;
    load_data("PubKey_1_3", pubKey, party_2.getSizePubkey());
    party_2.preExpand(pubKey, preExpand);
    save_data("PreExpand_2_1_3", preExpand, party_2.getSizePreExpand());
    chk = test_pre_expand(privKey, pubKey, preExpand) && chk;
    std::free(pubKey);
    std::free(privKey);
    std::free(preExpand);
    if (!chk)
      return false;
  }
  return true;
}
bool test_encrypt(bool msg, void *priv_key, void *pub_key, void *mainCipher,
                  void *randCipher, void *random) {
  // Private key
  TorusInteger *priv =
      (TorusInteger *)MemoryManagement::mallocMM(N * sizeof(TorusInteger));
  MemoryManagement::memcpyMM_h2d(priv, priv_key, N * sizeof(TorusInteger));
  // Public key
  TorusInteger *pub = (TorusInteger *)MemoryManagement::mallocMM(
      m * N * 2 * sizeof(TorusInteger));
  MemoryManagement::memcpyMM_h2d(pub, pub_key,
                                 m * N * 2 * sizeof(TorusInteger));
  // Random
  TorusInteger *dRandom = (TorusInteger *)MemoryManagement::mallocMM(
      2 * l * m * N * sizeof(TorusInteger));
  MemoryManagement::memcpyMM_h2d(dRandom, random,
                                 2 * l * m * N * sizeof(TorusInteger));
  TorusInteger *hRandom = (TorusInteger *)random;
  // Test main cipher
  bool chk = true;
  double avgErr = 0;
  TrgswCipher trgsw(N, 1, l, 1, sdFresh * (m * N + 1),
                    sdFresh * sdFresh * (m * N + 1));
  MemoryManagement::memcpyMM_h2d(trgsw._data, mainCipher,
                                 4 * l * N * sizeof(TorusInteger));
  if (msg)
    TrgswFunction::addMuGadget(-1, &trgsw);
  {
    BatchedFFT fft(N, 2 * l, m);
    for (int i = 0; i < 2 * l; i++) {
      for (int j = 0; j < m; j++)
        fft.setInp(dRandom + (i * m + j) * N, i, j);
    }
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < m; j++)
        fft.setInp(pub + (j * 2 + i) * N, j);
      for (int j = 0; j < 2 * l; j++) {
        for (int k = 0; k < m; k++)
          fft.setMul(j, k);
      }
      for (int j = 0; j < 2 * l; j++)
        fft.subAllOut(trgsw.get_pol_data(j, i), j);
    }
    fft.waitAllOut();
  }
  // Test random cipher
  TorusInteger *dRandCipher = (TorusInteger *)MemoryManagement::mallocMM(
      l * m * 4 * l * N * sizeof(TorusInteger));
  MemoryManagement::memcpyMM_h2d(dRandCipher, randCipher,
                                 l * m * 4 * l * N * sizeof(TorusInteger));
  TorusInteger *hRandCipher = (TorusInteger *)randCipher;
  {
    BatchedFFT fft(N, 2, 1);
    fft.setInp(priv, 0);
    for (int i = 0; i < 2 * l * m * l; i++) {
      fft.setInp(dRandCipher + 2 * i * N, i & 1, 0);
      fft.setMul(i & 1, 0);
      fft.subAllOut(dRandCipher + (2 * i + 1) * N, i & 1);
    }
    fft.waitAllOut();
  }
  // Get error
  MemoryManagement::memcpyMM_d2h(mainCipher, trgsw._data,
                                 4 * l * N * sizeof(TorusInteger));
  for (int i = 0; i < 4 * l * N; i++) {
    TorusInteger *ptr = (TorusInteger *)mainCipher;
    double e = std::abs(ptr[i] / std::pow(2, 8 * sizeof(TorusInteger)));
    if (e > 0.125)
      chk = false;
    avgErr += e;
  }
  MemoryManagement::memcpyMM_d2h(randCipher, dRandCipher,
                                 l * m * 4 * l * N * sizeof(TorusInteger));
  for (int i = 0; i < l; i++) {
    TorusInteger H = 1;
    H <<= 8 * sizeof(TorusInteger) - (i + 1);
    for (int j = 0; j < 2 * l; j++) {
      for (int k = 0; k < m; k++) {
        for (int h = 0; h < N; h++) {
          TorusInteger temp =
              hRandCipher[((j * m + k) * 2 * l + 2 * i + 1) * N + h];
          temp -= hRandom[(j * m + k) * N + h] * H;
          double e = std::abs(temp / std::pow(2, 8 * sizeof(TorusInteger)));
          if (e > 0.125)
            chk = false;
          avgErr += e;
        }
      }
    }
  }
  std::cout << avgErr / (2 * l * N * (1 + l * m)) << std::endl;
  // Free all
  MemoryManagement::freeMM(priv);
  MemoryManagement::freeMM(pub);
  MemoryManagement::freeMM(dRandom);
  MemoryManagement::freeMM(dRandCipher);
  return chk;
}
bool encrypt() {
  // Create plaintexts
  std::srand(std::time(nullptr));
  bool plain_0 = std::rand() & 1;
  bool plain_1 = std::rand() & 1;
  bool plain_2 = std::rand() & 1;
  save_data("Plain_0", &plain_0, sizeof(bool));
  save_data("Plain_1", &plain_1, sizeof(bool));
  save_data("Plain_2", &plain_2, sizeof(bool));
  // Create parties
  MpcApplication party_0(numParty, 0, N, m, l, sdFresh);
  MpcApplication party_1(numParty, 1, N, m, l, sdFresh);
  MpcApplication party_2(numParty, 2, N, m, l, sdFresh);
  // Encrypt
  {
    bool chk = true;
    void *privKey = std::malloc(party_0.getSizePrivkey());
    void *pubKey = std::malloc(party_0.getSizePubkey());
    void *mainCipher = std::malloc(party_0.getSizeMainCipher());
    void *randCipher = std::malloc(party_0.getSizeRandCipher());
    void *random = std::malloc(party_0.getSizeRandom());
    // >>> Import keys
    load_data("PrivKey_0_3", privKey, party_0.getSizePrivkey());
    party_0.importPrivkey(privKey);
    load_data("PubKey_0_3", pubKey, party_0.getSizePubkey());
    party_0.importPubkey(pubKey);
    // <<<
    party_0.encrypt(plain_0, mainCipher, randCipher, random);
    save_data("MainCipher_0", mainCipher, party_0.getSizeMainCipher());
    save_data("RandCipher_0", randCipher, party_0.getSizeRandCipher());
    save_data("Random_0", random, party_0.getSizeRandom());
    chk = test_encrypt(plain_0, privKey, pubKey, mainCipher, randCipher,
                       random) &&
          chk;
    std::free(privKey);
    std::free(pubKey);
    std::free(mainCipher);
    std::free(randCipher);
    std::free(random);
    if (!chk)
      return false;
  }
  {
    bool chk = true;
    void *privKey = std::malloc(party_1.getSizePrivkey());
    void *pubKey = std::malloc(party_1.getSizePubkey());
    void *mainCipher = std::malloc(party_1.getSizeMainCipher());
    void *randCipher = std::malloc(party_1.getSizeRandCipher());
    void *random = std::malloc(party_1.getSizeRandom());
    // >>> Import keys
    load_data("PrivKey_1_3", privKey, party_1.getSizePrivkey());
    party_1.importPrivkey(privKey);
    load_data("PubKey_1_3", pubKey, party_1.getSizePubkey());
    party_1.importPubkey(pubKey);
    // <<<
    party_1.encrypt(plain_1, mainCipher, randCipher, random);
    save_data("MainCipher_1", mainCipher, party_1.getSizeMainCipher());
    save_data("RandCipher_1", randCipher, party_1.getSizeRandCipher());
    save_data("Random_1", random, party_1.getSizeRandom());
    chk = test_encrypt(plain_1, privKey, pubKey, mainCipher, randCipher,
                       random) &&
          chk;
    std::free(privKey);
    std::free(pubKey);
    std::free(mainCipher);
    std::free(randCipher);
    std::free(random);
    if (!chk)
      return false;
  }
  {
    bool chk = true;
    void *privKey = std::malloc(party_2.getSizePrivkey());
    void *pubKey = std::malloc(party_2.getSizePubkey());
    void *mainCipher = std::malloc(party_2.getSizeMainCipher());
    void *randCipher = std::malloc(party_2.getSizeRandCipher());
    void *random = std::malloc(party_2.getSizeRandom());
    // >>> Import keys
    load_data("PrivKey_2_3", privKey, party_2.getSizePrivkey());
    party_2.importPrivkey(privKey);
    load_data("PubKey_2_3", pubKey, party_2.getSizePubkey());
    party_2.importPubkey(pubKey);
    // <<<
    party_2.encrypt(plain_2, mainCipher, randCipher, random);
    save_data("MainCipher_2", mainCipher, party_2.getSizeMainCipher());
    save_data("RandCipher_2", randCipher, party_2.getSizeRandCipher());
    save_data("Random_2", random, party_2.getSizeRandom());
    chk = test_encrypt(plain_2, privKey, pubKey, mainCipher, randCipher,
                       random) &&
          chk;
    std::free(privKey);
    std::free(pubKey);
    std::free(mainCipher);
    std::free(randCipher);
    std::free(random);
    if (!chk)
      return false;
  }
  return true;
}
bool expand_partDec() {
  // Create parties
  MpcApplication party_0(numParty, 0, N, m, l, sdFresh);
  MpcApplication party_1(numParty, 1, N, m, l, sdFresh);
  MpcApplication party_2(numParty, 2, N, m, l, sdFresh);
  // Import keys
  {
    void *privKey = std::malloc(party_0.getSizePrivkey());
    void *pubKey = std::malloc(party_0.getSizePubkey());
    load_data("PrivKey_0_3", privKey, party_0.getSizePrivkey());
    party_0.importPrivkey(privKey);
    load_data("PubKey_0_3", pubKey, party_0.getSizePubkey());
    party_0.importPubkey(pubKey);
    std::free(privKey);
    std::free(pubKey);
  }
  {
    void *privKey = std::malloc(party_1.getSizePrivkey());
    void *pubKey = std::malloc(party_1.getSizePubkey());
    load_data("PrivKey_1_3", privKey, party_1.getSizePrivkey());
    party_1.importPrivkey(privKey);
    load_data("PubKey_1_3", pubKey, party_1.getSizePubkey());
    party_1.importPubkey(pubKey);
    std::free(privKey);
    std::free(pubKey);
  }
  {
    void *privKey = std::malloc(party_2.getSizePrivkey());
    void *pubKey = std::malloc(party_2.getSizePubkey());
    load_data("PrivKey_2_3", privKey, party_2.getSizePrivkey());
    party_2.importPrivkey(privKey);
    load_data("PubKey_2_3", pubKey, party_2.getSizePubkey());
    party_2.importPubkey(pubKey);
    std::free(privKey);
    std::free(pubKey);
  }
  // Expand and Decrypt
  bool chk = true;
  std::vector<void *> preExpand(3);
  std::vector<TorusInteger> partPlain(3);
  bool plain, oriPlain;
  double error;
  {
    void *mainCipher = std::malloc(party_0.getSizeMainCipher());
    void *randCipher = std::malloc(party_0.getSizeRandCipher());
    void *random = std::malloc(party_0.getSizeRandom());
    load_data("MainCipher_0", mainCipher, party_0.getSizeMainCipher());
    load_data("RandCipher_0", randCipher, party_0.getSizeRandCipher());
    load_data("Random_0", random, party_0.getSizeRandom());
    load_data("Plain_0", &oriPlain, sizeof(bool));
    preExpand[0] = nullptr;
    preExpand[1] = std::malloc(party_0.getSizePreExpand());
    load_data("PreExpand_1_0_3", preExpand[1], party_0.getSizePreExpand());
    preExpand[2] = std::malloc(party_0.getSizePreExpand());
    load_data("PreExpand_2_0_3", preExpand[2], party_0.getSizePreExpand());
    {
      auto expandCipher = party_0.expandWithPlainRandom(preExpand, nullptr, 0,
                                                        mainCipher, random);
      partPlain[0] = party_0.partDec(expandCipher);
      partPlain[1] = party_1.partDec(expandCipher);
      partPlain[2] = party_2.partDec(expandCipher);
      plain = party_0.finDec(partPlain.data(), &error);
      chk = (plain == oriPlain) && chk;
      chk = (error < 0.125) && chk;
      std::cout << plain << " " << oriPlain << " " << error << std::endl;
      std::cout << expandCipher->_sdError << " "
                << std::sqrt(expandCipher->_varError) << std::endl;
      delete expandCipher;
    }
    {
      auto expandCipher =
          party_0.expand(preExpand, nullptr, 0, mainCipher, randCipher);
      partPlain[0] = party_0.partDec(expandCipher);
      partPlain[1] = party_1.partDec(expandCipher);
      partPlain[2] = party_2.partDec(expandCipher);
      plain = party_0.finDec(partPlain.data(), &error);
      chk = (plain == oriPlain) && chk;
      chk = (error < 0.125) && chk;
      std::cout << plain << " " << oriPlain << " " << error << std::endl;
      std::cout << expandCipher->_sdError << " "
                << std::sqrt(expandCipher->_varError) << std::endl;
      delete expandCipher;
    }
    preExpand[0] = nullptr;
    std::free(preExpand[1]);
    preExpand[1] = nullptr;
    std::free(preExpand[2]);
    preExpand[2] = nullptr;
    std::free(mainCipher);
    std::free(randCipher);
    std::free(random);
  }
  {
    void *mainCipher = std::malloc(party_1.getSizeMainCipher());
    void *randCipher = std::malloc(party_1.getSizeRandCipher());
    void *random = std::malloc(party_1.getSizeRandom());
    load_data("MainCipher_1", mainCipher, party_1.getSizeMainCipher());
    load_data("RandCipher_1", randCipher, party_1.getSizeRandCipher());
    load_data("Random_1", random, party_1.getSizeRandom());
    load_data("Plain_1", &oriPlain, sizeof(bool));
    preExpand[0] = std::malloc(party_1.getSizePreExpand());
    load_data("PreExpand_0_1_3", preExpand[0], party_1.getSizePreExpand());
    preExpand[1] = nullptr;
    preExpand[2] = std::malloc(party_1.getSizePreExpand());
    load_data("PreExpand_2_1_3", preExpand[2], party_1.getSizePreExpand());
    {
      auto expandCipher = party_1.expandWithPlainRandom(preExpand, nullptr, 1,
                                                        mainCipher, random);
      partPlain[0] = party_0.partDec(expandCipher);
      partPlain[1] = party_1.partDec(expandCipher);
      partPlain[2] = party_2.partDec(expandCipher);
      plain = party_1.finDec(partPlain.data(), &error);
      chk = (plain == oriPlain) && chk;
      chk = (error < 0.125) && chk;
      std::cout << plain << " " << oriPlain << " " << error << std::endl;
      std::cout << expandCipher->_sdError << " "
                << std::sqrt(expandCipher->_varError) << std::endl;
      delete expandCipher;
    }
    {
      auto expandCipher =
          party_1.expand(preExpand, nullptr, 1, mainCipher, randCipher);
      partPlain[0] = party_0.partDec(expandCipher);
      partPlain[1] = party_1.partDec(expandCipher);
      partPlain[2] = party_2.partDec(expandCipher);
      plain = party_1.finDec(partPlain.data(), &error);
      chk = (plain == oriPlain) && chk;
      chk = (error < 0.125) && chk;
      std::cout << plain << " " << oriPlain << " " << error << std::endl;
      std::cout << expandCipher->_sdError << " "
                << std::sqrt(expandCipher->_varError) << std::endl;
      delete expandCipher;
    }
    std::free(preExpand[0]);
    preExpand[0] = nullptr;
    preExpand[1] = nullptr;
    std::free(preExpand[2]);
    preExpand[2] = nullptr;
    std::free(mainCipher);
    std::free(randCipher);
    std::free(random);
  }
  {
    void *mainCipher = std::malloc(party_2.getSizeMainCipher());
    void *randCipher = std::malloc(party_2.getSizeRandCipher());
    void *random = std::malloc(party_2.getSizeRandom());
    load_data("MainCipher_2", mainCipher, party_2.getSizeMainCipher());
    load_data("RandCipher_2", randCipher, party_2.getSizeRandCipher());
    load_data("Random_2", random, party_2.getSizeRandom());
    load_data("Plain_2", &oriPlain, sizeof(bool));
    preExpand[0] = std::malloc(party_2.getSizePreExpand());
    load_data("PreExpand_0_2_3", preExpand[0], party_2.getSizePreExpand());
    preExpand[1] = std::malloc(party_2.getSizePreExpand());
    load_data("PreExpand_1_2_3", preExpand[1], party_2.getSizePreExpand());
    preExpand[2] = nullptr;
    {
      auto expandCipher = party_2.expandWithPlainRandom(preExpand, nullptr, 2,
                                                        mainCipher, random);
      partPlain[0] = party_0.partDec(expandCipher);
      partPlain[1] = party_1.partDec(expandCipher);
      partPlain[2] = party_2.partDec(expandCipher);
      plain = party_2.finDec(partPlain.data(), &error);
      chk = (plain == oriPlain) && chk;
      chk = (error < 0.125) && chk;
      std::cout << plain << " " << oriPlain << " " << error << std::endl;
      std::cout << expandCipher->_sdError << " "
                << std::sqrt(expandCipher->_varError) << std::endl;
      delete expandCipher;
    }
    {
      auto expandCipher =
          party_2.expand(preExpand, nullptr, 2, mainCipher, randCipher);
      partPlain[0] = party_0.partDec(expandCipher);
      partPlain[1] = party_1.partDec(expandCipher);
      partPlain[2] = party_2.partDec(expandCipher);
      plain = party_2.finDec(partPlain.data(), &error);
      chk = (plain == oriPlain) && chk;
      chk = (error < 0.125) && chk;
      std::cout << plain << " " << oriPlain << " " << error << std::endl;
      std::cout << expandCipher->_sdError << " "
                << std::sqrt(expandCipher->_varError) << std::endl;
      delete expandCipher;
    }
    std::free(preExpand[0]);
    preExpand[0] = nullptr;
    std::free(preExpand[1]);
    preExpand[1] = nullptr;
    preExpand[2] = nullptr;
    std::free(mainCipher);
    std::free(randCipher);
    std::free(random);
  }
  return chk;
}
bool test_operator() {
  // Create parties
  MpcApplication party_0(numParty, 0, N, m, l, sdFresh);
  MpcApplication party_1(numParty, 1, N, m, l, sdFresh);
  MpcApplication party_2(numParty, 2, N, m, l, sdFresh);
  // Import keys
  {
    void *privKey = std::malloc(party_0.getSizePrivkey());
    void *pubKey = std::malloc(party_0.getSizePubkey());
    load_data("PrivKey_0_3", privKey, party_0.getSizePrivkey());
    party_0.importPrivkey(privKey);
    load_data("PubKey_0_3", pubKey, party_0.getSizePubkey());
    party_0.importPubkey(pubKey);
    std::free(privKey);
    std::free(pubKey);
  }
  {
    void *privKey = std::malloc(party_1.getSizePrivkey());
    void *pubKey = std::malloc(party_1.getSizePubkey());
    load_data("PrivKey_1_3", privKey, party_1.getSizePrivkey());
    party_1.importPrivkey(privKey);
    load_data("PubKey_1_3", pubKey, party_1.getSizePubkey());
    party_1.importPubkey(pubKey);
    std::free(privKey);
    std::free(pubKey);
  }
  {
    void *privKey = std::malloc(party_2.getSizePrivkey());
    void *pubKey = std::malloc(party_2.getSizePubkey());
    load_data("PrivKey_2_3", privKey, party_2.getSizePrivkey());
    party_2.importPrivkey(privKey);
    load_data("PubKey_2_3", pubKey, party_2.getSizePubkey());
    party_2.importPubkey(pubKey);
    std::free(privKey);
    std::free(pubKey);
  }
  // Get expand ciphers
  TrgswCipher *cipher_0, *cipher_1;
  std::vector<void *> preExpand(3);
  std::vector<TorusInteger> partPlain(3);
  std::vector<bool> plain(3);
  bool oriPlain;
  {
    void *mainCipher = std::malloc(party_0.getSizeMainCipher());
    void *random = std::malloc(party_0.getSizeRandom());
    load_data("MainCipher_0", mainCipher, party_0.getSizeMainCipher());
    load_data("Random_0", random, party_0.getSizeRandom());
    load_data("Plain_0", &oriPlain, sizeof(bool));
    plain[0] = oriPlain;
    preExpand[0] = nullptr;
    preExpand[1] = std::malloc(party_0.getSizePreExpand());
    load_data("PreExpand_1_0_3", preExpand[1], party_0.getSizePreExpand());
    preExpand[2] = std::malloc(party_0.getSizePreExpand());
    load_data("PreExpand_2_0_3", preExpand[2], party_0.getSizePreExpand());
    cipher_0 = party_0.expandWithPlainRandom(
        preExpand, [](void *ptr) { std::free(ptr); }, 0, mainCipher, random);
    std::free(mainCipher);
    std::free(random);
  }
  {
    void *mainCipher = std::malloc(party_1.getSizeMainCipher());
    void *random = std::malloc(party_1.getSizeRandom());
    load_data("MainCipher_1", mainCipher, party_1.getSizeMainCipher());
    load_data("Random_1", random, party_1.getSizeRandom());
    load_data("Plain_1", &oriPlain, sizeof(bool));
    plain[1] = oriPlain;
    preExpand[0] = std::malloc(party_1.getSizePreExpand());
    load_data("PreExpand_0_1_3", preExpand[0], party_1.getSizePreExpand());
    preExpand[1] = nullptr;
    preExpand[2] = std::malloc(party_1.getSizePreExpand());
    load_data("PreExpand_2_1_3", preExpand[2], party_1.getSizePreExpand());
    cipher_1 = party_1.expandWithPlainRandom(
        preExpand, [](void *ptr) { std::free(ptr); }, 1, mainCipher, random);
    std::free(mainCipher);
    std::free(random);
  }
  // Test
  bool chk = true;
  double error = 0;
  {
    partPlain[0] = party_0.partDec(cipher_0);
    partPlain[1] = party_1.partDec(cipher_0);
    partPlain[2] = party_2.partDec(cipher_0);
    chk = (party_0.finDec(partPlain.data(), &error) == plain[0]) && chk;
    std::cout << error << " " << cipher_0->_sdError << " "
              << std::sqrt(cipher_0->_varError) << std::endl;
  }
  {
    partPlain[0] = party_0.partDec(cipher_1);
    partPlain[1] = party_1.partDec(cipher_1);
    partPlain[2] = party_2.partDec(cipher_1);
    chk = (party_1.finDec(partPlain.data(), &error) == plain[1]) && chk;
    std::cout << error << " " << cipher_1->_sdError << " "
              << std::sqrt(cipher_1->_varError) << std::endl;
  }
  {
    auto cipher = party_2.addOp(cipher_0, cipher_1);
    partPlain[0] = party_0.partDec(cipher);
    partPlain[1] = party_1.partDec(cipher);
    partPlain[2] = party_2.partDec(cipher);
    if ((plain[0] && plain[1]) || (!plain[0] && !plain[1]))
      oriPlain = false;
    else
      oriPlain = true;
    chk = (party_2.finDec(partPlain.data(), &error) == oriPlain) && chk;
    std::cout << error << " " << cipher->_sdError << " "
              << std::sqrt(cipher->_varError) << std::endl;
    delete cipher;
  }
  {
    auto cipher = party_2.subOp(cipher_0, cipher_1);
    partPlain[0] = party_0.partDec(cipher);
    partPlain[1] = party_1.partDec(cipher);
    partPlain[2] = party_2.partDec(cipher);
    if ((plain[0] && plain[1]) || (!plain[0] && !plain[1]))
      oriPlain = false;
    else
      oriPlain = true;
    chk = (party_2.finDec(partPlain.data(), &error) == oriPlain) && chk;
    std::cout << error << " " << cipher->_sdError << " "
              << std::sqrt(cipher->_varError) << std::endl;
    delete cipher;
  }
  {
    auto cipher = party_2.notOp(cipher_0);
    partPlain[0] = party_0.partDec(cipher);
    partPlain[1] = party_1.partDec(cipher);
    partPlain[2] = party_2.partDec(cipher);
    chk = (party_2.finDec(partPlain.data(), &error) != plain[0]) && chk;
    std::cout << error << " " << cipher->_sdError << " "
              << std::sqrt(cipher->_varError) << std::endl;
    delete cipher;
  }
  {
    auto cipher = party_2.notOp(cipher_1);
    partPlain[0] = party_0.partDec(cipher);
    partPlain[1] = party_1.partDec(cipher);
    partPlain[2] = party_2.partDec(cipher);
    chk = (party_2.finDec(partPlain.data(), &error) != plain[1]) && chk;
    std::cout << error << " " << cipher->_sdError << " "
              << std::sqrt(cipher->_varError) << std::endl;
    delete cipher;
  }
  {
    auto cipher = party_2.notXorOp(cipher_0, cipher_1);
    partPlain[0] = party_0.partDec(cipher);
    partPlain[1] = party_1.partDec(cipher);
    partPlain[2] = party_2.partDec(cipher);
    if ((plain[0] && plain[1]) || (!plain[0] && !plain[1]))
      oriPlain = true;
    else
      oriPlain = false;
    chk = (party_2.finDec(partPlain.data(), &error) == oriPlain) && chk;
    std::cout << error << " " << cipher->_sdError << " "
              << std::sqrt(cipher->_varError) << std::endl;
    delete cipher;
  }
  {
    DECLARE_TIMING(Mul);
    START_TIMING(Mul);
    auto cipher = party_2.mulOp(cipher_0, cipher_1);
    STOP_TIMING(Mul);
    partPlain[0] = party_0.partDec(cipher);
    partPlain[1] = party_1.partDec(cipher);
    partPlain[2] = party_2.partDec(cipher);
    if (plain[0] && plain[1])
      oriPlain = true;
    else
      oriPlain = false;
    chk = (party_2.finDec(partPlain.data(), &error) == oriPlain) && chk;
    std::cout << error << " " << cipher->_sdError << " "
              << std::sqrt(cipher->_varError) << std::endl;
    PRINT_TIMING(Mul);
    delete cipher;
  }
  delete cipher_0;
  delete cipher_1;
  return chk;
}
bool reduce() {
  // Create parties
  MpcApplication party_0(numParty, 0, N, m, l, sdFresh);
  MpcApplication party_1(numParty, 1, N, m, l, sdFresh);
  MpcApplication party_2(numParty, 2, N, m, l, sdFresh);
  // Import keys
  {
    void *privKey = std::malloc(party_0.getSizePrivkey());
    void *pubKey = std::malloc(party_0.getSizePubkey());
    load_data("PrivKey_0_3", privKey, party_0.getSizePrivkey());
    party_0.importPrivkey(privKey);
    load_data("PubKey_0_3", pubKey, party_0.getSizePubkey());
    party_0.importPubkey(pubKey);
    std::free(privKey);
    std::free(pubKey);
  }
  {
    void *privKey = std::malloc(party_1.getSizePrivkey());
    void *pubKey = std::malloc(party_1.getSizePubkey());
    load_data("PrivKey_1_3", privKey, party_1.getSizePrivkey());
    party_1.importPrivkey(privKey);
    load_data("PubKey_1_3", pubKey, party_1.getSizePubkey());
    party_1.importPubkey(pubKey);
    std::free(privKey);
    std::free(pubKey);
  }
  {
    void *privKey = std::malloc(party_2.getSizePrivkey());
    void *pubKey = std::malloc(party_2.getSizePubkey());
    load_data("PrivKey_2_3", privKey, party_2.getSizePrivkey());
    party_2.importPrivkey(privKey);
    load_data("PubKey_2_3", pubKey, party_2.getSizePubkey());
    party_2.importPubkey(pubKey);
    std::free(privKey);
    std::free(pubKey);
  }
  // Get expand ciphers
  TrgswCipher *cipher_0, *cipher_1;
  std::vector<void *> preExpand(3);
  std::vector<TorusInteger> partPlain(3);
  std::vector<bool> plain(3);
  bool oriPlain;
  {
    void *mainCipher = std::malloc(party_0.getSizeMainCipher());
    void *random = std::malloc(party_0.getSizeRandom());
    load_data("MainCipher_0", mainCipher, party_0.getSizeMainCipher());
    load_data("Random_0", random, party_0.getSizeRandom());
    load_data("Plain_0", &oriPlain, sizeof(bool));
    plain[0] = oriPlain;
    preExpand[0] = nullptr;
    preExpand[1] = std::malloc(party_0.getSizePreExpand());
    load_data("PreExpand_1_0_3", preExpand[1], party_0.getSizePreExpand());
    preExpand[2] = std::malloc(party_0.getSizePreExpand());
    load_data("PreExpand_2_0_3", preExpand[2], party_0.getSizePreExpand());
    cipher_0 = party_0.expandWithPlainRandom(
        preExpand, [](void *ptr) { std::free(ptr); }, 0, mainCipher, random);
    std::free(mainCipher);
    std::free(random);
  }
  {
    void *mainCipher = std::malloc(party_1.getSizeMainCipher());
    void *random = std::malloc(party_1.getSizeRandom());
    load_data("MainCipher_1", mainCipher, party_1.getSizeMainCipher());
    load_data("Random_1", random, party_1.getSizeRandom());
    load_data("Plain_1", &oriPlain, sizeof(bool));
    plain[1] = oriPlain;
    preExpand[0] = std::malloc(party_1.getSizePreExpand());
    load_data("PreExpand_0_1_3", preExpand[0], party_1.getSizePreExpand());
    preExpand[1] = nullptr;
    preExpand[2] = std::malloc(party_1.getSizePreExpand());
    load_data("PreExpand_2_1_3", preExpand[2], party_1.getSizePreExpand());
    cipher_1 = party_1.expandWithPlainRandom(
        preExpand, [](void *ptr) { std::free(ptr); }, 1, mainCipher, random);
    std::free(mainCipher);
    std::free(random);
  }
  // Test
  bool chk = true;
  double error = 0;
  {
    auto cipher = party_0.reduce(cipher_0);
    partPlain[0] = party_0.partDec(cipher);
    partPlain[1] = party_1.partDec(cipher);
    partPlain[2] = party_2.partDec(cipher);
    chk = (party_0.finDec(partPlain.data(), &error) == plain[0]) && chk;
    std::cout << error << " " << cipher->_sdError << " "
              << std::sqrt(cipher->_varError) << std::endl;
    delete cipher;
  }
  {
    auto cipher = party_1.reduce(cipher_1);
    partPlain[0] = party_0.partDec(cipher);
    partPlain[1] = party_1.partDec(cipher);
    partPlain[2] = party_2.partDec(cipher);
    chk = (party_1.finDec(partPlain.data(), &error) == plain[1]) && chk;
    std::cout << error << " " << cipher->_sdError << " "
              << std::sqrt(cipher->_varError) << std::endl;
    delete cipher;
  }
  {
    auto red_cipher_0 = party_0.reduce(cipher_0);
    auto red_cipher_1 = party_1.reduce(cipher_1);
    auto cipher = party_2.addOp(red_cipher_0, red_cipher_1);
    partPlain[0] = party_0.partDec(cipher);
    partPlain[1] = party_1.partDec(cipher);
    partPlain[2] = party_2.partDec(cipher);
    if ((plain[0] && plain[1]) || (!plain[0] && !plain[1]))
      oriPlain = false;
    else
      oriPlain = true;
    chk = (party_2.finDec(partPlain.data(), &error) == oriPlain) && chk;
    std::cout << error << " " << cipher->_sdError << " "
              << std::sqrt(cipher->_varError) << std::endl;
    delete red_cipher_0;
    delete red_cipher_1;
    delete cipher;
  }
  {
    auto red_cipher_0 = party_0.reduce(cipher_0);
    auto red_cipher_1 = party_1.reduce(cipher_1);
    auto cipher = party_2.subOp(red_cipher_0, red_cipher_1);
    partPlain[0] = party_0.partDec(cipher);
    partPlain[1] = party_1.partDec(cipher);
    partPlain[2] = party_2.partDec(cipher);
    if ((plain[0] && plain[1]) || (!plain[0] && !plain[1]))
      oriPlain = false;
    else
      oriPlain = true;
    chk = (party_2.finDec(partPlain.data(), &error) == oriPlain) && chk;
    std::cout << error << " " << cipher->_sdError << " "
              << std::sqrt(cipher->_varError) << std::endl;
    delete red_cipher_0;
    delete red_cipher_1;
    delete cipher;
  }
  {
    auto red_cipher_0 = party_0.reduce(cipher_0);
    auto cipher = party_2.notOp(red_cipher_0);
    partPlain[0] = party_0.partDec(cipher);
    partPlain[1] = party_1.partDec(cipher);
    partPlain[2] = party_2.partDec(cipher);
    chk = (party_2.finDec(partPlain.data(), &error) != plain[0]) && chk;
    std::cout << error << " " << cipher->_sdError << " "
              << std::sqrt(cipher->_varError) << std::endl;
    delete red_cipher_0;
    delete cipher;
  }
  {
    auto red_cipher_1 = party_1.reduce(cipher_1);
    auto cipher = party_2.notOp(red_cipher_1);
    partPlain[0] = party_0.partDec(cipher);
    partPlain[1] = party_1.partDec(cipher);
    partPlain[2] = party_2.partDec(cipher);
    chk = (party_2.finDec(partPlain.data(), &error) != plain[1]) && chk;
    std::cout << error << " " << cipher->_sdError << " "
              << std::sqrt(cipher->_varError) << std::endl;
    delete red_cipher_1;
    delete cipher;
  }
  {
    auto red_cipher_0 = party_0.reduce(cipher_0);
    auto red_cipher_1 = party_1.reduce(cipher_1);
    auto cipher = party_2.notXorOp(red_cipher_0, red_cipher_1);
    partPlain[0] = party_0.partDec(cipher);
    partPlain[1] = party_1.partDec(cipher);
    partPlain[2] = party_2.partDec(cipher);
    if ((plain[0] && plain[1]) || (!plain[0] && !plain[1]))
      oriPlain = true;
    else
      oriPlain = false;
    chk = (party_2.finDec(partPlain.data(), &error) == oriPlain) && chk;
    std::cout << error << " " << cipher->_sdError << " "
              << std::sqrt(cipher->_varError) << std::endl;
    delete red_cipher_0;
    delete red_cipher_1;
    delete cipher;
  }
  {
    DECLARE_TIMING(Mul);
    auto red_cipher_0 = party_0.reduce(cipher_0);
    START_TIMING(Mul);
    auto cipher = party_2.mulOp(red_cipher_0, cipher_1);
    STOP_TIMING(Mul);
    partPlain[0] = party_0.partDec(cipher);
    partPlain[1] = party_1.partDec(cipher);
    partPlain[2] = party_2.partDec(cipher);
    if (plain[0] && plain[1])
      oriPlain = true;
    else
      oriPlain = false;
    chk = (party_2.finDec(partPlain.data(), &error) == oriPlain) && chk;
    std::cout << error << " " << cipher->_sdError << " "
              << std::sqrt(cipher->_varError) << std::endl;
    PRINT_TIMING(Mul);
    delete red_cipher_0;
    delete cipher;
  }
  delete cipher_0;
  delete cipher_1;
  return chk;
}
