#include "gtest/gtest.h"

#include "mpc_application.h"
#include "thesis/batched_fft.h"
#include "thesis/memory_management.h"
#include "thesis/trlwe_function.h"
#include "thesis/torus_utility.h"

using namespace thesis;

const int numParty = 3;
const int N = 1024;
const int m = 6;
const int l = 64;
const double sdFresh = 1e-15;

bool is_file_exist(const char *fileName);
void save_data(const char *fileName, void *buffer, int sz);
void load_data(const char *fileName, void *buffer, int sz);

bool test_genkey(void *priv_key, void *pub_key);
bool genkey();

bool test_pre_expand(void *priv_key, void *pub_key, void *pre_expand);
bool pre_expand();

TEST(Mpc, Full) {
  ASSERT_TRUE(genkey());
  ASSERT_TRUE(pre_expand());
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
    TrlweFunction::getPlain(&fft, i & 1, pub + N * 2 * sizeof(TorusInteger) * i,
                            N, 1, plain + N * sizeof(TorusInteger) * i);
  fft.waitAllOut();
  // Round plain
  for (int i = 0; i < m; i++)
    TrlweFunction::roundPlain(plain + N * sizeof(TorusInteger) * i,
                              err + N * sizeof(double) * i, N);
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
    if (!chk)
      return false;
  }
  return true;
}
