#include "gtest/gtest.h"

#include "mpc_application.h"
#include "thesis/batched_fft.h"
#include "thesis/memory_management.h"
#include "thesis/trlwe_function.h"

using namespace thesis;

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
      (TorusInteger *)MemoryManagement::mallocMM(1024 * sizeof(TorusInteger));
  MemoryManagement::memcpyMM_h2d(priv, priv_key, 1024 * sizeof(TorusInteger));
  // Public key
  TorusInteger *pub = (TorusInteger *)MemoryManagement::mallocMM(
      6 * 1024 * 2 * sizeof(TorusInteger));
  MemoryManagement::memcpyMM_h2d(pub, pub_key,
                                 6 * 1024 * 2 * sizeof(TorusInteger));
  // Plaintext
  TorusInteger *plain = (TorusInteger *)MemoryManagement::mallocMM(
      6 * 1024 * sizeof(TorusInteger));
  MemoryManagement::memsetMM(plain, 0, 6 * 1024 * sizeof(TorusInteger));
  // Error
  double *err = (double *)MemoryManagement::mallocMM(6 * 1024 * sizeof(double));
  MemoryManagement::memsetMM(err, 0, 6 * 1024 * sizeof(double));
  // FFT
  BatchedFFT fft(1024, 2, 1);
  fft.setInp(priv, 0);
  // Get plain
  for (int i = 0; i < 6; i++)
    TrlweFunction::getPlain(&fft, i & 1,
                            pub + 1024 * 2 * sizeof(TorusInteger) * i, 1024, 1,
                            plain + 1024 * sizeof(TorusInteger) * i);
  fft.waitAllOut();
  // Round plain
  for (int i = 0; i < 6; i++)
    TrlweFunction::roundPlain(plain + 1024 * sizeof(TorusInteger) * i,
                              err + 1024 * sizeof(double) * i, 1024);
  // Check plain + error
  TorusInteger *hPlain = new TorusInteger[6 * 1024];
  double *hErr = new double[6 * 1024];
  MemoryManagement::memcpyMM_d2h(hPlain, plain,
                                 6 * 1024 * sizeof(TorusInteger));
  MemoryManagement::memcpyMM_d2h(hErr, err, 6 * 1024 * sizeof(double));
  double avgErr = 0;
  bool chk = true;
  for (int i = 0; i < 6 * 1024; i++) {
    if (hPlain[i] != 0)
      chk = false;
    if (hErr[i] > 0.125)
      chk = false;
    avgErr += hErr[i];
  }
  std::cout << avgErr / (6 * 1024) << std::endl;
  // Free all
  MemoryManagement::freeMM(priv);
  MemoryManagement::freeMM(pub);
  MemoryManagement::freeMM(plain);
  MemoryManagement::freeMM(err);
  delete[] hPlain;
  delete[] hErr;
  return chk;
}
// Generate key
TEST(Mpc, GenKey_0_3) {
  if (is_file_exist("PrivKey_0_3") && is_file_exist("PubKey_0_3"))
    return;
  MpcApplication mpcObj(2, 0, 1024, 6, 64, 1e-15);
  mpcObj.createPrivkey();
  mpcObj.createPubkey();
  void *privKey = std::malloc(mpcObj.getSizePrivkey());
  void *pubKey = std::malloc(mpcObj.getSizePubkey());
  mpcObj.exportPrivkey(privKey);
  mpcObj.exportPubkey(pubKey);
  save_data("PrivKey_0_3", privKey, mpcObj.getSizePrivkey());
  save_data("PubKey_0_3", pubKey, mpcObj.getSizePrivkey());
  // >>> Testing GenKey
  EXPECT_TRUE(test_genkey(privKey, pubKey));
  // <<< Testing GenKey
  std::free(privKey);
  std::free(pubKey);
}
TEST(Mpc, GenKey_1_3) {
  if (is_file_exist("PrivKey_1_3") && is_file_exist("PubKey_1_3"))
    return;
  MpcApplication mpcObj(2, 0, 1024, 6, 64, 1e-15);
  mpcObj.createPrivkey();
  mpcObj.createPubkey();
  void *privKey = std::malloc(mpcObj.getSizePrivkey());
  void *pubKey = std::malloc(mpcObj.getSizePubkey());
  mpcObj.exportPrivkey(privKey);
  mpcObj.exportPubkey(pubKey);
  save_data("PrivKey_1_3", privKey, mpcObj.getSizePrivkey());
  save_data("PubKey_1_3", pubKey, mpcObj.getSizePrivkey());
  // >>> Testing GenKey
  EXPECT_TRUE(test_genkey(privKey, pubKey));
  // <<< Testing GenKey
  std::free(privKey);
  std::free(pubKey);
}
TEST(Mpc, GenKey_2_3) {
  if (is_file_exist("PrivKey_2_3") && is_file_exist("PubKey_2_3"))
    return;
  MpcApplication mpcObj(2, 0, 1024, 6, 64, 1e-15);
  mpcObj.createPrivkey();
  mpcObj.createPubkey();
  void *privKey = std::malloc(mpcObj.getSizePrivkey());
  void *pubKey = std::malloc(mpcObj.getSizePubkey());
  mpcObj.exportPrivkey(privKey);
  mpcObj.exportPubkey(pubKey);
  save_data("PrivKey_2_3", privKey, mpcObj.getSizePrivkey());
  save_data("PubKey_2_3", pubKey, mpcObj.getSizePrivkey());
  // >>> Testing GenKey
  EXPECT_TRUE(test_genkey(privKey, pubKey));
  // <<< Testing GenKey
  std::free(privKey);
  std::free(pubKey);
}
// PreExpand
TEST(Mpc, PreExpand_0_3) {
  ASSERT_TRUE(is_file_exist("PrivKey_0_3") && is_file_exist("PubKey_1_3") &&
              is_file_exist("PubKey_2_3"));
}
