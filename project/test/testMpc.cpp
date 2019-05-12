#include "gtest/gtest.h"

#include "mpc_application.h"

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
  // <<< Testing GenKey
  std::free(privKey);
  std::free(pubKey);
}
