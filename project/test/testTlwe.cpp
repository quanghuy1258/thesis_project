#include "gtest/gtest.h"

#include "thesis/load_lib.h"
#include "thesis/tlwe.h"

TEST(Thesis, TlweEncryptDecrypt) {
  std::srand(std::time(nullptr));
  thesis::Tlwe tlweObj;

  int numberTests = 100;
  while (numberTests--) {
    tlweObj.clear_s();
    tlweObj.clear_ciphertexts();
    tlweObj.clear_plaintexts();
    tlweObj.generate_s();
    int x = rand(), y = 0;
    std::vector<bool> yy;
    for (unsigned int i = 0; i < sizeof(int) * 8; i++) {
      tlweObj.addPlaintext((x >> i) & 1);
    }
    tlweObj.encryptAll();
    tlweObj.clear_plaintexts();
    tlweObj.decryptAll();
    tlweObj.get_plaintexts(yy);
    for (unsigned int i = 0; i < sizeof(int) * 8; i++) {
      int temp = (yy[i]) ? 1 : 0;
      temp = temp << i;
      y = y | temp;
    }
    ASSERT_TRUE(x == y);
  }
}
