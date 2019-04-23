#include "gtest/gtest.h"

#include "thesis/load_lib.h"
#include "thesis/tlwe.h"
/*
TEST(Thesis, TlweEncryptDecrypt) {
  std::srand(std::time(nullptr));
  thesis::Tlwe tlweObj;
  std::vector<double> errors;
  std::vector<bool> expectedPlaintexts;

  int numberTests = 100;
  while (numberTests--) {
    tlweObj.clear_s();
    tlweObj.clear_ciphertexts();
    tlweObj.clear_plaintexts();
    tlweObj.generate_s();
    int x = rand(), y = 0;
    for (unsigned int i = 0; i < sizeof(int) * 8; i++) {
      tlweObj.addPlaintext((x >> i) & 1);
    }
    tlweObj.encryptAll();
    expectedPlaintexts = tlweObj.get_plaintexts();
    tlweObj.clear_plaintexts();
    tlweObj.decryptAll();
    for (unsigned int i = 0; i < sizeof(int) * 8; i++) {
      int temp = (tlweObj.get_plaintexts()[i]) ? 1 : 0;
      temp = temp << i;
      y = y | temp;
    }
    ASSERT_TRUE(x == y);
    tlweObj.getAllErrorsForDebugging(errors, expectedPlaintexts);
    for (unsigned int i = 0; i < sizeof(int) * 8; i++) {
      ASSERT_TRUE(errors[i] < 0.25);
    }
  }
}
*/
