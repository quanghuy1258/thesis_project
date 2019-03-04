#include "gtest/gtest.h"

#include "thesis/declarations.h"
#include "thesis/load_lib.h"
#include "thesis/trlwe.h"

TEST(Thesis, TrlweEncryptDecrypt) {
  std::srand(std::time(nullptr));
  thesis::Trlwe trlweObj;

  trlweObj.clear_s();
  trlweObj.clear_ciphertexts();
  trlweObj.clear_plaintexts();
  trlweObj.generate_s();

  std::vector<thesis::PolynomialBinary> x, y;

  int numberTests = 100;
  x.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    x[i].resize(thesis::Trlwe::get_N());
    for (int j = 0; j < thesis::Trlwe::get_N(); j++) {
      x[i][j] = (std::rand() % 2 == 1);
    }
    trlweObj.addPlaintext(x[i]);
  }
  trlweObj.encryptAll();
  trlweObj.clear_plaintexts();
  trlweObj.decryptAll();
  trlweObj.get_plaintexts(y);
  for (int i = 0; i < numberTests; i++) {
    for (int j = 0; j < thesis::Trlwe::get_N(); j++) {
      EXPECT_TRUE(x[i][j] == y[i][j]);
    }
  }
}
