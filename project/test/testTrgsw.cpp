#include "gtest/gtest.h"

#include "thesis/declarations.h"
#include "thesis/load_lib.h"
#include "thesis/trgsw.h"

TEST(Thesis, TrgswEncryptDecrypt) {
  std::srand(std::time(nullptr));
  thesis::Trgsw trgswObj;

  trgswObj.clear_s();
  trgswObj.clear_ciphertexts();
  trgswObj.clear_plaintexts();
  trgswObj.generate_s();

  std::vector<thesis::PolynomialInteger> x, y;

  int numberTests = 100;
  x.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    x[i].resize(trgswObj.get_N());
    for (int j = 0; j < trgswObj.get_N(); j++) {
      int bits = trgswObj.get_Bgbit() * trgswObj.get_l();
      x[i][j] = std::rand();
      if ((signed)sizeof(thesis::Integer) * 8 > bits) {
        uint64_t mask = 1;
        mask = mask << bits;
        mask = mask - 1;
        x[i][j] = x[i][j] & mask;
      }
    }
    trgswObj.addPlaintext(x[i]);
  }
  trgswObj.encryptAll();
  trgswObj.clear_plaintexts();
  trgswObj.decryptAll();
  trgswObj.get_plaintexts(y);
  for (int i = 0; i < numberTests; i++) {
    for (int j = 0; j < trgswObj.get_N(); j++) {
      ASSERT_TRUE(x[i][j] == y[i][j]);
    }
  }
}
