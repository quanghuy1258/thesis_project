#include "gtest/gtest.h"

#include "thesis/declarations.h"
#include "thesis/load_lib.h"
#include "thesis/trgsw.h"
#include "thesis/trlwe.h"

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

TEST(Thesis, Decomposition) {
  std::srand(std::time(nullptr));
  thesis::Trgsw trgswObj;
  thesis::Trlwe trlweObj;

  trlweObj.clear_s();
  trlweObj.clear_ciphertexts();
  trlweObj.clear_plaintexts();
  trlweObj.generate_s();

  std::vector<thesis::PolynomialBinary> x;
  std::vector<std::vector<thesis::PolynomialInteger>> y;
  std::vector<std::vector<thesis::PolynomialTorus>> z;

  int numberTests = 100;
  x.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    x[i].resize(trlweObj.get_N());
    for (int j = 0; j < trlweObj.get_N(); j++) {
      x[i][j] = (std::rand() % 2 == 1);
    }
    trlweObj.addPlaintext(x[i]);
  }
  trlweObj.encryptAll();
  trlweObj.clear_plaintexts();
  ASSERT_TRUE(trgswObj.decompositeAll(y, trlweObj));
  trlweObj.get_ciphertexts(z);
  for (int cipherID = 0; cipherID < (signed)z.size(); cipherID++) {
    for (int i = 0; i <= trgswObj.get_k(); i++) {
      for (int j = 0; j < trgswObj.get_N(); j++) {
        thesis::Torus value = 0;
        for (int k = 0; k < trgswObj.get_l(); k++) {
          int shift =
              sizeof(thesis::Torus) * 8 - trgswObj.get_Bgbit() * (k + 1);
          unsigned ushift = (shift < 0) ? (-shift) : shift;
          value += (shift < 0)
                       ? (y[cipherID][i * trgswObj.get_l() + k][j] >> ushift)
                       : (y[cipherID][i * trgswObj.get_l() + k][j] << ushift);
        }
        thesis::Torus ori_value = z[cipherID][i][j];
        thesis::Torus comp_value =
            (ori_value > value) ? (ori_value - value) : (value - ori_value);
        thesis::Torus mask = 0;
        if ((signed)sizeof(thesis::Torus) * 8 >=
            trgswObj.get_Bgbit() * trgswObj.get_l() + 1) {
          mask = 1;
          mask <<= ((signed)sizeof(thesis::Torus) * 8 -
                    trgswObj.get_Bgbit() * trgswObj.get_l() - 1);
        }
        ASSERT_TRUE(comp_value <= mask);
      }
    }
  }
}
