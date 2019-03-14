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
  for (int cipherID = 0; cipherID < numberTests; cipherID++) {
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

TEST(Thesis, ExternalProduct) {
  std::srand(std::time(nullptr));
  thesis::Trgsw trgswObj;
  thesis::Trlwe trlweObj[2];

  trgswObj.clear_s();
  trgswObj.clear_ciphertexts();
  trgswObj.clear_plaintexts();
  trgswObj.generate_s();
  trgswObj.setParamTo(trlweObj[0]);

  std::vector<thesis::PolynomialBinary> x, z;
  thesis::PolynomialInteger y;

  int numberTests = 100;
  x.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    x[i].resize(trlweObj[0].get_N());
    for (int j = 0; j < trlweObj[0].get_N(); j++) {
      x[i][j] = (std::rand() % 2 == 1);
    }
    trlweObj[0].addPlaintext(x[i]);
  }
  trlweObj[0].encryptAll();
  trlweObj[0].clear_plaintexts();

  y.resize(trgswObj.get_N());
  for (int i = 0; i < trgswObj.get_N(); i++) {
    y[i] = std::rand() & 1; // TRGSW Message Space: Z_N[X] -> B_N[X]
  }
  trgswObj.addPlaintext(y);
  trgswObj.encryptAll();
  trgswObj.clear_plaintexts();
  ASSERT_TRUE(trgswObj.externalProductAll(trlweObj[1], trlweObj[0], 0));
  trlweObj[1].decryptAll();
  trlweObj[1].get_plaintexts(z);

  for (int i = 0; i < numberTests; i++) {
    for (int j = 0; j < trgswObj.get_N(); j++) {
      int res = 0, a, b;
      for (int k = 0; k <= j; k++) {
        a = (x[i][k]) ? 1 : 0;
        b = y[j - k] & 1;
        res ^= a & b;
      }
      int ori_res = (z[i][j]) ? 1 : 0;
      ASSERT_TRUE(res == ori_res);
    }
  }
}
#ifdef ENABLE_TRGSW_INTERNAL_PRODUCT
TEST(Thesis, InternalProduct) {
  std::srand(std::time(nullptr));
  thesis::Trgsw trgswObj;

  trgswObj.clear_s();
  trgswObj.clear_ciphertexts();
  trgswObj.clear_plaintexts();
  trgswObj.generate_s();

  std::vector<thesis::PolynomialInteger> x, y;

  int numberTests = 10;
  x.resize(numberTests << 1);
  for (int i = 0; i < (numberTests << 1); i++) {
    x[i].resize(trgswObj.get_N());
    for (int j = 0; j < trgswObj.get_N(); j++) {
      x[i][j] = std::rand() & 1;
    }
    trgswObj.addPlaintext(x[i]);
  }
  trgswObj.encryptAll();
  trgswObj.clear_plaintexts();
  for (int i = 0; i < numberTests; i++) {
    int temp;
    trgswObj.internalProduct(temp, (i << 1), (i << 1) + 1);
  }
  trgswObj.decryptAll();
  trgswObj.get_plaintexts(y);

  for (int i = 0; i < numberTests; i++) {
    for (int j = 0; j < trgswObj.get_N(); j++) {
      thesis::Integer value = 0;
      for (int k = 0; k <= j; k++) {
        value += y[(i << 1)][k] * y[(i << 1) + 1][j - k];
      }
      ASSERT_TRUE(value == y[(numberTests << 1) + i][j]);
    }
  }
}
#endif
