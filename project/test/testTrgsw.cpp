#include "gtest/gtest.h"

#include "thesis/declarations.h"
#include "thesis/load_lib.h"
#include "thesis/trgsw.h"
#include "thesis/trlwe.h"

TEST(Thesis, TrgswEncryptDecrypt) {
  std::srand(std::time(nullptr));
  thesis::Trgsw trgswObj;
  std::vector<double> errors;

  trgswObj.clear_s();
  trgswObj.clear_ciphertexts();
  trgswObj.clear_plaintexts();
  trgswObj.generate_s();

  std::vector<bool> x;

  int numberTests = 100;
  x.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    x[i] = ((std::rand() & 1) == 1);
    trgswObj.addPlaintext(x[i]);
  }
  trgswObj.encryptAll();
  trgswObj.clear_plaintexts();
  trgswObj.decryptAll();
  for (int i = 0; i < numberTests; i++) {
    ASSERT_TRUE(x[i] == trgswObj.get_plaintexts()[i]);
  }
  trgswObj.getAllErrorsForDebugging(errors, x);
  for (int i = 0; i < numberTests; i++) {
    ASSERT_TRUE(errors[i] < std::pow(2, -trgswObj.get_Bgbit() - 1));
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

  int numberTests = 100;
  x.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    x[i].resize(trgswObj.get_N());
    for (int j = 0; j < trgswObj.get_N(); j++) {
      x[i][j] = (std::rand() % 2 == 1);
    }
    trlweObj.addPlaintext(x[i]);
  }
  trlweObj.encryptAll();
  trlweObj.clear_plaintexts();
  ASSERT_TRUE(trgswObj.decompositeAll(y, trlweObj));
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
        thesis::Torus ori_value = trlweObj.get_ciphertexts()[cipherID][i][j];
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
  std::vector<double> errors;

  trgswObj.clear_s();
  trgswObj.clear_ciphertexts();
  trgswObj.clear_plaintexts();
  trgswObj.generate_s();
  trgswObj.setParamTo(trlweObj[0]);

  std::vector<thesis::PolynomialBinary> x, expectedPlaintexts;

  int numberTests = 100;
  x.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    x[i].resize(trgswObj.get_N());
    for (int j = 0; j < trgswObj.get_N(); j++) {
      x[i][j] = (std::rand() % 2 == 1);
    }
    trlweObj[0].addPlaintext(x[i]);
  }
  trlweObj[0].encryptAll();
  trlweObj[0].clear_plaintexts();

  trgswObj.addPlaintext(false);
  trgswObj.addPlaintext(true);
  trgswObj.encryptAll();
  trgswObj.clear_plaintexts();
  std::vector<int> trlweCipherIds(numberTests), trgswCipherIds(numberTests);
  for (int i = 0; i < numberTests; i++) {
    trlweCipherIds[i] = i;
    trgswCipherIds[i] = (i & 1);
  }
  ASSERT_TRUE(trgswObj.externalProduct(trlweObj[1], trlweObj[0], trlweCipherIds,
                                       trgswCipherIds));
  trlweObj[1].decryptAll();

  expectedPlaintexts = x;
  for (int i = 0; i < numberTests; i++) {
    for (int j = 0; j < trgswObj.get_N(); j++) {
      int res = (i & 1) * ((x[i][j]) ? 1 : 0);
      int ori_res = (trlweObj[1].get_plaintexts()[i][j]) ? 1 : 0;
      ASSERT_TRUE(res == ori_res);
      expectedPlaintexts[i][j] = x[i][j] && ((i & 1) == 1);
    }
    double maxError =
        (trgswObj.get_k() + 1) * trgswObj.get_l() * trgswObj.get_N() *
            std::pow(2, trgswObj.get_Bgbit() - 1) *
            trgswObj.get_stddevErrors()[trgswCipherIds[i]] +
        (trgswObj.get_k() * trgswObj.get_N() + 1) *
            std::pow(2, -trgswObj.get_Bgbit() * trgswObj.get_l() - 1) +
        trlweObj[0].get_stddevErrors()[trlweCipherIds[i]];
    ASSERT_TRUE(trlweObj[1].get_stddevErrors()[i] <= maxError);
  }
  trlweObj[1].getAllErrorsForDebugging(errors, expectedPlaintexts);
  for (int i = 0; i < numberTests; i++) {
    ASSERT_TRUE(errors[i] < 0.25);
  }
}

TEST(Thesis, InternalProduct) {
  std::srand(std::time(nullptr));
  thesis::Trgsw trgswObj;
  std::vector<double> errors;

  trgswObj.clear_s();
  trgswObj.clear_ciphertexts();
  trgswObj.clear_plaintexts();
  trgswObj.generate_s();

  std::vector<bool> x, expectedPlaintexts;

  int numberTests = 10;
  x.resize(numberTests << 1);
  for (int i = 0; i < (numberTests << 1); i++) {
    x[i] = ((std::rand() & 1) == 1);
    trgswObj.addPlaintext(x[i]);
  }
  trgswObj.encryptAll();
  trgswObj.clear_plaintexts();
  for (int i = 0; i < numberTests; i++) {
    int temp;
    trgswObj.internalProduct(temp, (i << 1), (i << 1) + 1);
  }
  trgswObj.decryptAll();

  expectedPlaintexts.resize(numberTests * 3);
  for (int i = 0; i < numberTests; i++) {
    expectedPlaintexts[(i << 1)] = x[(i << 1)];
    expectedPlaintexts[(i << 1) + 1] = x[(i << 1) + 1];
    expectedPlaintexts[(numberTests << 1) + i] = x[(i << 1)] && x[(i << 1) + 1];
    ASSERT_TRUE(trgswObj.get_plaintexts()[(i << 1)] == x[(i << 1)]);
    ASSERT_TRUE(trgswObj.get_plaintexts()[(i << 1) + 1] == x[(i << 1) + 1]);
    ASSERT_TRUE((trgswObj.get_plaintexts()[(i << 1)] &&
                 trgswObj.get_plaintexts()[(i << 1) + 1]) ==
                trgswObj.get_plaintexts()[(numberTests << 1) + i]);
    int idA = (i << 1);
    int idB = (i << 1) + 1;
    if (trgswObj.get_stddevErrors()[idA] > trgswObj.get_stddevErrors()[idB]) {
      std::swap(idA, idB);
    }
    double maxError =
        (trgswObj.get_k() + 1) * trgswObj.get_l() * trgswObj.get_N() *
            std::pow(2, trgswObj.get_Bgbit() - 1) *
            trgswObj.get_stddevErrors()[idA] +
        (trgswObj.get_k() * trgswObj.get_N() + 1) *
            std::pow(2, -trgswObj.get_Bgbit() * trgswObj.get_l() - 1) +
        trgswObj.get_stddevErrors()[idB];
    ASSERT_TRUE(trgswObj.get_stddevErrors()[(numberTests << 1) + i] <=
                maxError);
  }
  trgswObj.getAllErrorsForDebugging(errors, expectedPlaintexts);
  for (int i = 0; i < numberTests * 3; i++) {
    ASSERT_TRUE(errors[i] < std::pow(2, -trgswObj.get_Bgbit() - 1));
  }
}

TEST(Thesis, CMux) {
  std::srand(std::time(nullptr));
  thesis::Trgsw trgswObj;
  thesis::Trlwe trlweObj[2];
  std::vector<double> errors;

  trgswObj.clear_s();
  trgswObj.clear_ciphertexts();
  trgswObj.clear_plaintexts();
  trgswObj.generate_s();
  trgswObj.setParamTo(trlweObj[0]);

  std::vector<thesis::PolynomialBinary> x, expectedPlaintexts;
  std::vector<bool> y;
  std::vector<int> trlweCipherTrueIds, trlweCipherFalseIds, trgswCipherIds;

  int numberTests = 100;
  x.resize(numberTests << 1);
  for (int i = 0; i < (numberTests << 1); i++) {
    x[i].resize(trgswObj.get_N());
    for (int j = 0; j < trgswObj.get_N(); j++) {
      x[i][j] = (std::rand() % 2 == 1);
    }
    trlweObj[0].addPlaintext(x[i]);
  }
  trlweObj[0].encryptAll();
  trlweObj[0].clear_plaintexts();
  y.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    y[i] = ((std::rand() & 1) == 1);
    trgswObj.addPlaintext(y[i]);
  }
  trgswObj.encryptAll();
  trgswObj.clear_plaintexts();
  trlweCipherTrueIds.resize(numberTests);
  trlweCipherFalseIds.resize(numberTests);
  trgswCipherIds.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    trlweCipherTrueIds[i] = (i << 1);
    trlweCipherFalseIds[i] = (i << 1) + 1;
    trgswCipherIds[i] = i;
  }
  ASSERT_TRUE(trgswObj.cMux(trlweObj[1], trlweObj[0], trlweCipherTrueIds,
                            trlweCipherFalseIds, trgswCipherIds));
  trlweObj[1].decryptAll();

  expectedPlaintexts.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    bool check = y[i];
    expectedPlaintexts[i] =
        ((check) ? x[trlweCipherTrueIds[i]] : x[trlweCipherFalseIds[i]]);
    for (int j = 0; j < trgswObj.get_N(); j++) {
      ASSERT_TRUE(trlweObj[1].get_plaintexts()[i][j] ==
                  ((check) ? x[trlweCipherTrueIds[i]][j]
                           : x[trlweCipherFalseIds[i]][j]));
    }
    double maxError =
        std::max(trlweObj[0].get_stddevErrors()[trlweCipherTrueIds[i]],
                 trlweObj[0].get_stddevErrors()[trlweCipherFalseIds[i]]) +
        (trgswObj.get_k() + 1) * trgswObj.get_l() * trgswObj.get_N() *
            std::pow(2, trgswObj.get_Bgbit() - 1) *
            trgswObj.get_stddevErrors()[i] +
        (trgswObj.get_k() * trgswObj.get_N() + 1) *
            std::pow(2, -trgswObj.get_Bgbit() * trgswObj.get_l() - 1);
    ASSERT_TRUE(trlweObj[1].get_stddevErrors()[i] <= maxError);
  }
  trlweObj[1].getAllErrorsForDebugging(errors, expectedPlaintexts);
  for (int i = 0; i < numberTests; i++) {
    ASSERT_TRUE(errors[i] < 0.25);
  }
}
