#include "gtest/gtest.h"

#include "thesis/batched_fft.h"
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
    x[i] = std::rand() & 1;
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
      x[i][j] = std::rand() & 1;
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
      x[i][j] = std::rand() & 1;
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
    trlweCipherIds[i] = std::rand() % numberTests;
    trgswCipherIds[i] = (i & 1);
  }
  ASSERT_TRUE(trgswObj.externalProduct(trlweObj[1], trlweObj[0], trlweCipherIds,
                                       trgswCipherIds));
  trlweObj[1].decryptAll();

  expectedPlaintexts = x;
  for (int i = 0; i < numberTests; i++) {
    for (int j = 0; j < trgswObj.get_N(); j++) {
      int res = (i & 1) * ((x[trlweCipherIds[i]][j]) ? 1 : 0);
      int ori_res = (trlweObj[1].get_plaintexts()[i][j]) ? 1 : 0;
      ASSERT_TRUE(res == ori_res);
      expectedPlaintexts[i][j] = x[trlweCipherIds[i]][j] && (i & 1);
    }
    double maxStddevError =
        (trgswObj.get_k() + 1) * trgswObj.get_l() * trgswObj.get_N() *
            std::pow(2, trgswObj.get_Bgbit() - 1) *
            trgswObj.get_stddevErrors()[trgswCipherIds[i]] +
        (trgswObj.get_k() * trgswObj.get_N() + 1) *
            std::pow(2, -trgswObj.get_Bgbit() * trgswObj.get_l() - 1) +
        trlweObj[0].get_stddevErrors()[trlweCipherIds[i]];
    double maxVarError =
        (trgswObj.get_k() + 1) * trgswObj.get_l() * trgswObj.get_N() *
            std::pow(2, trgswObj.get_Bgbit() * 2 - 2) *
            trgswObj.get_varianceErrors()[trgswCipherIds[i]] +
        (trgswObj.get_k() * trgswObj.get_N() + 1) *
            std::pow(2, -trgswObj.get_Bgbit() * trgswObj.get_l() * 2 - 2) +
        trlweObj[0].get_varianceErrors()[trlweCipherIds[i]];
    ASSERT_TRUE(trlweObj[1].get_stddevErrors()[i] <= maxStddevError);
    ASSERT_TRUE(trlweObj[1].get_varianceErrors()[i] <= maxVarError);
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
  std::unique_ptr<thesis::BatchedFFT> ptr = thesis::BatchedFFT::createInstance(
      trgswObj.get_N(),
      (trgswObj.get_k() + 1) * trgswObj.get_l() * (trgswObj.get_k() + 2),
      (trgswObj.get_k() + 1) * trgswObj.get_l(), false);
  for (int i = 0; i < numberTests; i++) {
    int temp;
    ASSERT_TRUE(trgswObj._internalProduct(temp, (i << 1), (i << 1) + 1, ptr));
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
    double maxStddevError =
        (trgswObj.get_k() + 1) * trgswObj.get_l() * trgswObj.get_N() *
            std::pow(2, trgswObj.get_Bgbit() - 1) *
            trgswObj.get_stddevErrors()[idA] +
        (trgswObj.get_k() * trgswObj.get_N() + 1) *
            std::pow(2, -trgswObj.get_Bgbit() * trgswObj.get_l() - 1) +
        trgswObj.get_stddevErrors()[idB];
    double maxVarError =
        (trgswObj.get_k() + 1) * trgswObj.get_l() * trgswObj.get_N() *
            std::pow(2, trgswObj.get_Bgbit() * 2 - 2) *
            trgswObj.get_varianceErrors()[idA] +
        (trgswObj.get_k() * trgswObj.get_N() + 1) *
            std::pow(2, -trgswObj.get_Bgbit() * trgswObj.get_l() * 2 - 2) +
        trgswObj.get_varianceErrors()[idB];
    ASSERT_TRUE(trgswObj.get_stddevErrors()[(numberTests << 1) + i] <=
                maxStddevError);
    ASSERT_TRUE(trgswObj.get_varianceErrors()[(numberTests << 1) + i] <=
                maxVarError);
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
    double maxStddevError =
        std::max(trlweObj[0].get_stddevErrors()[trlweCipherTrueIds[i]],
                 trlweObj[0].get_stddevErrors()[trlweCipherFalseIds[i]]) +
        (trgswObj.get_k() + 1) * trgswObj.get_l() * trgswObj.get_N() *
            std::pow(2, trgswObj.get_Bgbit() - 1) *
            trgswObj.get_stddevErrors()[i] +
        (trgswObj.get_k() * trgswObj.get_N() + 1) *
            std::pow(2, -trgswObj.get_Bgbit() * trgswObj.get_l() - 1);
    double maxVarError =
        std::max(trlweObj[0].get_varianceErrors()[trlweCipherTrueIds[i]],
                 trlweObj[0].get_varianceErrors()[trlweCipherFalseIds[i]]) +
        (trgswObj.get_k() + 1) * trgswObj.get_l() * trgswObj.get_N() *
            std::pow(2, trgswObj.get_Bgbit() * 2 - 2) *
            trgswObj.get_varianceErrors()[i] +
        (trgswObj.get_k() * trgswObj.get_N() + 1) *
            std::pow(2, -trgswObj.get_Bgbit() * trgswObj.get_l() * 2 - 2);
    ASSERT_TRUE(trlweObj[1].get_stddevErrors()[i] <= maxStddevError);
    ASSERT_TRUE(trlweObj[1].get_varianceErrors()[i] <= maxVarError);
  }
  trlweObj[1].getAllErrorsForDebugging(errors, expectedPlaintexts);
  for (int i = 0; i < numberTests; i++) {
    ASSERT_TRUE(errors[i] < 0.25);
  }
}
/*
TEST(Thesis, BlindRotate) {
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
  std::vector<int> trlweCipherIds, coefficients, trgswCipherIds;

  int numberTests = 10;
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
  y.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    y[i] = ((std::rand() & 1) == 1);
    trgswObj.addPlaintext(y[i]);
  }
  trgswObj.encryptAll();
  trgswObj.clear_plaintexts();
  trlweCipherIds.resize(numberTests);
  coefficients.resize(numberTests + 1);
  trgswCipherIds.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    trlweCipherIds[i] = std::rand() % numberTests;
    coefficients[i] = std::rand() % (2 * trgswObj.get_N());
    trgswCipherIds[i] = std::rand() % numberTests;
  }
  coefficients[numberTests] = std::rand();
  ASSERT_TRUE(trgswObj.blindRotate(trlweObj[1], trlweObj[0], trlweCipherIds,
                                   coefficients, trgswCipherIds));
  trlweObj[1].decryptAll();

  int p = coefficients[numberTests];
  for (int i = 0; i < numberTests; i++) {
    p -= (y[trgswCipherIds[i]] ? coefficients[i] : 0);
  }
  p = ((p % (2 * trgswObj.get_N())) + 2 * trgswObj.get_N()) %
      (2 * trgswObj.get_N());
  expectedPlaintexts.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    expectedPlaintexts[i].resize(trgswObj.get_N());
    for (int j = 0; j < trgswObj.get_N(); j++) {
      if ((j + p >= trgswObj.get_N()) && (j + p < 2 * trgswObj.get_N())) {
        expectedPlaintexts[i][j] =
            -x[trlweCipherIds[i]][j + p - trgswObj.get_N()];
      } else {
        expectedPlaintexts[i][j] =
            x[trlweCipherIds[i]][(j + p) % trgswObj.get_N()];
      }
      ASSERT_TRUE(trlweObj[1].get_plaintexts()[i][j] ==
                  expectedPlaintexts[i][j]);
    }
    double maxStddevError =
        trlweObj[0].get_stddevErrors()[trlweCipherIds[i]] +
        ((trgswObj.get_k() + 1) * trgswObj.get_l() * trgswObj.get_N() *
             std::pow(2, trgswObj.get_Bgbit() - 1) *
             trgswObj.get_stddevErrors()[trgswCipherIds[i]] +
         (trgswObj.get_k() * trgswObj.get_N() + 1) *
             std::pow(2, -trgswObj.get_Bgbit() * trgswObj.get_l() - 1)) *
            numberTests;
    double maxVarError =
        trlweObj[0].get_varianceErrors()[trlweCipherIds[i]] +
        ((trgswObj.get_k() + 1) * trgswObj.get_l() * trgswObj.get_N() *
             std::pow(2, trgswObj.get_Bgbit() * 2 - 2) *
             trgswObj.get_varianceErrors()[trgswCipherIds[i]] +
         (trgswObj.get_k() * trgswObj.get_N() + 1) *
             std::pow(2, -trgswObj.get_Bgbit() * trgswObj.get_l() * 2 - 2)) *
            numberTests;
    ASSERT_TRUE(trlweObj[1].get_stddevErrors()[i] <= maxStddevError);
    ASSERT_TRUE(trlweObj[1].get_varianceErrors()[i] <= maxVarError);
  }
  trlweObj[1].getAllErrorsForDebugging(errors, expectedPlaintexts);
  for (int i = 0; i < numberTests; i++) {
    ASSERT_TRUE(errors[i] < 0.25);
  }
}

TEST(Thesis, BootstrapTLWE) {
  std::srand(std::time(nullptr));
  thesis::Trgsw trgswObj;
  thesis::Tlwe tlweObj[2];
  std::vector<int> trgswCipherIds;
  std::vector<double> errors;
  std::vector<bool> expectedPlaintexts;

  trgswObj.clear_s();
  trgswObj.clear_ciphertexts();
  trgswObj.clear_plaintexts();
  trgswObj.generate_s();

  tlweObj[0].clear_s();
  tlweObj[0].clear_ciphertexts();
  tlweObj[0].clear_plaintexts();
  tlweObj[0].set_n(5);
  tlweObj[0].generate_s();

  trgswCipherIds.resize(tlweObj[0].get_n());
  for (int i = 0; i < tlweObj[0].get_n(); i++) {
    trgswObj.addPlaintext(tlweObj[0].get_s()[tlweObj[0].get_n() - 1 - i]);
    trgswCipherIds[i] = tlweObj[0].get_n() - 1 - i;
  }
  trgswObj.encryptAll();
  tlweObj[0].addPlaintext(false);
  tlweObj[0].addPlaintext(true);
  tlweObj[0].encryptAll();

  int numberTests = 100;
  std::vector<thesis::Torus> constants(numberTests);
  for (int i = 0; i < numberTests; i++) {
    while (true) {
      constants[i] = std::rand() & 15;
      if (constants[i] != 4 && constants[i] != 12)
        break;
    }
    constants[i] <<= sizeof(thesis::Torus) * 8 - 4;
  }
  trgswObj.bootstrapTLWE(tlweObj[1], constants, tlweObj[0], 1, trgswCipherIds);
  tlweObj[1].decryptAll();

  double maxStddevError =
      tlweObj[0].get_n() * (trgswObj.get_k() * trgswObj.get_N() + 1) *
      std::pow(2, -trgswObj.get_Bgbit() * trgswObj.get_l() - 1);
  double maxVarError =
      tlweObj[0].get_n() * (trgswObj.get_k() * trgswObj.get_N() + 1) *
      std::pow(2, -trgswObj.get_Bgbit() * trgswObj.get_l() * 2 - 2);
  for (int i = 0; i < tlweObj[0].get_n(); i++) {
    maxStddevError += (trgswObj.get_k() + 1) * trgswObj.get_l() *
                      trgswObj.get_N() * std::pow(2, trgswObj.get_Bgbit() - 1) *
                      trgswObj.get_stddevErrors()[trgswCipherIds[i]];
    maxVarError += (trgswObj.get_k() + 1) * trgswObj.get_l() *
                   trgswObj.get_N() *
                   std::pow(2, trgswObj.get_Bgbit() * 2 - 2) *
                   trgswObj.get_varianceErrors()[trgswCipherIds[i]];
  }

  expectedPlaintexts.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    constants[i] >>= sizeof(thesis::Torus) * 8 - 4;
    constants[i] &= 15;
    if (constants[i] > 4 && constants[i] < 12) {
      expectedPlaintexts[i] = true;
    } else {
      expectedPlaintexts[i] = false;
    }
    ASSERT_TRUE(tlweObj[1].get_plaintexts()[i] == expectedPlaintexts[i]);
    ASSERT_TRUE(tlweObj[1].get_stddevErrors()[i] <= maxStddevError);
    ASSERT_TRUE(tlweObj[1].get_varianceErrors()[i] <= maxVarError);
  }
  tlweObj[1].getAllErrorsForDebugging(errors, expectedPlaintexts);
  for (int i = 0; i < numberTests; i++) {
    ASSERT_TRUE(errors[i] < 0.25);
  }
}

TEST(Thesis, GateBootstrap) {
  std::srand(std::time(nullptr));
  thesis::Trgsw trgswObj;
  thesis::Tlwe tlweObj[3];
  std::vector<int> trgswCipherIds;
  std::vector<double> errors;
  std::vector<bool> trgswObj_s, expectedPlaintexts;

  trgswObj.clear_s();
  trgswObj.clear_ciphertexts();
  trgswObj.clear_plaintexts();
  trgswObj.generate_s();
  trgswObj_s.resize(trgswObj.get_k() * trgswObj.get_N());
  for (int i = 0; i < trgswObj.get_k(); i++) {
    for (int j = 0; j < trgswObj.get_N(); j++) {
      trgswObj_s[i * trgswObj.get_N() + j] = trgswObj.get_s()[i][j];
    }
  }

  tlweObj[0].clear_s();
  tlweObj[0].clear_ciphertexts();
  tlweObj[0].clear_plaintexts();
  tlweObj[0].set_n(5);
  tlweObj[0].generate_s();

  int t = 16;
  tlweObj[2].set_n(5);
  tlweObj[2].set_s(tlweObj[0].get_s());
  tlweObj[2].initPublicKeySwitching(trgswObj_s, t);

  trgswCipherIds.resize(tlweObj[0].get_n());
  for (int i = 0; i < tlweObj[0].get_n(); i++) {
    trgswObj.addPlaintext(tlweObj[0].get_s()[tlweObj[0].get_n() - 1 - i]);
    trgswCipherIds[i] = tlweObj[0].get_n() - 1 - i;
  }
  trgswObj.encryptAll();
  tlweObj[0].addPlaintext(false);
  tlweObj[0].addPlaintext(true);
  tlweObj[0].encryptAll();

  int numberTests = 100;
  std::vector<thesis::Torus> constants(numberTests);
  for (int i = 0; i < numberTests; i++) {
    while (true) {
      constants[i] = std::rand() & 15;
      if (constants[i] != 4 && constants[i] != 12)
        break;
    }
    constants[i] <<= sizeof(thesis::Torus) * 8 - 4;
  }
  trgswObj.gateBootstrap(tlweObj[1], constants, tlweObj[0], 1, trgswCipherIds,
                         tlweObj[2], t);
  tlweObj[1].decryptAll();

  double maxStddevError =
      tlweObj[0].get_n() * (trgswObj.get_k() * trgswObj.get_N() + 1) *
          std::pow(2, -trgswObj.get_Bgbit() * trgswObj.get_l() - 1) +
      trgswObj.get_k() * trgswObj.get_N() * std::pow(2, -(t + 1));
  double maxVarError =
      tlweObj[0].get_n() * (trgswObj.get_k() * trgswObj.get_N() + 1) *
          std::pow(2, -trgswObj.get_Bgbit() * trgswObj.get_l() * 2 - 2) +
      trgswObj.get_k() * trgswObj.get_N() * std::pow(2, -(t + 1) * 2);
  for (int i = 0; i < tlweObj[0].get_n(); i++) {
    maxStddevError += (trgswObj.get_k() + 1) * trgswObj.get_l() *
                      trgswObj.get_N() * std::pow(2, trgswObj.get_Bgbit() - 1) *
                      trgswObj.get_stddevErrors()[trgswCipherIds[i]];
    maxVarError += (trgswObj.get_k() + 1) * trgswObj.get_l() *
                   trgswObj.get_N() *
                   std::pow(2, trgswObj.get_Bgbit() * 2 - 2) *
                   trgswObj.get_varianceErrors()[trgswCipherIds[i]];
  }
  for (int i = 0; i < trgswObj.get_k() * trgswObj.get_N(); i++) {
    for (int j = 0; j < t; j++) {
      maxStddevError += tlweObj[2].get_stddevErrors()[i * t + j];
      maxVarError += tlweObj[2].get_varianceErrors()[i * t + j];
    }
  }

  expectedPlaintexts.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    constants[i] >>= sizeof(thesis::Torus) * 8 - 4;
    constants[i] &= 15;
    if (constants[i] > 4 && constants[i] < 12) {
      expectedPlaintexts[i] = true;
    } else {
      expectedPlaintexts[i] = false;
    }
    ASSERT_TRUE(tlweObj[1].get_plaintexts()[i] == expectedPlaintexts[i]);
    ASSERT_TRUE(tlweObj[1].get_stddevErrors()[i] <= maxStddevError);
    ASSERT_TRUE(tlweObj[1].get_varianceErrors()[i] <= maxVarError);
  }
  tlweObj[1].getAllErrorsForDebugging(errors, expectedPlaintexts);
  for (int i = 0; i < numberTests; i++) {
    ASSERT_TRUE(errors[i] < 0.25);
  }
}
*/
