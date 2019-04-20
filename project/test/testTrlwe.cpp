#include "gtest/gtest.h"

#include "thesis/declarations.h"
#include "thesis/load_lib.h"
#include "thesis/tlwe.h"
#include "thesis/trlwe.h"

TEST(Thesis, TrlweEncryptDecrypt) {
  std::srand(std::time(nullptr));
  thesis::Trlwe trlweObj;
  std::vector<double> errors;

  trlweObj.clear_s();
  trlweObj.clear_ciphertexts();
  trlweObj.clear_plaintexts();
  trlweObj.generate_s();

  std::vector<thesis::PolynomialBinary> x;

  int numberTests = 100;
  x.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    x[i].resize(trlweObj.get_N());
    for (int j = 0; j < trlweObj.get_N(); j++) {
      x[i][j] = std::rand() & 1;
    }
    trlweObj.addPlaintext(x[i]);
  }
  trlweObj.encryptAll();
  trlweObj.clear_plaintexts();
  trlweObj.decryptAll();
  for (int i = 0; i < numberTests; i++) {
    for (int j = 0; j < trlweObj.get_N(); j++) {
      ASSERT_TRUE(x[i][j] == trlweObj.get_plaintexts()[i][j]);
    }
  }
  trlweObj.getAllErrorsForDebugging(errors, x);
  for (int i = 0; i < numberTests; i++) {
    ASSERT_TRUE(errors[i] < 0.25);
  }
}

TEST(Thesis, TrlweExtractAllToTlwe) {
  std::srand(std::time(nullptr));
  thesis::Trlwe trlweObj;
  thesis::Tlwe tlweObj;
  std::vector<double> errors;
  std::vector<bool> expectedPlaintexts;

  trlweObj.clear_s();
  trlweObj.clear_ciphertexts();
  trlweObj.clear_plaintexts();

  trlweObj.generate_s();
  std::vector<thesis::PolynomialBinary> x;
  int numberTests = 100;
  x.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    x[i].resize(trlweObj.get_N());
    for (int j = 0; j < trlweObj.get_N(); j++) {
      x[i][j] = std::rand() & 1;
    }
    trlweObj.addPlaintext(x[i]);
  }
  trlweObj.encryptAll();
  trlweObj.clear_plaintexts();
  trlweObj.tlweExtractAll(tlweObj);
  tlweObj.decryptAll();
  std::cout << "Number of plaintexts: " << tlweObj.get_plaintexts().size()
            << std::endl;
  expectedPlaintexts.resize(numberTests * trlweObj.get_N());
  for (int i = 0; i < numberTests; i++) {
    for (int j = 0; j < trlweObj.get_N(); j++) {
      ASSERT_TRUE(x[i][j] ==
                  tlweObj.get_plaintexts()[i * trlweObj.get_N() + j]);
      expectedPlaintexts[i * trlweObj.get_N() + j] = x[i][j];
    }
  }
  tlweObj.getAllErrorsForDebugging(errors, expectedPlaintexts);
  for (int i = 0; i < numberTests * trlweObj.get_N(); i++) {
    ASSERT_TRUE(errors[i] < 0.25);
  }
}

TEST(Thesis, TrlweExtractToTlwe) {
  std::srand(std::time(nullptr));
  thesis::Trlwe trlweObj;
  thesis::Tlwe tlweObj;
  std::vector<double> errors;
  std::vector<bool> expectedPlaintexts;

  trlweObj.clear_s();
  trlweObj.clear_ciphertexts();
  trlweObj.clear_plaintexts();
  trlweObj.generate_s();

  std::vector<thesis::PolynomialBinary> x;
  int numberTests = 100;
  x.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    x[i].resize(trlweObj.get_N());
    for (int j = 0; j < trlweObj.get_N(); j++) {
      x[i][j] = std::rand() & 1;
    }
    trlweObj.addPlaintext(x[i]);
  }
  trlweObj.encryptAll();
  trlweObj.clear_plaintexts();
  std::vector<int> ps(numberTests), cipherIDs(numberTests);
  for (int i = 0; i < numberTests; i++) {
    ps[i] = std::rand() % trlweObj.get_N();
    cipherIDs[i] = std::rand() % numberTests;
  }
  trlweObj.tlweExtract(tlweObj, ps, cipherIDs);
  tlweObj.decryptAll();
  std::cout << "Number of plaintexts: " << tlweObj.get_plaintexts().size()
            << std::endl;
  expectedPlaintexts.resize(numberTests);
  for (int i = 0; i < numberTests; i++) {
    ASSERT_TRUE(x[cipherIDs[i]][ps[i]] == tlweObj.get_plaintexts()[i]);
    expectedPlaintexts[i] = x[cipherIDs[i]][ps[i]];
  }
  tlweObj.getAllErrorsForDebugging(errors, expectedPlaintexts);
  for (int i = 0; i < numberTests; i++) {
    ASSERT_TRUE(errors[i] < 0.25);
  }
}
