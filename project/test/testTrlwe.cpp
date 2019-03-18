#include "gtest/gtest.h"

#include "thesis/declarations.h"
#include "thesis/load_lib.h"
#include "thesis/tlwe.h"
#include "thesis/trlwe.h"

TEST(Thesis, TrlweEncryptDecrypt) {
  std::srand(std::time(nullptr));
  thesis::Trlwe trlweObj;

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
      x[i][j] = (std::rand() % 2 == 1);
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
}

TEST(Thesis, TrlweExtractAllToTlwe) {
  std::srand(std::time(nullptr));
  thesis::Trlwe trlweObj;
  thesis::Tlwe tlweObj;

  trlweObj.clear_s();
  trlweObj.clear_ciphertexts();
  trlweObj.clear_plaintexts();

  trlweObj.generate_s();
  tlweObj.set_n(trlweObj.get_N() * trlweObj.get_k(), true);
  std::vector<thesis::Integer> sTlwe(trlweObj.get_N() * trlweObj.get_k());
  for (int i = 0; i < trlweObj.get_N() * trlweObj.get_k(); i++) {
    sTlwe[i] =
        (trlweObj.get_s()[i / trlweObj.get_N()][i % trlweObj.get_N()]) ? 1 : 0;
  }
  tlweObj.set_s(sTlwe);

  std::vector<thesis::PolynomialBinary> x;
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
  trlweObj.clear_s();
  trlweObj.clear_plaintexts();
  trlweObj.tlweExtractAll(tlweObj);
  tlweObj.decryptAll();
  std::cout << "Number of plaintexts: " << tlweObj.get_plaintexts().size()
            << std::endl;
  for (int i = 0; i < numberTests; i++) {
    for (int j = 0; j < trlweObj.get_N(); j++) {
      ASSERT_TRUE(x[i][j] ==
                  tlweObj.get_plaintexts()[i * trlweObj.get_N() + j]);
    }
  }
}

TEST(Thesis, TrlweExtractOneToTlwe) {
  std::srand(std::time(nullptr));
  thesis::Trlwe trlweObj;
  thesis::Tlwe tlweObj;

  trlweObj.clear_s();
  trlweObj.clear_ciphertexts();
  trlweObj.clear_plaintexts();

  trlweObj.generate_s();
  tlweObj.set_n(trlweObj.get_N() * trlweObj.get_k(), true);
  std::vector<thesis::Integer> sTlwe(trlweObj.get_N() * trlweObj.get_k());
  for (int i = 0; i < trlweObj.get_N() * trlweObj.get_k(); i++) {
    sTlwe[i] =
        (trlweObj.get_s()[i / trlweObj.get_N()][i % trlweObj.get_N()]) ? 1 : 0;
  }
  tlweObj.set_s(sTlwe);

  std::vector<thesis::PolynomialBinary> x;
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
  trlweObj.clear_s();
  trlweObj.clear_plaintexts();
  int p = std::rand() % trlweObj.get_N();
  for (int i = 0; i < numberTests; i++) {
    trlweObj.tlweExtractOne(tlweObj, p, i);
  }
  tlweObj.decryptAll();
  std::cout << "Number of plaintexts: " << tlweObj.get_plaintexts().size()
            << std::endl;
  for (int i = 0; i < numberTests; i++) {
    ASSERT_TRUE(x[i][p] == tlweObj.get_plaintexts()[i]);
  }
}
