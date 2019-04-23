#include "gtest/gtest.h"

#include "thesis/batched_fft.h"
#include "thesis/declarations.h"
#include "thesis/load_lib.h"
/*
TEST(Thesis, BachedFFT) {
  std::srand(std::time(nullptr));
  std::unique_ptr<thesis::BatchedFFT> ptr =
      thesis::BatchedFFT::createInstance(8, 5, 6);
  ASSERT_TRUE(ptr != nullptr);
  std::vector<thesis::PolynomialTorus> torusInp(3);
  std::vector<thesis::PolynomialInteger> integerInp(2);
  thesis::PolynomialTorus result(8), expect(8);
  for (int i = 0; i < 3; i++)
    torusInp[i].resize(8);
  for (int i = 0; i < 2; i++)
    integerInp[i].resize(8);
  int numberTests = 100;
  while (numberTests--) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 8; j++) {
        torusInp[i][j] = std::rand();
      }
    }
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 8; j++) {
        integerInp[i][j] = std::rand();
      }
    }
    for (int i = 0; i < 3; i++)
      ASSERT_TRUE(ptr->setTorusInp(torusInp[i], i));
    for (int i = 0; i < 2; i++)
      ASSERT_TRUE(ptr->setIntegerInp(integerInp[i], i + 3));
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 2; j++)
        ASSERT_TRUE(ptr->setMulPair(i, j + 3, i * 2 + j));
    }
    std::memset(result.data(), 0, sizeof(thesis::Torus) * 8);
    ASSERT_TRUE(ptr->addAllOut(result));
    std::memset(expect.data(), 0, sizeof(thesis::Torus) * 8);
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 64; k++) {
          int a = k / 8;
          int b = k % 8;
          if (a + b < 8)
            expect[a + b] += torusInp[i][a] * integerInp[j][b];
          else
            expect[a + b - 8] -= torusInp[i][a] * integerInp[j][b];
        }
      }
    }
    ptr->waitAll();
    for (int i = 0; i < 8; i++) {
      ASSERT_TRUE(result[i] == expect[i]);
    }
  }
}
*/
