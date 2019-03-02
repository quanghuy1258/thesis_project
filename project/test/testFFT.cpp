#include "gtest/gtest.h"

#include "thesis/declarations.h"
#include "thesis/fft.h"
#include "thesis/load_lib.h"

TEST(Thesis, FFT) {
  std::srand(std::time(nullptr));
  thesis::FFT fftObj;
  ASSERT_TRUE(fftObj.set_N(8));
  std::vector<thesis::Integer> polyA(8);
  std::vector<thesis::Torus> polyB(8), polyResult(8);
  for (int i = 0; i < 8; i++) {
    polyA[i] = std::rand();
    polyB[i] = std::rand();
  }
  EXPECT_TRUE(fftObj.torusPolynomialMultiplication(polyResult, polyA, polyB));
  for (int i = 0; i < 8; i++) {
    thesis::Torus temp = 0;
    for (int j = 0; j <= i; j++) {
      temp += polyA[i - j] * polyB[j];
    }
    EXPECT_TRUE(polyResult[i] == temp);
  }
}
