#include "gtest/gtest.h"

#include "thesis/declarations.h"
#include "thesis/fft.h"
#include "thesis/load_lib.h"

TEST(Thesis, FFT) {
  std::srand(std::time(nullptr));
  thesis::FFT fftObj;
  int deg = 8;
  ASSERT_TRUE(fftObj.set_N(deg));
  std::vector<thesis::Integer> polyA(deg);
  std::vector<thesis::Torus> polyB(deg), polyResult(deg), polyTest(deg);
  for (int i = 0; i < deg; i++) {
    polyA[i] = std::rand();
    polyB[i] = std::rand();
  }
  std::fill(polyTest.begin(), polyTest.end(), 0);
  for (int i = 0; i < deg; i++) {
    for (int j = 0; j < deg; j++) {
      if (i + j < deg) {
        polyTest[i + j] += polyA[i] * polyB[j];
      } else {
        polyTest[i + j - deg] -= polyA[i] * polyB[j];
      }
    }
  }
  ASSERT_TRUE(fftObj.torusPolynomialMultiplication(polyResult, polyA, polyB));
  for (int i = 0; i < deg; i++) {
    ASSERT_TRUE(polyResult[i] == polyTest[i]);
  }
}
