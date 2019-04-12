#include "gtest/gtest.h"

#include "thesis/batched_fft.h"
#include "thesis/declarations.h"
#include "thesis/load_lib.h"

TEST(Thesis, BachedFFT) {
  std::srand(std::time(nullptr));
  std::unique_ptr<thesis::BatchedFFT> ptr =
      thesis::BatchedFFT::createInstance(8, 8, 8);
  std::vector<thesis::PolynomialTorus> torusInp(8);
  std::vector<thesis::PolynomialInteger> integerInp(8);
  thesis::PolynomialTorus result(8, 0), expect(8, 0);
  for (int i = 0; i < 8; i++) {
    torusInp[i].resize(8);
    integerInp[i].resize(8);
    for (int j = 0; j < 8; j++) {
      torusInp[i][j] = std::rand();
      integerInp[i][j] = std::rand();
    }
  }
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      for (int k = 0; k < 8; k++) {
        if (i + j < 8)
          expect[i + j] += integerInp[k][i] * torusInp[k][j];
        else
          expect[i + j - 8] -= integerInp[k][i] * torusInp[k][j];
      }
    }
  }
  {
    Eigen::Barrier notifier(8);
    for (int i = 0; i < 8; i++)
      ASSERT_TRUE(ptr->setIntegerInput(integerInp[i], i, &notifier));
    notifier.Wait();
  }
  ptr->doFFT();
  {
    Eigen::Barrier notifier(8);
    for (int i = 0; i < 8; i++)
      ASSERT_TRUE(ptr->copyTo(i, i + 8, &notifier));
    notifier.Wait();
  }
  {
    Eigen::Barrier notifier(8);
    for (int i = 0; i < 8; i++)
      ASSERT_TRUE(ptr->setTorusInput(torusInp[i], i, &notifier));
    notifier.Wait();
  }
  ptr->doFFT();
  for (int i = 0; i < 8; i++)
    ASSERT_TRUE(ptr->setMultiplicationPair(i, i + 8, i));
  ptr->doMultiplicationAndIFFT();
  for (int i = 0; i < 8; i++)
    ASSERT_TRUE(ptr->addOutput(result, i));
  for (int i = 0; i < 8; i++) {
    ASSERT_TRUE(result[i] == expect[i]);
  }
}
