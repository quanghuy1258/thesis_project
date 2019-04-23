#include "gtest/gtest.h"

#include "thesis/batched_fft.h"
#include "thesis/declarations.h"
#include "thesis/load_lib.h"
#include "thesis/memory_management.h"

TEST(Thesis, BachedFFT) {
  std::srand(std::time(nullptr));
  thesis::BatchedFFT fft(8, 2, 3);
  thesis::TorusInteger *testInp =
      (thesis::TorusInteger *)thesis::MemoryManagement::mallocMM(
          8 * (2 + 1) * 3 * sizeof(thesis::TorusInteger));
  thesis::TorusInteger *testOut =
      (thesis::TorusInteger *)thesis::MemoryManagement::mallocMM(
          8 * 2 * sizeof(thesis::TorusInteger));
  std::vector<thesis::TorusInteger> vecInp(8 * (2 + 1) * 3);
  std::vector<thesis::TorusInteger> calOut(8 * 2), expOut(8 * 2);
  int numberTests = 100;
  while (numberTests--) {
    for (int i = 0; i < 8 * (2 + 1) * 3; i++)
      vecInp[i] = std::rand();
    for (int i = 0; i < 8 * 2; i++)
      calOut[i] = expOut[i] = std::rand();
    thesis::MemoryManagement::memcpyMM_h2d(
        testInp, vecInp.data(), 8 * (2 + 1) * 3 * sizeof(thesis::TorusInteger));
    thesis::MemoryManagement::memcpyMM_h2d(
        testOut, calOut.data(), 8 * 2 * sizeof(thesis::TorusInteger));
    for (int i = 0; i < 3; i++)
      fft.setInp(testInp + 8 * (2 * 3 + i), i);
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++)
        fft.setInp(testInp + 8 * (i * 3 + j), i, j);
    }
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++)
        fft.setMul(i, j);
    }
    fft.addAllOut(testOut, 0);
    fft.subAllOut(testOut + 8, 1);
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 64; j++) {
        int a = j / 8;
        int b = j % 8;
        if (a + b < 8)
          expOut[a + b] += vecInp[i * 8 + a] * vecInp[(i + 6) * 8 + b];
        else
          expOut[a + b - 8] -= vecInp[i * 8 + a] * vecInp[(i + 6) * 8 + b];
      }
    }
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 64; j++) {
        int a = j / 8;
        int b = j % 8;
        if (a + b < 8)
          expOut[8 + a + b] -=
              vecInp[(i + 3) * 8 + a] * vecInp[(i + 6) * 8 + b];
        else
          expOut[8 + a + b - 8] +=
              vecInp[(i + 3) * 8 + a] * vecInp[(i + 6) * 8 + b];
      }
    }
    fft.waitAllOut();
    thesis::MemoryManagement::memcpyMM_d2h(
        calOut.data(), testOut, 8 * 2 * sizeof(thesis::TorusInteger));
    for (int i = 0; i < 8 * 2; i++)
      ASSERT_TRUE(expOut[i] == calOut[i]);
  }
  thesis::MemoryManagement::freeMM(testInp);
  thesis::MemoryManagement::freeMM(testOut);
}
