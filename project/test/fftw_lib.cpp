#include "thesis/declarations.h"
#include "thesis/load_lib.h"
#include "gtest/gtest.h"
#include <fftw3.h>

TEST(FFTW, TestLibrary) {
  try {
    std::default_random_engine generator(
        std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    int N = 100;

    fftw_complex *in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int i = 0; i < N; i++) {
      in[i][0] = distribution(generator);
      in[i][1] = distribution(generator);
    }
    fftw_execute(p);
    for (int i = 0; i < N; i++) {
      double exp[2] = {0, 0};
      for (int j = 0; j < N; j++) {
        exp[0] += in[j][0] * std::cos(2 * thesis::CONST_PI * i * j / N) +
                  in[j][1] * std::sin(2 * thesis::CONST_PI * i * j / N);
        exp[1] -= in[j][0] * std::sin(2 * thesis::CONST_PI * i * j / N) -
                  in[j][1] * std::cos(2 * thesis::CONST_PI * i * j / N);
      }
      EXPECT_NEAR(exp[0], out[i][0], 1e-10);
      EXPECT_NEAR(exp[1], out[i][1], 1e-10);
    }

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
  } catch (...) {
    FAIL() << "Expected: No exception";
  }
}
