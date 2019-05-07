#include "thesis/batched_fft.h"
#include "thesis/memory_management.h"
#include "thesis/random.h"
#include "thesis/stream.h"
#include "thesis/tlwe_function.h"
#include "thesis/trlwe_cipher.h"
#include "thesis/trlwe_function.h"

namespace thesis {

void TrlweFunction::genkey(TorusInteger *s, int N, int k) {
  if (!s || N < 2 || (N & (N - 1)) || k < 1)
    return;
  TlweFunction::genkey(s, N * k);
}
void TrlweFunction::keyToFFT(TorusInteger *s, int N, int k, BatchedFFT *fft) {
  if (!s || N < 2 || (N & (N - 1)) || k < 1 || !fft || fft->get_N() != N ||
      fft->get_col() != k)
    return;
  for (int i = 0; i < k; i++)
    fft->setInp(s + N * i, i);
}
void TrlweFunction::createSample(BatchedFFT *fftWithS, int rowFFT,
                                 TrlweCipher *cipher) {
  if (!fftWithS || rowFFT < 0 || rowFFT >= fftWithS->get_row() || !cipher ||
      cipher->_N != fftWithS->get_N() || cipher->_k != fftWithS->get_col())
    return;
  Random::setUniform(cipher->_data, cipher->_N * cipher->_k);
  Random::setNormalTorus(cipher->_data + cipher->_N * cipher->_k, cipher->_N,
                         cipher->_sdError);
  for (int i = 0; i < cipher->_k; i++)
    fftWithS->setInp(cipher->_data + cipher->_N * i, rowFFT, i);
  for (int i = 0; i < cipher->_k; i++)
    fftWithS->setMul(rowFFT, i);
  fftWithS->addAllOut(cipher->_data + cipher->_N * cipher->_k, rowFFT);
}
void TrlweFunction::createSample(BatchedFFT *fftWithS, int rowFFT,
                                 TorusInteger *data, int N, int k,
                                 double sdError) {
  if (!fftWithS || rowFFT < 0 || rowFFT >= fftWithS->get_row() || !data ||
      N < 2 || (N & (N - 1)) || k < 1 || sdError < 0 ||
      N != fftWithS->get_N() || k != fftWithS->get_col())
    return;
  Random::setUniform(data, N * k);
  Random::setNormalTorus(data + N * k, N, sdError);
  for (int i = 0; i < k; i++)
    fftWithS->setInp(data + N * i, rowFFT, i);
  for (int i = 0; i < k; i++)
    fftWithS->setMul(rowFFT, i);
  fftWithS->addAllOut(data + N * k, rowFFT);
}
void TrlweFunction::putPlain(TrlweCipher *sample, TorusInteger *plain,
                             void *streamPtr) {
  if (!sample || !plain)
    return;
#ifdef USING_CUDA
  cudaPutPlain(sample, plain, streamPtr);
#else
  Stream::scheduleS(
      [sample, plain]() {
        TorusInteger bit = 1;
        bit <<= 8 * sizeof(TorusInteger) - 1;
        for (int i = 0; i < sample->_N; i++)
          sample->_data[sample->_N * sample->_k + i] += plain[i] * bit;
      },
      streamPtr);
#endif
}
void TrlweFunction::getPlain(BatchedFFT *fftWithS, int rowFFT,
                             TrlweCipher *cipher,
                             TorusInteger *plainWithError) {
  if (!fftWithS || rowFFT < 0 || rowFFT >= fftWithS->get_row() || !cipher ||
      cipher->_N != fftWithS->get_N() || cipher->_k != fftWithS->get_col() ||
      !plainWithError)
    return;
  for (int i = 0; i < cipher->_k; i++)
    fftWithS->setInp(cipher->_data + cipher->_N * i, rowFFT, i);
  for (int i = 0; i < cipher->_k; i++)
    fftWithS->setMul(rowFFT, i);
  MemoryManagement::memcpyMM_d2d(plainWithError,
                                 cipher->_data + cipher->_N * cipher->_k,
                                 cipher->_N * sizeof(TorusInteger));
  fftWithS->subAllOut(plainWithError, rowFFT);
}
void TrlweFunction::roundPlain(TorusInteger *plain, double *abs_err, int N,
                               void *streamPtr) {
  if (!plain)
    return;
#ifdef USING_CUDA
  cudaRoundPlain(plain, abs_err, N, streamPtr);
#else
  Stream::scheduleS(
      [plain, abs_err, N]() {
        for (int i = 0; i < N; i++) {
          double x = std::abs(plain[i] / std::pow(2, 8 * sizeof(TorusInteger)));
          plain[i] = (x < 0.25) ? 0 : 1;
          if (abs_err)
            abs_err[i] = (x < 0.25) ? x : (0.5 - x);
        }
      },
      streamPtr);
#endif
}

} // namespace thesis
