#ifndef TRLWE_FUNCTION_H
#define TRLWE_FUNCTION_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class TrlweFunction {
private:
#ifdef USING_CUDA
  static void cudaPutPlain(TrlweCipher *sample, TorusInteger plainScalar,
                           void *streamPtr = nullptr);
  static void cudaPutPlain(TrlweCipher *sample, TorusInteger *plainPol,
                           void *streamPtr = nullptr);
  static void cudaRoundPlain(TorusInteger *plain, double *abs_err, int N,
                             void *streamPtr = nullptr);
  static void cudaRotate(TorusInteger *out, TorusInteger *inp, int N, int k,
                         int deg, void *streamPtr = nullptr);
#endif

public:
  static void genkey(TorusInteger *s, int N, int k);
  static void keyToFFT(TorusInteger *s, int N, int k, BatchedFFT *fft);
  // Encrypt
  static void createSample(BatchedFFT *fftWithS, int rowFFT,
                           TrlweCipher *cipher);
  static void createSample(BatchedFFT *fftWithS, int rowFFT, TorusInteger *data,
                           int N, int k, double sdError);
  static void putPlain(TrlweCipher *sample, TorusInteger plainScalar,
                       void *streamPtr = nullptr);
  static void putPlain(TrlweCipher *sample, TorusInteger *plainPol,
                       void *streamPtr = nullptr);
  // Decrypt
  static void getPlain(BatchedFFT *fftWithS, int rowFFT, TrlweCipher *cipher,
                       TorusInteger *plainWithError);
  static void getPlain(BatchedFFT *fftWithS, int rowFFT, TorusInteger *data,
                       int N, int k, TorusInteger *plainWithError);
  static void roundPlain(TorusInteger *plain, double *abs_err, int N,
                         void *streamPtr = nullptr);

  // Rotation
  //   out = inp * X^deg
  static void rotate(TorusInteger *out, TorusInteger *inp, int N, int k,
                     int deg, void *streamPtr = nullptr);
  static void rotate(TrlweCipher *out, TrlweCipher *inp, int deg,
                     void *streamPtr = nullptr);
};

} // namespace thesis

#endif
