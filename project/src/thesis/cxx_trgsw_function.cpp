#include "thesis/stream.h"
#include "thesis/trgsw_cipher.h"
#include "thesis/trgsw_function.h"

namespace thesis {

const int bitsize_Torus = 8 * sizeof(TorusInteger);

void TrgswFunction::addMuGadget(TorusInteger *pol, TrgswCipher *sample,
                                void *streamPtr) {
  if (!pol || !sample)
    return;
#ifdef USING_CUDA
  cudaAddMuGadget(pol, sample, streamPtr);
#else
  Stream::scheduleS(
      [pol, sample](int parallelId, int parallel) {
        int l = parallelId / (sample->_k + 1);
        int k = parallelId % (sample->_k + 1);
        TorusInteger H = 1;
        H <<= bitsize_Torus - sample->_Bgbit * (l + 1);
        for (int i = 0; i < sample->_N; i++)
          sample->get_pol_data(k * sample->_l + l, k)[i] += H * pol[i];
      },
      sample->_kpl, streamPtr);
#endif
}
void TrgswFunction::addMuGadget(TorusInteger scalar, TrgswCipher *sample,
                                void *streamPtr) {
  if (!sample)
    return;
#ifdef USING_CUDA
  cudaAddMuGadget(scalar, sample, streamPtr);
#else
  Stream::scheduleS(
      [scalar, sample]() {
        for (int i = 0; i < sample->_l; i++) {
          if (bitsize_Torus < sample->_Bgbit * (i + 1))
            break;
          TorusInteger H = 1;
          H <<= bitsize_Torus - sample->_Bgbit * (i + 1);
          for (int j = 0; j <= sample->_k; j++)
            sample->get_pol_data(j * sample->_l + i, j)[0] += H * scalar;
        }
      },
      streamPtr);
#endif
}
void TrgswFunction::partDecrypt(TrgswCipher *cipher,
                                TorusInteger *plainWithError, int rowInBlock,
                                int msgSize, TorusInteger *outPol,
                                void *streamPtr) {
  if (!cipher || !plainWithError || rowInBlock < 0 || msgSize < 1 || !outPol ||
      msgSize > bitsize_Torus ||
      bitsize_Torus < cipher->_Bgbit * (rowInBlock + 1))
    return;
  TorusInteger decomp = 1;
  decomp <<= bitsize_Torus - msgSize;
  decomp += cipher->_offset;
  decomp >>= bitsize_Torus - cipher->_Bgbit * (rowInBlock + 1);
  decomp &= cipher->_maskMod;
  decomp -= cipher->_halfBg;
  if (decomp == 0)
    return;
  int N = cipher->_N;
#ifdef USING_CUDA
  cudaPartDecrypt(decomp, plainWithError, N, outPol, streamPtr);
#else
  Stream::scheduleS(
      [decomp, plainWithError, N, outPol]() {
        for (int i = 0; i < N; i++)
          outPol[i] += plainWithError[i] * decomp;
      },
      streamPtr);
#endif
}
void TrgswFunction::finalDecrypt(TorusInteger *outPol, int N, int msgSize,
                                 void *streamPtr) {
  if (!outPol || N < 2 || (N & (N - 1)) || msgSize < 1 ||
      msgSize > bitsize_Torus)
    return;
#ifdef USING_CUDA
  cudaFinalDecrypt(outPol, N, msgSize);
#else
  Stream::scheduleS(
      [outPol, N, msgSize]() {
        TorusInteger mask = 1;
        mask <<= msgSize;
        mask -= 1;
        for (int i = 0; i < N; i++) {
          double x = outPol[i];
          x /= std::pow(2, bitsize_Torus - msgSize);
          outPol[i] = std::llround(x);
          outPol[i] &= mask;
        }
      },
      streamPtr);
#endif
}

} // namespace thesis
