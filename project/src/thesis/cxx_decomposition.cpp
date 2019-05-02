#include "thesis/decomposition.h"
#include "thesis/stream.h"
#include "thesis/trgsw_cipher.h"
#include "thesis/trlwe_cipher.h"

namespace thesis {

const int bitsize_Torus = 8 * sizeof(TorusInteger);

void Decomposition::onlyDecomp(TrlweCipher *inp, TrgswCipher *param,
                               TorusInteger *out, void *streamPtr) {
  if (!inp || !param || !out || inp->_N != param->_N || inp->_k != param->_k)
    return;
#ifdef USING_CUDA
  cudaOnlyDecomp(inp, param, out, streamPtr);
#else
  Stream::scheduleS(
      [inp, param, out](int parallelId, int parallel) {
        int k = parallelId / param->_l;
        int l = parallelId % param->_l;
        if (bitsize_Torus < param->_Bgbit * (l + 1))
          throw std::runtime_error("Cannot implement decomposition here");
        for (int i = 0; i < param->_N; i++) {
          TorusInteger decomp = inp->get_pol_data(k)[i] + param->_offset;
          out[(k * param->_l + l) * param->_N + i] =
              decomp >> (bitsize_Torus - param->_Bgbit * (l + 1));
          out[(k * param->_l + l) * param->_N + i] &= param->_maskMod;
          out[(k * param->_l + l) * param->_N + i] -= param->_halfBg;
        }
      },
      param->_kpl, streamPtr);
#endif
}
void Decomposition::forBlindRotate(TrlweCipher *inp, TrgswCipher *param,
                                   int deg, TorusInteger *out,
                                   void *streamPtr) {
  if (!inp || !param || !out || inp->_N != param->_N || inp->_k != param->_k)
    return;
  int Nx2 = param->_N * 2;
  deg = (deg % Nx2 + Nx2) % Nx2;
#ifdef USING_CUDA
  cudaForBlindRotate(inp, param, deg, out, streamPtr);
#else
  Stream::scheduleS(
      [inp, param, deg, out](int parallelId, int parallel) {
        int k = parallelId / param->_l;
        int l = parallelId % param->_l;
        if (bitsize_Torus < param->_Bgbit * (l + 1))
          throw std::runtime_error("Cannot implement decomposition here");
        for (int i = 0; i < param->_N; i++) {
          TorusInteger decomp = 0;
          if (i >= deg)
            decomp += inp->get_pol_data(k)[i - deg];
          else if (i + param->_N >= deg)
            decomp -= inp->get_pol_data(k)[i + param->_N - deg];
          else
            decomp += inp->get_pol_data(k)[i + param->_N * 2 - deg];
          decomp -= inp->get_pol_data(k)[i];
          decomp += param->_offset;
          out[(k * param->_l + l) * param->_N + i] =
              decomp >> (bitsize_Torus - param->_Bgbit * (l + 1));
          out[(k * param->_l + l) * param->_N + i] &= param->_maskMod;
          out[(k * param->_l + l) * param->_N + i] -= param->_halfBg;
        }
      },
      param->_kpl, streamPtr);
#endif
}

} // namespace thesis
