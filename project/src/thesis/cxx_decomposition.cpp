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
  auto fn = [inp, param, out]() {
    for (int i = 0; i <= param->_k; i++) {
      for (int j = 0; j < param->_N; j++) {
        TorusInteger decomp = inp->get_pol_data(i)[j] + param->_offset;
        for (int k = 0; k < param->_l; k++) {
          if (bitsize_Torus < param->_Bgbit * (k + 1))
            throw std::runtime_error("Cannot implement decomposition here");
          out[(i * param->_l + k) * param->_N + j] =
              decomp >> (bitsize_Torus - param->_Bgbit * (k + 1));
          out[(i * param->_l + k) * param->_N + j] &= param->_maskMod;
          out[(i * param->_l + k) * param->_N + j] -= param->_halfBg;
        }
      }
    }
  };
  if (streamPtr)
    Stream::scheduleS(streamPtr, std::move(fn));
  else
    fn();
#endif
}

} // namespace thesis
