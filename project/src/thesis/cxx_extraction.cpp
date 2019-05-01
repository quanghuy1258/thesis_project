#include "thesis/extraction.h"
#include "thesis/stream.h"
#include "thesis/tlwe_cipher.h"
#include "thesis/trlwe_cipher.h"

namespace thesis {

void Extraction::extract(TrlweCipher *inp, int deg, TlweCipher *out,
                         void *streamPtr) {
  if (!inp || !out || deg < 0 || deg >= inp->_N || inp->_N * inp->_k != out->_n)
    return;
#ifdef USING_CUDA
  cudaExtract(inp, deg, out, streamPtr);
#else
  auto fn = [inp, deg, out]() {
    for (int i = 0; i < inp->_k; i++) {
      for (int j = 0; j < inp->_N; j++) {
        if (j > deg)
          out->_data[inp->_N * i + j] =
              -inp->get_pol_data(i)[deg - j + inp->_N];
        else
          out->_data[inp->_N * i + j] = inp->get_pol_data(i)[deg - j];
      }
    }
    out->_data[out->_n] = inp->get_pol_data(inp->_k)[deg];
    out->_sdError = inp->_sdError;
    out->_varError = inp->_varError;
  };
  if (streamPtr)
    Stream::scheduleS(streamPtr, std::move(fn));
  else
    fn();
#endif
}

} // namespace thesis
