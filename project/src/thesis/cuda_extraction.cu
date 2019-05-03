#ifdef USING_CUDA

#include "thesis/extraction.h"
#include "thesis/tlwe_cipher.h"
#include "thesis/trlwe_cipher.h"

namespace thesis {

__global__ void _cudaExtract(int N, int k, int deg, TorusInteger *inp,
                             TorusInteger *out) {
  int _N = blockIdx.x * blockDim.x + threadIdx.x;
  int _k = blockIdx.y * blockDim.y + threadIdx.y;
  if (_N < N && _k < k) {
    if (_N > deg)
      out[N * _k + _N] = -inp[N * _k + deg - _N + N];
    else
      out[N * _k + _N] = inp[N * _k + deg - _N];
  } else if (_N == N && _k == 0)
    out[N * k] = inp[N * k + deg];
}

void Extraction::cudaExtract(TrlweCipher *inp, int deg, TlweCipher *out,
                             void *streamPtr) {
  int threadsPerBlock = 512;
  // _N + 512 = _N + 1 + (512 - 1)
  dim3 numBlocks((inp->_N + 512) / 512, inp->_k);
  if (streamPtr) {
    cudaStream_t *s = (cudaStream_t *)streamPtr;
    _cudaExtract<<<numBlocks, threadsPerBlock, 0, *s>>>(inp->_N, inp->_k, deg,
                                                        inp->_data, out->_data);
  } else
    _cudaExtract<<<numBlocks, threadsPerBlock>>>(inp->_N, inp->_k, deg,
                                                 inp->_data, out->_data);
}

} // namespace thesis

#endif
