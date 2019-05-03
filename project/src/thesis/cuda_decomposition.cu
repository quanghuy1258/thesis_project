#ifdef USING_CUDA

#include "thesis/decomposition.h"
#include "thesis/trgsw_cipher.h"
#include "thesis/trlwe_cipher.h"

#include <assert.h>

namespace thesis {

const int bitsize_Torus = 8 * sizeof(TorusInteger);

__global__ void _cudaOnlyDecomp(int N, int k, int l, int Bgbit,
                                TorusInteger offset, TorusInteger maskMod,
                                TorusInteger halfBg, TorusInteger *inp,
                                TorusInteger *out) {
  int _N = blockIdx.x * blockDim.x + threadIdx.x;
  int _l = blockIdx.y * blockDim.y + threadIdx.y;
  int _k = blockIdx.z * blockDim.z + threadIdx.z;
  if (_N < N || _k < k || _l < l) {
    assert(bitsize_Torus >= Bgbit * (_l + 1));
    TorusInteger decomp = inp[N * _k + _N] + offset;
    out[N * (l * _k + _l) + _N] = decomp >> (bitsize_Torus - Bgbit * (_l + 1));
    out[N * (l * _k + _l) + _N] &= maskMod;
    out[N * (l * _k + _l) + _N] -= halfBg;
  }
}

__global__ void _cudaForBlindRotate(int N, int k, int l, int Bgbit, int deg,
                                    TorusInteger offset, TorusInteger maskMod,
                                    TorusInteger halfBg, TorusInteger *inp,
                                    TorusInteger *out) {
  int _N = blockIdx.x * blockDim.x + threadIdx.x;
  int _l = blockIdx.y * blockDim.y + threadIdx.y;
  int _k = blockIdx.z * blockDim.z + threadIdx.z;
  if (_N < N || _k < k || _l < l) {
    assert(bitsize_Torus >= Bgbit * (_l + 1));
    TorusInteger decomp = 0;
    if (_N >= deg)
      decomp += inp[N * _k + _N - deg];
    else if (_N + N >= deg)
      decomp -= inp[N * _k + _N + N - deg];
    else
      decomp += inp[N * _k + _N + N * 2 - deg];
    decomp -= inp[N * _k + _N];
    decomp += offset;
    out[N * (l * _k + _l) + _N] = decomp >> (bitsize_Torus - Bgbit * (_l + 1));
    out[N * (l * _k + _l) + _N] &= maskMod;
    out[N * (l * _k + _l) + _N] -= halfBg;
  }
}

void Decomposition::cudaOnlyDecomp(TrlweCipher *inp, TrgswCipher *param,
                                   TorusInteger *out, void *streamPtr) {
  int threadsPerBlock = 512;
  // _N + 511 = _N + (512 - 1)
  dim3 numBlocks((param->_N + 511) / 512, param->_l, param->_k);
  if (streamPtr) {
    cudaStream_t *s = (cudaStream_t *)streamPtr;
    _cudaOnlyDecomp<<<numBlocks, threadsPerBlock, 0, *s>>>(
        param->_N, param->_k, param->_l, param->_Bgbit, param->_offset,
        param->_maskMod, param->_halfBg, inp->_data, out);
  } else
    _cudaOnlyDecomp<<<numBlocks, threadsPerBlock>>>(
        param->_N, param->_k, param->_l, param->_Bgbit, param->_offset,
        param->_maskMod, param->_halfBg, inp->_data, out);
}
void Decomposition::cudaForBlindRotate(TrlweCipher *inp, TrgswCipher *param,
                                       int deg, TorusInteger *out,
                                       void *streamPtr) {
  int threadsPerBlock = 512;
  // _N + 511 = _N + (512 - 1)
  dim3 numBlocks((param->_N + 511) / 512, param->_l, param->_k);
  if (streamPtr) {
    cudaStream_t *s = (cudaStream_t *)streamPtr;
    _cudaForBlindRotate<<<numBlocks, threadsPerBlock, 0, *s>>>(
        param->_N, param->_k, param->_l, param->_Bgbit, deg, param->_offset,
        param->_maskMod, param->_halfBg, inp->_data, out);
  } else
    _cudaForBlindRotate<<<numBlocks, threadsPerBlock>>>(
        param->_N, param->_k, param->_l, param->_Bgbit, deg, param->_offset,
        param->_maskMod, param->_halfBg, inp->_data, out);
}

} // namespace thesis

#endif
