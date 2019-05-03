#ifdef USING_CUDA

#include "thesis/trgsw_cipher.h"
#include "thesis/trgsw_function.h"

#include <assert.h>

namespace thesis {

const int bitsize_Torus = 8 * sizeof(TorusInteger);

__global__ void _cudaAddMuGadget(int N, int k, int l, int Bgbit,
                                 TorusInteger *pol, TorusInteger *out) {
  int _N = blockIdx.x * blockDim.x + threadIdx.x;
  int _l = blockIdx.y * blockDim.y + threadIdx.y;
  int _k = blockIdx.z * blockDim.z + threadIdx.z;
  if (_N < N && _k <= k && _l < l) {
    assert(bitsize_Torus >= Bgbit * (_l + 1));
    TorusInteger H = 1;
    H <<= bitsize_Torus - Bgbit * (_l + 1);
    out[N * ((k + 1) * (l * _k + _l) + _k) + _N] += H * pol[_N];
  }
}

__global__ void _cudaAddMuGadget(int N, int k, int l, int Bgbit,
                                 TorusInteger scalar, TorusInteger *out) {
  int _l = blockIdx.x * blockDim.x + threadIdx.x;
  int _k = blockIdx.y * blockDim.y + threadIdx.y;
  if (_k <= k && _l < l) {
    assert(bitsize_Torus >= Bgbit * (_l + 1));
    TorusInteger H = 1;
    H <<= bitsize_Torus - Bgbit * (_l + 1);
    out[N * ((k + 1) * (l * _k + _l) + _k)] += H * scalar;
  }
}

__global__ void _cudaPartDecrypt(TorusInteger decomp,
                                 TorusInteger *plainWithError, int N,
                                 TorusInteger *outPol) {
  int _N = blockIdx.x * blockDim.x + threadIdx.x;
  if (_N < N)
    outPol[_N] += plainWithError[_N] * decomp;
}

__global__ void _cudaFinalDecrypt(TorusInteger *outPol, int N, int msgSize) {
  int _N = blockIdx.x * blockDim.x + threadIdx.x;
  if (_N < N) {
    TorusInteger mask = 1;
    mask <<= msgSize;
    mask -= 1;
    double x = outPol[_N];
    x /= std::pow(2, bitsize_Torus - msgSize);
    outPol[_N] = std::llround(x);
    outPol[_N] &= mask;
  }
}

void TrgswFunction::cudaAddMuGadget(TorusInteger *pol, TrgswCipher *sample,
                                    void *streamPtr) {
  int threadsPerBlock = 512;
  // _N + 511 = _N + (512 - 1)
  dim3 numBlocks((sample->_N + 511) / 512, sample->_l, sample->_k + 1);
  if (streamPtr) {
    cudaStream_t *s = (cudaStream_t *)streamPtr;
    _cudaAddMuGadget<<<numBlocks, threadsPerBlock, 0, *s>>>(
        sample->_N, sample->_k, sample->_l, sample->_Bgbit, pol, sample->_data);
  } else
    _cudaAddMuGadget<<<numBlocks, threadsPerBlock>>>(
        sample->_N, sample->_k, sample->_l, sample->_Bgbit, pol, sample->_data);
}
void TrgswFunction::cudaAddMuGadget(TorusInteger scalar, TrgswCipher *sample,
                                    void *streamPtr) {
  int threadsPerBlock = 512;
  // _l + 511 = _l + (512 - 1)
  dim3 numBlocks((sample->_l + 511) / 512, sample->_k + 1);
  if (streamPtr) {
    cudaStream_t *s = (cudaStream_t *)streamPtr;
    _cudaAddMuGadget<<<numBlocks, threadsPerBlock, 0, *s>>>(
        sample->_N, sample->_k, sample->_l, sample->_Bgbit, scalar,
        sample->_data);
  } else
    _cudaAddMuGadget<<<numBlocks, threadsPerBlock>>>(sample->_N, sample->_k,
                                                     sample->_l, sample->_Bgbit,
                                                     scalar, sample->_data);
}
void TrgswFunction::cudaPartDecrypt(TorusInteger decomp,
                                    TorusInteger *plainWithError, int N,
                                    TorusInteger *outPol, void *streamPtr) {
  int threadsPerBlock = 512;
  // N + 511 = N + (512 - 1)
  int numBlocks = (N + 511) / 512;
  if (streamPtr) {
    cudaStream_t *s = (cudaStream_t *)streamPtr;
    _cudaPartDecrypt<<<numBlocks, threadsPerBlock, 0, *s>>>(
        decomp, plainWithError, N, outPol);
  } else
    _cudaPartDecrypt<<<numBlocks, threadsPerBlock>>>(decomp, plainWithError, N,
                                                     outPol);
}
void TrgswFunction::cudaFinalDecrypt(TorusInteger *outPol, int N, int msgSize,
                                     void *streamPtr) {
  int threadsPerBlock = 512;
  // N + 511 = N + (512 - 1)
  int numBlocks = (N + 511) / 512;
  if (streamPtr) {
    cudaStream_t *s = (cudaStream_t *)streamPtr;
    _cudaFinalDecrypt<<<numBlocks, threadsPerBlock, 0, *s>>>(outPol, N,
                                                             msgSize);
  } else
    _cudaFinalDecrypt<<<numBlocks, threadsPerBlock>>>(outPol, N, msgSize);
}

} // namespace thesis

#endif
