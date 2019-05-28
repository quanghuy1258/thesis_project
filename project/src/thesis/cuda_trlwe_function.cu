#ifdef USING_CUDA

#include "thesis/trlwe_cipher.h"
#include "thesis/trlwe_function.h"

namespace thesis {

__global__ void _cudaPutPlain(TorusInteger plainScalar, TorusInteger *cipher) {
  TorusInteger bit = 1;
  bit <<= 8 * sizeof(TorusInteger) - 1;
  cipher[0] += plainScalar * bit;
}

__global__ void _cudaPutPlain(int N, TorusInteger *plainPol,
                              TorusInteger *cipher) {
  int _N = blockIdx.x * blockDim.x + threadIdx.x;
  if (_N < N) {
    TorusInteger bit = 1;
    bit <<= 8 * sizeof(TorusInteger) - 1;
    cipher[_N] += plainPol[_N] * bit;
  }
}

__global__ void _cudaRoundPlain(TorusInteger *plain, double *abs_err, int N) {
  int _N = blockIdx.x * blockDim.x + threadIdx.x;
  if (_N < N) {
    double x = std::abs(plain[_N] / std::pow(2, 8 * sizeof(TorusInteger)));
    plain[_N] = (x < 0.25) ? 0 : 1;
    if (abs_err)
      abs_err[_N] = (x < 0.25) ? x : (0.5 - x);
  }
}

__global__ void _cudaRotate(TorusInteger *out, TorusInteger *inp, int N, int k,
                            int deg) {
  int _N = blockIdx.x * blockDim.x + threadIdx.x;
  int _k = blockIdx.y * blockDim.y + threadIdx.y;
  if (_N < N && _k <= k) {
    if (_N >= deg)
      out[_k * N + _N] = inp[_k * N + _N - deg];
    else if (_N + N >= deg)
      out[_k * N + _N] = -inp[_k * N + _N + N - deg];
    else
      out[_k * N + _N] = inp[_k * N + _N + N * 2 - deg];
  }
}

void TrlweFunction::cudaPutPlain(TrlweCipher *sample, TorusInteger plainScalar,
                                 void *streamPtr) {
  if (streamPtr) {
    cudaStream_t *s = (cudaStream_t *)streamPtr;
    _cudaPutPlain<<<1, 1, 0, *s>>>(plainScalar,
                                   sample->get_pol_data(sample->_k));
  } else
    _cudaPutPlain<<<1, 1>>>(plainScalar, sample->get_pol_data(sample->_k));
}
void TrlweFunction::cudaPutPlain(TrlweCipher *sample, TorusInteger *plainPol,
                                 void *streamPtr) {
  int threadsPerBlock = 512;
  // _N + 511 = _N + (512 - 1)
  int numBlocks = (sample->_N + 511) / 512;
  if (streamPtr) {
    cudaStream_t *s = (cudaStream_t *)streamPtr;
    _cudaPutPlain<<<numBlocks, threadsPerBlock, 0, *s>>>(
        sample->_N, plainPol, sample->get_pol_data(sample->_k));
  } else
    _cudaPutPlain<<<numBlocks, threadsPerBlock>>>(
        sample->_N, plainPol, sample->get_pol_data(sample->_k));
}
void TrlweFunction::cudaRoundPlain(TorusInteger *plain, double *abs_err, int N,
                                   void *streamPtr) {
  int threadsPerBlock = 512;
  // N + 511 = _N + (512 - 1)
  int numBlocks = (N + 511) / 512;
  if (streamPtr) {
    cudaStream_t *s = (cudaStream_t *)streamPtr;
    _cudaRoundPlain<<<numBlocks, threadsPerBlock, 0, *s>>>(plain, abs_err, N);
  } else
    _cudaRoundPlain<<<numBlocks, threadsPerBlock>>>(plain, abs_err, N);
}
void TrlweFunction::cudaRotate(TorusInteger *out, TorusInteger *inp, int N,
                               int k, int deg, void *streamPtr) {
  int threadsPerBlock = 512;
  // N + 511 = _N + (512 - 1)
  dim3 numBlocks((N + 511) / 512, k + 1);
  if (streamPtr) {
    cudaStream_t *s = (cudaStream_t *)streamPtr;
    _cudaRotate<<<numBlocks, threadsPerBlock, 0, *s>>>(out, inp, N, k, deg);
  } else
    _cudaRotate<<<numBlocks, threadsPerBlock>>>(out, inp, N, k, deg);
}

} // namespace thesis

#endif
