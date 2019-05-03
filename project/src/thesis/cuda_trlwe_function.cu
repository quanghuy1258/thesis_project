#ifdef USING_CUDA

#include "thesis/trlwe_cipher.h"
#include "thesis/trlwe_function.h"

namespace thesis {

__global__ void _cudaPutPlain(int N, TorusInteger *plain,
                              TorusInteger *cipher) {
  int _N = blockIdx.x * blockDim.x + threadIdx.x;
  if (_N < N) {
    TorusInteger bit = 1;
    bit <<= 8 * sizeof(TorusInteger) - 1;
    cipher[_N] += plain[_N] * bit;
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

void TrlweFunction::cudaPutPlain(TrlweCipher *sample, TorusInteger *plain,
                                 void *streamPtr) {
  int threadsPerBlock = 512;
  // _N + 511 = _N + (512 - 1)
  int numBlocks = (sample->_N + 511) / 512;
  if (streamPtr) {
    cudaStream_t *s = (cudaStream_t *)streamPtr;
    _cudaPutPlain<<<numBlocks, threadsPerBlock, 0, *s>>>(
        sample->_N, plain, sample->get_pol_data(sample->_k));
  } else
    _cudaPutPlain<<<numBlocks, threadsPerBlock>>>(
        sample->_N, plain, sample->get_pol_data(sample->_k));
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

} // namespace thesis

#endif
