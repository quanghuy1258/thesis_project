#ifdef USING_CUDA

#include "thesis/tlwe_cipher.h"
#include "thesis/tlwe_function.h"

namespace thesis {

__global__ void _cudaGenkey(int n, TorusInteger *s) {
  int _n = blockIdx.x * blockDim.x + threadIdx.x;
  if (_n < n)
    s[_n] &= 1;
}

__global__ void _cudaEncrypt(int n, TorusInteger *s, TorusInteger plain,
                             TorusInteger *cipher) {
  for (int i = 0; i < n; i++)
    cipher[n] += s[i] * cipher[i];
  TorusInteger bit = 1;
  bit <<= 8 * sizeof(TorusInteger) - 1;
  cipher[n] += plain * bit;
}

__global__ void _cudaDecrypt(int n, TorusInteger *s, TorusInteger *cipher,
                             TorusInteger *plain, double *abs_err) {
  TorusInteger x = cipher[n];
  for (int i = 0; i < n; i++)
    x -= s[i] * cipher[i];
  double y = std::abs(x / std::pow(2, 8 * sizeof(TorusInteger)));
  if (plain)
    *plain = (y < 0.25) ? 0 : 1;
  if (abs_err)
    *abs_err = (y < 0.25) ? y : (0.5 - y);
}

void TlweFunction::cudaGenkey(TorusInteger *s, int n, void *streamPtr) {
  int threadsPerBlock = 512;
  // n + 511 = n + (512 - 1)
  int numBlocks = (n + 511) / 512;
  if (streamPtr) {
    cudaStream_t *str = (cudaStream_t *)streamPtr;
    _cudaGenkey<<<numBlocks, threadsPerBlock, 0, *str>>>(n, s);
  } else
    _cudaGenkey<<<numBlocks, threadsPerBlock>>>(n, s);
}
void TlweFunction::cudaEncrypt(TorusInteger *s, TorusInteger plain,
                               TlweCipher *cipher, void *streamPtr) {
  if (streamPtr) {
    cudaStream_t *str = (cudaStream_t *)streamPtr;
    _cudaEncrypt<<<1, 1, 0, *str>>>(cipher->_n, s, plain, cipher->_data);
  } else
    _cudaEncrypt<<<1, 1>>>(cipher->_n, s, plain, cipher->_data);
}
void TlweFunction::cudaDecrypt(TorusInteger *s, TlweCipher *cipher,
                               TorusInteger *plain, double *abs_err,
                               void *streamPtr) {
  if (streamPtr) {
    cudaStream_t *str = (cudaStream_t *)streamPtr;
    _cudaDecrypt<<<1, 1, 0, *str>>>(cipher->_n, s, cipher->_data, plain,
                                    abs_err);
  } else
    _cudaDecrypt<<<1, 1>>>(cipher->_n, s, cipher->_data, plain, abs_err);
}

} // namespace thesis

#endif
