#ifdef USING_CUDA

#include "thesis/torus_utility.h"

namespace thesis {

__global__ void _cudaAddVector(TorusInteger *dest, TorusInteger *src, int len) {
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l < len)
    dest[l] += src[l];
}

__global__ void _cudaSubVector(TorusInteger *dest, TorusInteger *src, int len) {
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l < len)
    dest[l] -= src[l];
}

void TorusUtility::cudaAddVector(TorusInteger *dest, TorusInteger *src,
                                 size_t len, void *streamPtr) {
  int threadsPerBlock = 512;
  // len + 511 = len + (512 - 1)
  int numBlocks = (len + 511) / 512;
  if (streamPtr) {
    cudaStream_t *s = (cudaStream_t *)streamPtr;
    _cudaAddVector<<<numBlocks, threadsPerBlock, 0, *s>>>(dest, src, len);
  } else
    _cudaAddVector<<<numBlocks, threadsPerBlock>>>(dest, src, len);
}
void TorusUtility::cudaSubVector(TorusInteger *dest, TorusInteger *src,
                                 size_t len, void *streamPtr) {
  int threadsPerBlock = 512;
  // len + 511 = len + (512 - 1)
  int numBlocks = (len + 511) / 512;
  if (streamPtr) {
    cudaStream_t *s = (cudaStream_t *)streamPtr;
    _cudaSubVector<<<numBlocks, threadsPerBlock, 0, *s>>>(dest, src, len);
  } else
    _cudaSubVector<<<numBlocks, threadsPerBlock>>>(dest, src, len);
}

} // namespace thesis

#endif
