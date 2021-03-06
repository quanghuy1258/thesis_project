#ifdef USING_CUDA

#include "thesis/stream.h"

namespace thesis {

void *Stream::cudaCreateS() {
  cudaStream_t *streamPtr = (cudaStream_t *)std::malloc(sizeof(cudaStream_t));
  if (cudaStreamCreate(streamPtr) != cudaSuccess) {
    std::free(streamPtr);
    return nullptr;
  }
  return streamPtr;
}
void Stream::cudaSynchronizeS(void *streamPtr) {
  if (streamPtr) {
    cudaStream_t *s = (cudaStream_t *)streamPtr;
    cudaStreamSynchronize(*s);
  }
}
void Stream::cudaDestroyS(void *streamPtr) {
  if (streamPtr) {
    cudaStream_t *s = (cudaStream_t *)streamPtr;
    cudaStreamDestroy(*s);
    std::free(s);
  }
}

} // namespace thesis

#endif
