#include "thesis/torus_utility.h"
#include "thesis/stream.h"

namespace thesis {

void TorusUtility::addVector(TorusInteger *dest, TorusInteger *src, size_t len,
                             void *streamPtr) {
  if (!dest || !src || !len)
    return;
#ifdef USING_CUDA
  cudaAddVector(dest, src, len, streamPtr);
#else
  Stream::scheduleS(
      [dest, src, len]() {
        for (size_t i = 0; i < len; i++)
          dest[i] += src[i];
      },
      streamPtr);
#endif
}
void TorusUtility::subVector(TorusInteger *dest, TorusInteger *src, size_t len,
                             void *streamPtr) {
  if (!dest || !src || !len)
    return;
#ifdef USING_CUDA
  cudaSubVector(dest, src, len, streamPtr);
#else
  Stream::scheduleS(
      [dest, src, len]() {
        for (size_t i = 0; i < len; i++)
          dest[i] -= src[i];
      },
      streamPtr);
#endif
}

} // namespace thesis
