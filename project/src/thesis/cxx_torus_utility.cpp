#include "thesis/stream.h"
#include "thesis/torus_utility.h"

namespace thesis {

void TorusUtility::addVector(TorusInteger *dest, TorusInteger *src, int len,
                             void *streamPtr) {
  if (!dest || !src || len < 1)
    return;
#ifdef USING_CUDA
  cudaAddVector(dest, src, len, streamPtr);
#else
  Stream::scheduleS(
      [dest, src, len]() {
        for (int i = 0; i < len; i++)
          dest[i] += src[i];
      },
      streamPtr);
#endif
}
void TorusUtility::subVector(TorusInteger *dest, TorusInteger *src, int len,
                             void *streamPtr) {
  if (!dest || !src || len < 1)
    return;
#ifdef USING_CUDA
  cudaSubVector(dest, src, len, streamPtr);
#else
  Stream::scheduleS(
      [dest, src, len]() {
        for (int i = 0; i < len; i++)
          dest[i] -= src[i];
      },
      streamPtr);
#endif
}

} // namespace thesis
