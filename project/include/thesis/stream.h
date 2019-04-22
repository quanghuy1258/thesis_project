#ifndef STREAM_H
#define STREAM_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class Stream {
private:
#ifdef USING_CUDA
  static void *cudaCreateS();
  static void cudaSynchronizeS(void *streamPtr);
  static void cudaDestroyS(void *streamPtr);
#endif

public:
  static void *createS();
  static void synchronizeS(void *streamPtr);
  static void destroyS(void *streamPtr);

#ifndef USING_CUDA
  static void scheduleS(void *streamPtr, std::function<void()> fn);
#endif
};

} // namespace thesis

#endif
