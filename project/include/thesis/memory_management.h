#ifndef MEMORY_MANAGEMENT_H
#define MEMORY_MANAGEMENT_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class MemoryManagement {
private:
#ifdef USING_CUDA
  static void *cudaMallocMM(size_t size);
  static void cudaFreeMM(void *ptr);
  static void cudaMemsetMM(void *ptr, int ch, size_t count,
                           void *stream_ptr = nullptr);
  static void cudaMemcpyMM_h2d(void *dest, void *src, size_t count,
                               void *stream_ptr = nullptr);
  static void cudaMemcpyMM_d2h(void *dest, void *src, size_t count,
                               void *stream_ptr = nullptr);
  static void cudaMemcpyMM_d2d(void *dest, void *src, size_t count,
                               void *stream_ptr = nullptr);
#endif

public:
  // Try to use VRAM if possible
  static void *mallocMM(size_t size);
  static void freeMM(void *ptr);
  static void memsetMM(void *ptr, int ch, size_t count,
                       void *stream_ptr = nullptr);
  static void memcpyMM_h2d(void *dest, void *src, size_t count,
                           void *stream_ptr = nullptr);
  static void memcpyMM_d2h(void *dest, void *src, size_t count,
                           void *stream_ptr = nullptr);
  static void memcpyMM_d2d(void *dest, void *src, size_t count,
                           void *stream_ptr = nullptr);
};

} // namespace thesis

#endif
