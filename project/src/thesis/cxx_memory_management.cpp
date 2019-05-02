#include "thesis/memory_management.h"
#include "thesis/stream.h"

namespace thesis {

void *MemoryManagement::mallocMM(size_t size) {
  if (!size)
    return nullptr;
#ifdef USING_CUDA
  void *ptr = cudaMallocMM(size);
#else
  void *ptr = std::malloc(size);
#endif
  return ptr;
}
void MemoryManagement::freeMM(void *ptr) {
  if (!ptr)
    return;
#ifdef USING_CUDA
  cudaFreeMM(ptr);
#else
  std::free(ptr);
#endif
}
void MemoryManagement::memsetMM(void *ptr, int ch, size_t count,
                                void *stream_ptr) {
  if (!ptr || !count)
    return;
#ifdef USING_CUDA
  cudaMemsetMM(ptr, ch, count, stream_ptr);
#else
  Stream::scheduleS([ptr, ch, count]() { std::memset(ptr, ch, count); },
                    stream_ptr);
#endif
}
void MemoryManagement::memcpyMM_h2d(void *dest, void *src, size_t count,
                                    void *stream_ptr) {
  if (!dest || !src || !count)
    return;
#ifdef USING_CUDA
  cudaMemcpyMM_h2d(dest, src, count, stream_ptr);
#else
  Stream::scheduleS([dest, src, count]() { std::memcpy(dest, src, count); },
                    stream_ptr);
#endif
}
void MemoryManagement::memcpyMM_d2h(void *dest, void *src, size_t count,
                                    void *stream_ptr) {
  if (!dest || !src || !count)
    return;
#ifdef USING_CUDA
  cudaMemcpyMM_d2h(dest, src, count, stream_ptr);
#else
  Stream::scheduleS([dest, src, count]() { std::memcpy(dest, src, count); },
                    stream_ptr);
#endif
}
void MemoryManagement::memcpyMM_d2d(void *dest, void *src, size_t count,
                                    void *stream_ptr) {
  if (!dest || !src || !count)
    return;
#ifdef USING_CUDA
  cudaMemcpyMM_d2d(dest, src, count, stream_ptr);
#else
  Stream::scheduleS([dest, src, count]() { std::memcpy(dest, src, count); },
                    stream_ptr);
#endif
}

} // namespace thesis
