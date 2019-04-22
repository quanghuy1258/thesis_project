#ifdef USING_CUDA

#include "thesis/memory_management.h"

namespace thesis {

void *MemoryManagement::cudaMallocMM(size_t size) {
  void *ptr;
  if (cudaMalloc(&ptr, size) != cudaSuccess)
    return nullptr;
  return ptr;
}
void MemoryManagement::cudaFreeMM(void *ptr) { cudaFree(ptr); }
/* TODO: Need these features?
void MemoryManagement::cudaMemsetMM(void *ptr, int ch, size_t count,
                                    void *stream_ptr) {
  if (stream_ptr) {
    cudaStream_t stream = *(cudaStream_t *)stream_ptr;
    cudaMemsetAsync(ptr, ch, count, stream);
  } else
    cudaMemset(ptr, ch, count);
}
void MemoryManagement::cudaMemcpyMM_h2d(void *dest, void *src, size_t count,
                                        void *stream_ptr) {
  if (stream_ptr) {
    cudaStream_t stream = *(cudaStream_t *)stream_ptr;
    cudaMemcpyAsync(dest, src, count, cudaMemcpyHostToDevice, stream);
  } else
    cudaMemcpy(dest, src, count, cudaMemcpyHostToDevice);
}
void MemoryManagement::cudaMemcpyMM_d2h(void *dest, void *src, size_t count,
                                        void *stream_ptr) {
  if (stream_ptr) {
    cudaStream_t stream = *(cudaStream_t *)stream_ptr;
    cudaMemcpyAsync(dest, src, count, cudaMemcpyDeviceToHost, stream);
  } else
    cudaMemcpy(dest, src, count, cudaMemcpyDeviceToHost);
}
void MemoryManagement::cudaMemcpyMM_d2d(void *dest, void *src, size_t count,
                                        void *stream_ptr) {
  if (stream_ptr) {
    cudaStream_t stream = *(cudaStream_t *)stream_ptr;
    cudaMemcpyAsync(dest, src, count, cudaMemcpyDeviceToDevice, stream);
  } else
    cudaMemcpy(dest, src, count, cudaMemcpyDeviceToDevice);
}
*/

} // namespace thesis

#endif
