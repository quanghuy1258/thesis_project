#ifdef USING_CUDA

#include "thesis/memory_management.h"

namespace thesis {

void *MemoryManagement::_cudaMallocMM(size_t size) {
  void *ptr;
  if (cudaMalloc(&ptr, size) != cudaSuccess)
    return nullptr;
  return ptr;
}
void MemoryManagement::_cudaFreeMM(void *ptr) { cudaFree(ptr); }

} // namespace thesis

#endif
