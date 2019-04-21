#ifndef MEMORY_MANAGEMENT_H
#define MEMORY_MANAGEMENT_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class MemoryManagement {
private:
  MemoryManagement();
  MemoryManagement(const MemoryManagement &) = delete;

  MemoryManagement &operator=(const MemoryManagement &) = delete;

  ~MemoryManagement();

#ifdef USING_CUDA
  void *_cudaMallocMM(size_t size);
  void _cudaFreeMM(void *ptr);
#endif

public:
  static MemoryManagement &getInstance();

  // RAM
  void *mallocMM(size_t size);
  void freeMM(void *ptr);

  // VRAM
  void *cudaMallocMM(size_t size);
  void cudaFreeMM(void *ptr);
};

} // namespace thesis

#endif
