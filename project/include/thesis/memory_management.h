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
  void *cudaMallocMM(size_t size);
  void cudaFreeMM(void *ptr);
  /* TODO: Need these features?
  void cudaMemsetMM(void *ptr, int ch, size_t count,
                    void *stream_ptr = nullptr);
  void cudaMemcpyMM_h2d(void *dest, void *src, size_t count,
                        void *stream_ptr = nullptr);
  void cudaMemcpyMM_d2h(void *dest, void *src, size_t count,
                        void *stream_ptr = nullptr);
  void cudaMemcpyMM_d2d(void *dest, void *src, size_t count,
                        void *stream_ptr = nullptr);
  */
#endif

public:
  static MemoryManagement &getInstance();

  // VRAM
  void *mallocMM(size_t size);
  bool freeMM(void *ptr);
  /* TODO: Need these features?
  bool memsetMM(void *ptr, int ch, size_t count, void *stream_ptr = nullptr);
  bool memcpyMM_h2d(void *dest, void *src, size_t count,
                    void *stream_ptr = nullptr);
  bool memcpyMM_d2h(void *dest, void *src, size_t count,
                    void *stream_ptr = nullptr);
  bool memcpyMM_d2d(void *dest, void *src, size_t count,
                    void *stream_ptr = nullptr);
  */
};

} // namespace thesis

#endif
