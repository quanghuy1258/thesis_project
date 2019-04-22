#include "thesis/memory_management.h"

namespace thesis {

static std::mutex m_ptr_mtx;
static std::map<void *, bool> m_ptr;

MemoryManagement::MemoryManagement() {
  std::lock_guard<std::mutex> guard(m_ptr_mtx);
  m_ptr.clear();
}

MemoryManagement::~MemoryManagement() {
  std::lock_guard<std::mutex> guard(m_ptr_mtx);
  for (auto &x : m_ptr) {
    if (x.second)
      continue;
#ifdef USING_CUDA
      // Fixed: CUDA will free this memory area automatically when exiting
      // cudaFreeMM(x.first);
#else
    free(x.first);
#endif
    x.second = true;
  }
}

MemoryManagement &MemoryManagement::getInstance() {
  static MemoryManagement instance;
  return instance;
}

void *MemoryManagement::mallocMM(size_t size) {
  if (size == 0)
    return nullptr;
  std::lock_guard<std::mutex> guard(m_ptr_mtx);
#ifdef USING_CUDA
  void *ptr = cudaMallocMM(size);
#else
  void *ptr = malloc(size);
#endif
  if (ptr == nullptr)
    return nullptr;
  m_ptr[ptr] = false;
  return ptr;
}
bool MemoryManagement::freeMM(void *ptr) {
  if (ptr == nullptr)
    return false;
  std::lock_guard<std::mutex> guard(m_ptr_mtx);
  auto it = m_ptr.find(ptr);
  if (it == m_ptr.end() || it->second)
    return false;
#ifdef USING_CUDA
  cudaFreeMM(ptr);
#else
  free(ptr);
#endif
  it->second = true;
  return true;
}
/* TODO: Need these features?
bool MemoryManagement::memsetMM(void *ptr, int ch, size_t count,
                                void *stream_ptr) {
  if (ptr == nullptr)
    return false;
  std::lock_guard<std::mutex> guard(m_ptr_mtx);
  auto it = m_ptr.find(ptr);
  if (it == m_ptr.end() || it->second)
    return false;
#ifdef USING_CUDA
  cudaMemsetMM(ptr, ch, count, stream_ptr);
#else
  memset(ptr, ch, count);
#endif
  return true;
}
bool MemoryManagement::memcpyMM_h2d(void *dest, void *src, size_t count,
                                    void *stream_ptr) {
  if (dest == nullptr || src == nullptr)
    return false;
  std::lock_guard<std::mutex> guard(m_ptr_mtx);
  auto it = m_ptr.find(dest);
  if (it == m_ptr.end() || it->second)
    return false;
#ifdef USING_CUDA
  cudaMemcpyMM_h2d(dest, src, count, stream_ptr);
#else
  memcpy(dest, src, count);
#endif
  return true;
}
bool MemoryManagement::memcpyMM_d2h(void *dest, void *src, size_t count,
                                    void *stream_ptr) {
  if (dest == nullptr || src == nullptr)
    return false;
  std::lock_guard<std::mutex> guard(m_ptr_mtx);
  auto it = m_ptr.find(src);
  if (it == m_ptr.end() || it->second)
    return false;
#ifdef USING_CUDA
  cudaMemcpyMM_d2h(dest, src, count, stream_ptr);
#else
  memcpy(dest, src, count);
#endif
  return true;
}
bool MemoryManagement::memcpyMM_d2d(void *dest, void *src, size_t count,
                                    void *stream_ptr) {
  if (dest == nullptr || src == nullptr)
    return false;
  std::lock_guard<std::mutex> guard(m_ptr_mtx);
  auto it_src = m_ptr.find(src);
  auto it_dest = m_ptr.find(dest);
  if (it_src == m_ptr.end() || it_src->second || it_dest == m_ptr.end() ||
      it_dest->second)
    return false;
#ifdef USING_CUDA
  cudaMemcpyMM_d2d(dest, src, count, stream_ptr);
#else
  memcpy(dest, src, count);
#endif
  return true;
}
*/

} // namespace thesis
