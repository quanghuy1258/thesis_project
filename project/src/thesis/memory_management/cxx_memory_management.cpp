#include "thesis/memory_management.h"

namespace thesis {

static std::mutex m_ram_mtx;
static std::map<void *, bool> m_ram;

#ifdef USING_CUDA
static std::mutex m_vram_mtx;
static std::map<void *, bool> m_vram;
#endif

MemoryManagement::MemoryManagement() {
  {
    std::lock_guard<std::mutex> guard(m_ram_mtx);
    m_ram.clear();
  }
#ifdef USING_CUDA
  {
    std::lock_guard<std::mutex> guard(m_vram_mtx);
    m_vram.clear();
  }
#endif
}

MemoryManagement::~MemoryManagement() {
  {
    std::lock_guard<std::mutex> guard(m_ram_mtx);
    for (auto &x : m_ram) {
      if (x.second)
        continue;
      free(x.first);
      x.second = true;
    }
  }
#ifdef USING_CUDA
  {
    std::lock_guard<std::mutex> guard(m_vram_mtx);
    for (auto &x : m_vram) {
      if (x.second)
        continue;
      // Fixed: CUDA will free this memory area automatically when exiting
      //_cudaFreeMM(x.first);
      x.second = true;
    }
  }
#endif
}

MemoryManagement &MemoryManagement::getInstance() {
  static MemoryManagement instance;
  return instance;
}

void *MemoryManagement::mallocMM(size_t size) {
  if (size == 0)
    return nullptr;
  void *ptr = malloc(size);
  if (ptr == nullptr)
    return nullptr;
  {
    std::lock_guard<std::mutex> guard(m_ram_mtx);
    m_ram[ptr] = false;
  }
  return ptr;
}
void MemoryManagement::freeMM(void *ptr) {
  if (ptr == nullptr)
    return;
  std::lock_guard<std::mutex> guard(m_ram_mtx);
  auto it = m_ram.find(ptr);
  if (it == m_ram.end() || it->second)
    return;
  free(ptr);
  it->second = true;
}

void *MemoryManagement::cudaMallocMM(size_t size) {
#ifdef USING_CUDA
  if (size == 0)
    return nullptr;
  void *ptr = _cudaMallocMM(size);
  if (ptr == nullptr)
    return nullptr;
  {
    std::lock_guard<std::mutex> guard(m_vram_mtx);
    m_vram[ptr] = false;
  }
  return ptr;
#else
  return mallocMM(size);
#endif
}
void MemoryManagement::cudaFreeMM(void *ptr) {
#ifdef USING_CUDA
  if (ptr == nullptr)
    return;
  std::lock_guard<std::mutex> guard(m_vram_mtx);
  auto it = m_vram.find(ptr);
  if (it == m_vram.end() || it->second)
    return;
  _cudaFreeMM(ptr);
  it->second = true;
#else
  freeMM(ptr);
#endif
}

} // namespace thesis
