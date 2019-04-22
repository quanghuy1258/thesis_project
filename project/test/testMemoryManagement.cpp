#include "gtest/gtest.h"

#include "thesis/declarations.h"
#include "thesis/load_lib.h"
#include "thesis/memory_management.h"

TEST(Thesis, MemoryManagement) {
  {
    void *p = thesis::MemoryManagement::mallocMM(8);
    ASSERT_TRUE(p != nullptr);
    thesis::MemoryManagement::freeMM(p);
  }
  {
    void *p1 = thesis::MemoryManagement::mallocMM(8);
    void *p2 = thesis::MemoryManagement::mallocMM(8);
    ASSERT_TRUE(p1 != nullptr);
    ASSERT_TRUE(p2 != nullptr);
    ASSERT_TRUE(p1 != p2);
    thesis::MemoryManagement::freeMM(p1);
    thesis::MemoryManagement::freeMM(p2);
  }
  {
    std::vector<int> vec1 = {2, 3, 5, 7};
    std::vector<int> vec2(vec1.size());
    void *p1 = thesis::MemoryManagement::mallocMM(sizeof(int) * vec1.size());
    void *p2 = thesis::MemoryManagement::mallocMM(sizeof(int) * vec1.size());
    thesis::MemoryManagement::memcpyMM_h2d(p1, vec1.data(),
                                           sizeof(int) * vec1.size());
    thesis::MemoryManagement::memcpyMM_d2d(p2, p1, sizeof(int) * vec1.size());
    thesis::MemoryManagement::memcpyMM_d2h(vec2.data(), p2,
                                           sizeof(int) * vec1.size());
    for (size_t i = 0; i < vec1.size(); i++)
      ASSERT_TRUE(vec1[i] == vec2[i]);
    thesis::MemoryManagement::memsetMM(p2, 255, sizeof(int) * vec1.size());
    thesis::MemoryManagement::memcpyMM_d2h(vec1.data(), p2,
                                           sizeof(int) * vec1.size());
    for (size_t i = 0; i < vec1.size(); i++)
      ASSERT_TRUE(vec1[i] == -1);
    thesis::MemoryManagement::freeMM(p1);
    thesis::MemoryManagement::freeMM(p2);
  }
}
