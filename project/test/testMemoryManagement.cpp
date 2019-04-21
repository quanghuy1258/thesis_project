#include "gtest/gtest.h"

#include "thesis/declarations.h"
#include "thesis/load_lib.h"
#include "thesis/memory_management.h"

TEST(Thesis, MemoryManagement) {
  {
    thesis::MemoryManagement *p1 = &thesis::MemoryManagement::getInstance();
    thesis::MemoryManagement *p2 = &thesis::MemoryManagement::getInstance();
    ASSERT_TRUE(p1 != nullptr);
    ASSERT_TRUE(p2 != nullptr);
    ASSERT_TRUE(p1 == p2);
  }
  {
    void *p = thesis::MemoryManagement::getInstance().mallocMM(8);
    ASSERT_TRUE(p != nullptr);
    thesis::MemoryManagement::getInstance().freeMM(p);
    ASSERT_NO_FATAL_FAILURE(thesis::MemoryManagement::getInstance().freeMM(p));
  }
  {
    void *p1 = thesis::MemoryManagement::getInstance().mallocMM(8);
    void *p2 = thesis::MemoryManagement::getInstance().mallocMM(8);
    ASSERT_TRUE(p1 != nullptr);
    ASSERT_TRUE(p2 != nullptr);
    ASSERT_TRUE(p1 != p2);
  }
  {
    void *p = thesis::MemoryManagement::getInstance().cudaMallocMM(8);
    ASSERT_TRUE(p != nullptr);
    thesis::MemoryManagement::getInstance().cudaFreeMM(p);
    ASSERT_NO_FATAL_FAILURE(thesis::MemoryManagement::getInstance().cudaFreeMM(p));
  }
  {
    void *p1 = thesis::MemoryManagement::getInstance().cudaMallocMM(8);
    void *p2 = thesis::MemoryManagement::getInstance().cudaMallocMM(8);
    ASSERT_TRUE(p1 != nullptr);
    ASSERT_TRUE(p2 != nullptr);
    ASSERT_TRUE(p1 != p2);
  }
}
