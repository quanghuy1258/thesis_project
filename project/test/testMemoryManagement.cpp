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
    ASSERT_TRUE(thesis::MemoryManagement::getInstance().freeMM(p));
    ASSERT_NO_FATAL_FAILURE(thesis::MemoryManagement::getInstance().freeMM(p));
  }
  {
    void *p1 = thesis::MemoryManagement::getInstance().mallocMM(8);
    void *p2 = thesis::MemoryManagement::getInstance().mallocMM(8);
    ASSERT_TRUE(p1 != nullptr);
    ASSERT_TRUE(p2 != nullptr);
    ASSERT_TRUE(p1 != p2);
  }
  /* TODO: Need these features?
  {
    std::vector<int> vec1 = {2, 3, 5, 7};
    std::vector<int> vec2(vec1.size());
    void *p1 = thesis::MemoryManagement::getInstance().mallocMM(sizeof(int) *
                                                                vec1.size());
    void *p2 = thesis::MemoryManagement::getInstance().mallocMM(sizeof(int) *
                                                                vec1.size());
    ASSERT_TRUE(thesis::MemoryManagement::getInstance().memcpyMM_h2d(
        p1, vec1.data(), sizeof(int) * vec1.size()));
    ASSERT_TRUE(thesis::MemoryManagement::getInstance().memcpyMM_d2d(
        p2, p1, sizeof(int) * vec1.size()));
    ASSERT_TRUE(thesis::MemoryManagement::getInstance().memcpyMM_d2h(
        vec2.data(), p2, sizeof(int) * vec1.size()));
    for (size_t i = 0; i < vec1.size(); i++)
      ASSERT_TRUE(vec1[i] == vec2[i]);
    ASSERT_TRUE(thesis::MemoryManagement::getInstance().memsetMM(
        p2, 255, sizeof(int) * vec1.size()));
    ASSERT_TRUE(thesis::MemoryManagement::getInstance().memcpyMM_d2h(
        vec1.data(), p2, sizeof(int) * vec1.size()));
    for (size_t i = 0; i < vec1.size(); i++)
      ASSERT_TRUE(vec1[i] == -1);
  }
  */
}
