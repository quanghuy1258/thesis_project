#include "thesis/load_lib.h"
#include "gtest/gtest.h"
#include <unsupported/Eigen/CXX11/ThreadPool>

TEST(Eigen, ThreadPool) {
  try {
    int n = std::thread::hardware_concurrency();
    std::srand(std::time(nullptr));
    Eigen::ThreadPool tp(n);
    Eigen::Barrier b(n);
    for (int i = 0; i < n; i++) {
      tp.Schedule([&, i]() {
        int rd = std::rand() % 2 + 1;
        std::this_thread::sleep_for(std::chrono::seconds(rd));
        std::stringstream ss;
        ss << "[" << i << "," << rd << "]" << std::endl;
        std::cout << ss.str();
        b.Notify();
      });
    }
    b.Wait();
  } catch (...) {
    FAIL() << "Expected: No exception";
  }
}
