#include "thesis/thread_management.h"

#include <unsupported/Eigen/CXX11/ThreadPool>

namespace thesis {

static std::mutex mtx;
static int n_threads = std::thread::hardware_concurrency();
static Eigen::ThreadPool thread_pool(n_threads);

int ThreadManagement::getNumberThreadsInPool() { return n_threads; }
void ThreadManagement::schedule(std::function<void()> fn) {
  std::lock_guard<std::mutex> guard(mtx);
  thread_pool.Schedule(std::move(fn));
}

} // namespace thesis

