#include "thesis/threadpool.h"
#include "thesis/load_lib.h"

namespace thesis {

int ThreadPool::_numberThreads = std::thread::hardware_concurrency();
Eigen::ThreadPool ThreadPool::_threadPool(ThreadPool::get_numberThreads());
int ThreadPool::get_numberThreads() { return _numberThreads; }
Eigen::ThreadPool &ThreadPool::get_threadPool() { return _threadPool; }

} // namespace thesis
