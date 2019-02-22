#ifndef THESIS_THREADPOOL_H
#define THESIS_THREADPOOL_H

#include <unsupported/Eigen/CXX11/ThreadPool>

namespace thesis {

class ThreadPool {
private:
  static int _numberThreads;
  static Eigen::ThreadPool _threadPool;

public:
  static int get_numberThreads();
  static Eigen::ThreadPool &get_threadPool();
};

} // namespace thesis

#endif
