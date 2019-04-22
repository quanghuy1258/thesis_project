#ifndef THREAD_MANAGEMENT_H
#define THREAD_MANAGEMENT_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class ThreadManagement {
public:
  static int getNumberThreadsInPool();
  static void schedule(std::function<void()> fn);
};

} // namespace thesis

#endif
