#ifndef THESIS_BARRIER_H
#define THESIS_BARRIER_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class Barrier {
private:
  void *_data;

public:
  Barrier();
  Barrier(const Barrier &) = delete;

  Barrier &operator=(const Barrier &) = delete;

  ~Barrier();

  void reset(size_t count = 0);
  void notify();
  void wait();
};

} // namespace thesis

#endif
