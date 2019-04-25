#ifndef BOOTSTRAP_H
#define BOOTSTRAP_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class Bootstrap {
private:
  int _N;
  int _k;
  int _l;
  int _Bgbit;

public:
  Bootstrap() = delete;
  Bootstrap(const Bootstrap &) = delete;
  Bootstrap(int N, int k, int l, int Bgbit);

  Bootstrap &operator=(const Bootstrap &) = delete;

  ~Bootstrap();

  void bootstrap();
};

} // namespace thesis

#endif
