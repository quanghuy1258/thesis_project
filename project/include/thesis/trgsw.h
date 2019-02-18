#ifndef THESIS_TRGSW_H
#define THESIS_TRGSW_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class Trgsw {
private:
  static int _N;
  static int _k;

public:
  // Constructors
  Trgsw();
  Trgsw(const Trgsw &obj);

  // Destructor
  ~Trgsw();

  // Get params
  static int get_N();
  static int get_k();
};

} // namespace thesis

#endif
