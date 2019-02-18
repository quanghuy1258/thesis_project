#ifndef THESIS_TRLWE_H
#define THESIS_TRLWE_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class Trlwe {
private:
  static int _N;
  static int _k;

public:
  // Constructors
  Trlwe();
  Trlwe(const Trlwe &obj);

  // Destructor
  ~Trlwe();

  // Get params
  static int get_N();
  static int get_k();
};

} // namespace thesis

#endif
