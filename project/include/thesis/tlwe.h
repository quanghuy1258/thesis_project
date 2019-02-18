#ifndef THESIS_TLWE_H
#define THESIS_TLWE_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class Tlwe {
private:
  static int _N;
  static int _k;
public:
  // Constructors
  Tlwe();
  Tlwe(const Tlwe &obj);

  // Destructor
  ~Tlwe();
};

}; // namespace thesis

#endif
