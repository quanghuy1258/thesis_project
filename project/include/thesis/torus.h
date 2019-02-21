#ifndef THESIS_TORUS_H
#define THESIS_TORUS_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class Torus {
private:
  std::bitset<NUMBER_BIT_SIZE> _data;

public:
  // Constructors
  Torus();
  Torus(const INTEGER &n);
  Torus(const REAL &n);
  Torus(const Torus &obj);

  // Destructor
  ~Torus();

  // Copy assignment operator
  Torus &operator=(const Torus &obj);

  // Setting functions
  void setInteger(const INTEGER &n);
  void setReal(const REAL &n);

  // Converting functions
  INTEGER toInteger() const;
  REAL toReal() const;
};

} // namespace thesis

#endif
