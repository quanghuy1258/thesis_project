#include "thesis/torus.h"

namespace thesis {

// Constructors
Torus::Torus() { _data.reset(); }
Torus::Torus(const INTEGER &n) {}
Torus::Torus(const REAL &n) {}
Torus::Torus(const Torus &obj) { _data = obj._data; }

// Destructor
Torus::~Torus() {}

// Copy assignment operator
Torus &Torus::operator=(const Torus &obj) {
  _data = obj._data;
  return *this;
}

// Setting functions
void Torus::setInteger(const INTEGER &n) {}
void Torus::setReal(const REAL &n) {}

// Converting functions
INTEGER Torus::toInteger() const { return 0; }
REAL Torus::toReal() const { return 0; }

} // namespace thesis
