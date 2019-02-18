#include "thesis/trlwe.h"

namespace thesis {

int Trlwe::_N = 1024;
int Trlwe::_k = 1;

// Constructors
Trlwe::Trlwe() {}
Trlwe::Trlwe(const Trlwe &obj) {}

// Destructor
Trlwe::~Trlwe() {}

// Get params
int Trlwe::get_N() { return _N; }
int Trlwe::get_k() { return _k; }

} // namespace thesis
