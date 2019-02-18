#include "thesis/trgsw.h"

namespace thesis {

int Trgsw::_N = 1024;
int Trgsw::_k = 1;

// Constructors
Trgsw::Trgsw() {}
Trgsw::Trgsw(const Trgsw &obj) {}

// Destructor
Trgsw::~Trgsw() {}

// Get params
int Trgsw::get_N() { return _N; }
int Trgsw::get_k() { return _k; }

} // namespace thesis
