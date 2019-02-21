#ifndef THESIS_DECLARATIONS_H
#define THESIS_DECLARATIONS_H

#include "thesis/load_lib.h"

namespace thesis {

#ifdef USING_64BIT
typedef int64_t INTEGER;
typedef double REAL;
const int NUMBER_BIT_SIZE = 64;
#else
typedef int32_t INTEGER;
typedef float REAL;
const int NUMBER_BIT_SIZE = 32;
#endif

const REAL CONST_PI = 4. * std::atan(1.);

class Tfhe;
class Tlwe;
class Trlwe;
class Trgsw;
class Torus;

} // namespace thesis

#endif
