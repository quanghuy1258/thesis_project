#ifndef THESIS_DECLARATIONS_H
#define THESIS_DECLARATIONS_H

#include "thesis/load_lib.h"

namespace thesis {

class Tfhe;
class Tlwe;
class Trlwe;
class Trgsw;

const double CONST_PI = 4. * std::atan(1.);

#ifdef USING_INTEGER_64BITS
typedef int64_t INTEGER;
#else
typedef int32_t INTEGER;
#endif

#ifdef USING_REAL_64BITS
typedef double REAL;
#else
typedef float REAL;
#endif

} // namespace thesis

#endif
