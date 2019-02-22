#ifndef THESIS_DECLARATIONS_H
#define THESIS_DECLARATIONS_H

#include "thesis/load_lib.h"

namespace thesis {

#if defined(USING_64BIT)
typedef int64_t Torus;
typedef int64_t Integer;
#elif defined(USING_32BIT)
typedef int32_t Torus;
typedef int32_t Integer;
#elif defined(USING_16BIT)
typedef int16_t Torus;
typedef int16_t Integer;
#else
typedef int8_t Torus;
typedef int8_t Integer;
#endif

const double CONST_PI = 4. * std::atan(1.);

class Tfhe;
class Tlwe;
class Trlwe;
class Trgsw;

} // namespace thesis

#endif
