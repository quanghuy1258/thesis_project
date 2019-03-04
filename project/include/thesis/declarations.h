#ifndef THESIS_DECLARATIONS_H
#define THESIS_DECLARATIONS_H

#include "thesis/load_lib.h"

#define USING_32BIT

namespace thesis {

#if defined(USING_32BIT)
typedef int32_t Torus;
typedef int32_t Integer;
#elif defined(USING_16BIT)
typedef int16_t Torus;
typedef int16_t Integer;
#else
typedef int8_t Torus;
typedef int8_t Integer;
#endif
typedef std::vector<bool> PolynomialBinary;
typedef std::vector<Integer> PolynomialInteger;
typedef std::vector<Torus> PolynomialTorus;

const double CONST_PI = 4. * std::atan(1.);

class Tfhe;
class Tlwe;
class Trlwe;
class Trgsw;

class FFT;

} // namespace thesis

#endif
