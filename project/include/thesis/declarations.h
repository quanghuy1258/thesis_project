#ifndef THESIS_DECLARATIONS_H
#define THESIS_DECLARATIONS_H

#include "thesis/load_lib.h"

#define USING_32BIT

namespace thesis {

// TODO: 32 bit first. If possible, add 16 bit and 64 bit later
typedef int32_t Torus;
typedef int32_t Integer;

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
