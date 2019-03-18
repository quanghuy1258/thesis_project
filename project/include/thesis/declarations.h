#ifndef THESIS_DECLARATIONS_H
#define THESIS_DECLARATIONS_H

#include "thesis/load_lib.h"

namespace thesis {

#if defined(USING_32BIT)
typedef int32_t Torus;
typedef int32_t Integer;
#else
typedef int64_t Torus;
typedef int64_t Integer;
#endif

typedef std::vector<bool> PolynomialBinary;
typedef std::vector<Integer> PolynomialInteger;
typedef std::vector<Torus> PolynomialTorus;

const double CONST_PI = 4. * std::atan(1.);

class Tlwe;
class Trlwe;
class Trgsw;

class FFT;
class Random;
class ThreadPool;

} // namespace thesis

#endif
