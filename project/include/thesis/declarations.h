#ifndef THESIS_DECLARATIONS_H
#define THESIS_DECLARATIONS_H

#include "thesis/load_lib.h"

namespace thesis {

#if defined(USING_32BIT)
typedef int32_t TorusInteger;
#else
typedef int64_t TorusInteger;
#endif

const double CONST_PI = 4. * std::atan(1.);

class Barrier;
class MemoryManagement;
class Random;
class Stream;
class ThreadManagement;
class BatchedFFT;

class Cipher;
class TlweCipher;
class TrlweCipher;
class TrgswCipher;
/*
typedef std::vector<bool> PolynomialBinary;
typedef std::vector<Integer> PolynomialInteger;
typedef std::vector<Torus> PolynomialTorus;
*/

//class Tlwe;
//class Trlwe;
//class Trgsw;

//class ThreadPool;

} // namespace thesis

#endif
