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
class TorusUtility;

class Cipher;
class TlweCipher;
class TrlweCipher;
class TrgswCipher;

class TlweFunction;
class TrlweFunction;
class TrgswFunction;
class Extraction;
class Decomposition;

} // namespace thesis

#endif
