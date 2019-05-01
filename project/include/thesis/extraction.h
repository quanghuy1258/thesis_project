#ifndef EXTRACTION_H
#define EXTRACTION_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class Extraction {
private:
#ifdef USING_CUDA
  static void cudaExtract(TrlweCipher *inp, int deg, TlweCipher *out,
                          void *streamPtr = nullptr);
#endif

public:
  static void extract(TrlweCipher *inp, int deg, TlweCipher *out,
                      void *streamPtr = nullptr);
};

} // namespace thesis

#endif
