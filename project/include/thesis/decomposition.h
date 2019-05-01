#ifndef DECOMPOSITION_H
#define DECOMPOSITION_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class Decomposition {
private:
#ifdef USING_CUDA
  static void cudaOnlyDecomp(TrlweCipher *inp, TrgswCipher *param,
                             TorusInteger *out, void *streamPtr = nullptr);
  static void cudaForBlindRotate(TrlweCipher *inp, TrgswCipher *param, int deg,
                                 TorusInteger *out, void *streamPtr = nullptr);
#endif

public:
  static void onlyDecomp(TrlweCipher *inp, TrgswCipher *param,
                         TorusInteger *out, void *streamPtr = nullptr);
  // Decomposition of TRLWE * (X^deg - 1)
  static void forBlindRotate(TrlweCipher *inp, TrgswCipher *param, int deg,
                             TorusInteger *out, void *streamPtr = nullptr);
};

} // namespace thesis

#endif
