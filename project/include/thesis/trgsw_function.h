#ifndef TRGSW_FUNCTION_H
#define TRGSW_FUNCTION_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class TrgswFunction {
private:
#ifdef USING_CUDA
  static void cudaAddMuGadget(TorusInteger *pol, TrgswCipher *sample,
                              void *streamPtr = nullptr);
  static void cudaAddMuGadget(TorusInteger scalar, TrgswCipher *sample,
                              void *streamPtr = nullptr);
  static void cudaPartDecrypt(TorusInteger decomp, TorusInteger *plainWithError,
                              int N, TorusInteger *outPol,
                              void *streamPtr = nullptr);
  static void cudaFinalDecrypt(TorusInteger *outPol, int N, int msgSize,
                               void *streamPtr = nullptr);
#endif

public:
  // Encrypt (after creating TRLWE samples)
  static void addMuGadget(TorusInteger *pol, TrgswCipher *sample,
                          void *streamPtr = nullptr);
  static void addMuGadget(TorusInteger scalar, TrgswCipher *sample,
                          void *streamPtr = nullptr);
  // Decrypt (after getting plain from TRLWE ciphers)
  static void partDecrypt(TrgswCipher *cipher, TorusInteger *plainWithError,
                          int rowInBlock, int msgSize, TorusInteger *outPol,
                          void *streamPtr = nullptr);
  static void finalDecrypt(TorusInteger *outPol, int N, int msgSize,
                           void *streamPtr = nullptr);
};

} // namespace thesis

#endif
