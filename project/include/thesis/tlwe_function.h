#ifndef TLWE_FUNCTION_H
#define TLWE_FUNCTION_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class TlweFunction {
private:
#ifdef USING_CUDA
  static void cudaGenkey(TorusInteger *s, int n, void *streamPtr = nullptr);
  static void cudaEncrypt(TorusInteger *s, TorusInteger plain,
                          TlweCipher *cipher, void *streamPtr = nullptr);
  static void cudaDecrypt(TorusInteger *s, TlweCipher *cipher,
                          TorusInteger *plain, double *abs_err,
                          void *streamPtr = nullptr);
#endif

public:
  static void genkey(TorusInteger *s, int n, void *streamPtr = nullptr);
  static void encrypt(TorusInteger *s, TorusInteger plain, TlweCipher *cipher,
                      void *streamPtr = nullptr);
  static void decrypt(TorusInteger *s, TlweCipher *cipher, TorusInteger *plain,
                      double *abs_err, void *streamPtr = nullptr);
};

} // namespace thesis

#endif
