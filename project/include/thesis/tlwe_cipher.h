#ifndef TLWE_CIPHER_H
#define TLWE_CIPHER_H

#include "thesis/cipher.h"
#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class TlweCipher : public Cipher {
public:
  int _n;

  TlweCipher() = delete;
  TlweCipher(const TlweCipher &) = delete;
  TlweCipher(int n, double sdError, double varError);
  TlweCipher(TorusInteger *data, int size, int n, double sdError,
             double varError);

  using Cipher::operator=;
  TlweCipher &operator=(const TlweCipher &) = delete;

  TlweCipher(TlweCipher &&obj);
  TlweCipher &operator=(TlweCipher &&obj);

  ~TlweCipher();
};

} // namespace thesis

#endif
