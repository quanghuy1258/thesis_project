#ifndef TLWE_CIPHER_H
#define TLWE_CIPHER_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"
#include "thesis/cipher.h"

namespace thesis {

class TlweCipher : public Cipher {
public:
  size_t _n;

  TlweCipher() = delete;
  TlweCipher(const TlweCipher &) = delete;
  TlweCipher(size_t n, double sdError, double varError);
  TlweCipher(TorusInteger *data, size_t size, size_t n, double sdError,
             double varError);

  TlweCipher &operator=(const TlweCipher &);

  ~TlweCipher();
};

} // namespace thesis

#endif
