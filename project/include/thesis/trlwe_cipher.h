#ifndef TRLWE_CIPHER_H
#define TRLWE_CIPHER_H

#include "thesis/cipher.h"
#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class TrlweCipher : public Cipher {
public:
  size_t _N;
  size_t _k;

  TrlweCipher() = delete;
  TrlweCipher(const TlweCipher &) = delete;
  TrlweCipher(size_t N, size_t k, double sdError, double varError);
  TrlweCipher(TorusInteger *data, size_t size, size_t N, size_t k,
              double sdError, double varError);

  TrlweCipher &operator=(const TlweCipher &) = delete;

  ~TrlweCipher();
};

} // namespace thesis

#endif
