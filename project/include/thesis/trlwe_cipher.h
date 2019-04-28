#ifndef TRLWE_CIPHER_H
#define TRLWE_CIPHER_H

#include "thesis/cipher.h"
#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class TrlweCipher : public Cipher {
public:
  int _N;
  int _k;

  TrlweCipher() = delete;
  TrlweCipher(const TlweCipher &) = delete;
  TrlweCipher(int N, int k, double sdError, double varError);
  TrlweCipher(TorusInteger *data, int size, int N, int k, double sdError,
              double varError);

  TrlweCipher &operator=(const TlweCipher &) = delete;

  TrlweCipher(TrlweCipher &&obj);
  TrlweCipher &operator=(TrlweCipher &&obj);

  ~TrlweCipher();

  TorusInteger *get_pol_data(int i);
};

} // namespace thesis

#endif
