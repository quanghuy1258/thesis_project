#ifndef TRGSW_CIPHER_H
#define TRGSW_CIPHER_H

#include "thesis/cipher.h"
#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class TrgswCipher : public Cipher {
public:
  size_t _N;
  size_t _k;
  size_t _l;
  size_t _Bgbit;

  TrgswCipher() = delete;
  TrgswCipher(const TrgswCipher &) = delete;
  TrgswCipher(size_t N, size_t k, size_t l, size_t Bgbit, double sdError,
              double varError);
  TrgswCipher(TorusInteger *data, size_t size, size_t N, size_t k, size_t l,
              size_t Bgbit, double sdError, double varError);

  TrgswCipher &operator=(const TrgswCipher &) = delete;

  ~TrgswCipher();
};

} // namespace thesis

#endif
