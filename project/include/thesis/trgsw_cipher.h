#ifndef TRGSW_CIPHER_H
#define TRGSW_CIPHER_H

#include "thesis/cipher.h"
#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class TrgswCipher : public Cipher {
public:
  int _N;
  int _k;
  int _l;
  int _Bgbit;

  TrgswCipher() = delete;
  TrgswCipher(const TrgswCipher &) = delete;
  TrgswCipher(int N, int k, int l, int Bgbit, double sdError, double varError);
  TrgswCipher(TorusInteger *data, int size, int N, int k, int l, int Bgbit,
              double sdError, double varError);

  TrgswCipher &operator=(const TrgswCipher &) = delete;

  TrgswCipher(TrgswCipher &&obj);
  TrgswCipher &operator=(TrgswCipher &&obj);

  ~TrgswCipher();

  TorusInteger *get_trlwe_data(int r);
  TorusInteger *get_pol_data(int r, int c);
};

} // namespace thesis

#endif
