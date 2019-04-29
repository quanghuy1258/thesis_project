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
  int _l;                // decomp length
  int _Bgbit;            // log_2(Bg)
  int _kpl;              // number of rows = (k+1)*l
  TorusInteger _Bg;      // decomposition base (must be a power of 2)
  TorusInteger _halfBg;  // Bg / 2
  TorusInteger _maskMod; // Bg - 1
  TorusInteger
      _offset; // Bg/2 * (1/Bg + 1/(Bg^2) + ... + 1/(Bg^l) + 1/(Bg^(l+1)))

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
  TrlweCipher get_trlwe(int r);
};

} // namespace thesis

#endif
