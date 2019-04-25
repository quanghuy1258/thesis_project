#include "thesis/trgsw_cipher.h"

namespace thesis {

TrgswCipher::TrgswCipher(int N, int k, int l, int Bgbit, double sdError,
                         double varError)
    : Cipher(N * (k + 1) * l * (k + 1), sdError, varError) {
  if (N < 2 || (N & (N - 1)) || k <= 0 || l <= 0 || Bgbit <= 0)
    throw std::invalid_argument(
        "N = 2^a with a > 0 ; k > 0 ; l > 0 ; Bgbit > 0");
  _N = N;
  _k = k;
  _l = l;
  _Bgbit = Bgbit;
}
TrgswCipher::TrgswCipher(TorusInteger *data, int size, int N, int k, int l,
                         int Bgbit, double sdError, double varError)
    : Cipher(data, size, sdError, varError) {
  if (N < 2 || (N & (N - 1)) || k <= 0 || l <= 0 || Bgbit <= 0 ||
      size < N * (k + 1) * l * (k + 1))
    throw std::invalid_argument("N = 2^a with a > 0 ; k > 0 ; l > 0 ; Bgbit > "
                                "0 ; size >= N * (k + 1) * l * (k + 1)");
  _N = N;
  _k = k;
  _l = l;
  _Bgbit = Bgbit;
}
TrgswCipher::~TrgswCipher() {}

} // namespace thesis
