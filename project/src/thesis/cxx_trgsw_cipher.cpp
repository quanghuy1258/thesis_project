#include "thesis/trgsw_cipher.h"

namespace thesis {

TrgswCipher::TrgswCipher(size_t N, size_t k, size_t l, size_t Bgbit,
                         double sdError, double varError)
    : Cipher(N * (k + 1) * l * (k + 1), sdError, varError) {
  if (!N || !k || !l || !Bgbit)
    throw std::invalid_argument("N > 0 ; k > 0 ; l > 0 ; Bgbit > 0");
  _N = N;
  _k = k;
  _l = l;
  _Bgbit = Bgbit;
}
TrgswCipher::TrgswCipher(TorusInteger *data, size_t size, size_t N, size_t k,
                         size_t l, size_t Bgbit, double sdError,
                         double varError)
    : Cipher(data, size, sdError, varError) {
  if (!N || !k || !l || !Bgbit || size < N * (k + 1) * l * (k + 1))
    throw std::invalid_argument("N > 0 ; k > 0 ; l > 0 ; Bgbit > 0 ; size >= N "
                                "* (k + 1) * l * (k + 1)");
  _N = N;
  _k = k;
  _l = l;
  _Bgbit = Bgbit;
}
TrgswCipher::~TrgswCipher() {}

} // namespace thesis
