#include "thesis/trlwe_cipher.h"

namespace thesis {

TrlweCipher::TrlweCipher(int N, int k, double sdError, double varError)
    : Cipher(N * (k + 1), sdError, varError) {
  if (N < 2 || (N & (N - 1)) || k < 1)
    throw std::invalid_argument("N = 2^a with a > 0 ; k > 0");
  _N = N;
  _k = k;
}
TrlweCipher::TrlweCipher(TorusInteger *data, int size, int N, int k,
                         double sdError, double varError)
    : Cipher(data, size, sdError, varError) {
  if (N < 2 || (N & (N - 1)) || k < 1 || size < N * (k + 1))
    throw std::invalid_argument(
        "N = 2^a with a > 0 ; k > 0 ; size >= N * (k + 1)");
  _N = N;
  _k = k;
}
TrlweCipher::TrlweCipher(TrlweCipher &&obj) : Cipher(std::move(obj)) {
  _N = obj._N;
  _k = obj._k;
  obj._N = 0;
  obj._k = 0;
}
TrlweCipher &TrlweCipher::operator=(TrlweCipher &&obj) {
  Cipher::operator=(std::move(obj));
  _N = obj._N;
  _k = obj._k;
  obj._N = 0;
  obj._k = 0;
  return *this;
}
TrlweCipher::~TrlweCipher() {}
TorusInteger *TrlweCipher::get_pol_data(int i) {
  if (i < 0 || i > _k)
    return nullptr;
  return _data + _N * i;
}

} // namespace thesis
