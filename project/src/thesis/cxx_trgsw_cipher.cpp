#include "thesis/trgsw_cipher.h"
#include "thesis/trlwe_cipher.h"

namespace thesis {

TrgswCipher::TrgswCipher(int N, int k, int l, int Bgbit, double sdError,
                         double varError)
    : Cipher(N * (k + 1) * l * (k + 1), sdError, varError) {
  if (N < 2 || (N & (N - 1)) || k < 1 || l < 1 || Bgbit < 1)
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
  if (N < 2 || (N & (N - 1)) || k < 1 || l < 1 || Bgbit < 1 ||
      size < N * (k + 1) * l * (k + 1))
    throw std::invalid_argument("N = 2^a with a > 0 ; k > 0 ; l > 0 ; Bgbit > "
                                "0 ; size >= N * (k + 1) * l * (k + 1)");
  _N = N;
  _k = k;
  _l = l;
  _Bgbit = Bgbit;
}
TrgswCipher::TrgswCipher(TrgswCipher &&obj) : Cipher(std::move(obj)) {
  _N = obj._N;
  _k = obj._k;
  _l = obj._l;
  _Bgbit = obj._Bgbit;
  obj._N = 0;
  obj._k = 0;
  obj._l = 0;
  obj._Bgbit = 0;
}
TrgswCipher &TrgswCipher::operator=(TrgswCipher &&obj) {
  Cipher::operator=(std::move(obj));
  _N = obj._N;
  _k = obj._k;
  _l = obj._l;
  _Bgbit = obj._Bgbit;
  obj._N = 0;
  obj._k = 0;
  obj._l = 0;
  obj._Bgbit = 0;
  return *this;
}
TrgswCipher::~TrgswCipher() {}
TorusInteger *TrgswCipher::get_trlwe_data(int r) {
  if (r < 0 || r >= (_k + 1) * _l)
    return nullptr;
  return _data + (_k + 1) * r * _N;
}
TorusInteger *TrgswCipher::get_pol_data(int r, int c) {
  if (r < 0 || r >= (_k + 1) * _l || c < 0 || c > _k)
    return nullptr;
  return _data + ((_k + 1) * r + c) * _N;
}
TrlweCipher TrgswCipher::get_trlwe(int r) {
  if (r < 0 || r >= (_k + 1) * _l)
    throw std::invalid_argument("0 <= r < (_k + 1) * _l");
  return TrlweCipher(get_trlwe_data(r), (_k + 1) * _N, _N, _k, _sdError,
                     _varError);
}

} // namespace thesis
