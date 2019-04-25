#include "thesis/trlwe_cipher.h"

namespace thesis {

TrlweCipher::TrlweCipher(size_t N, size_t k, double sdError, double varError)
    : Cipher(N * (k + 1), sdError, varError) {
  if (!N || !k)
    throw std::invalid_argument("N > 0 ; k > 0");
  _N = N;
  _k = k;
}
TrlweCipher::TrlweCipher(TorusInteger *data, size_t size, size_t N, size_t k,
                         double sdError, double varError)
    : Cipher(data, size, sdError, varError) {
  if (!N || !k || size < N * (k + 1))
    throw std::invalid_argument("N > 0 ; k > 0 ; size >= N * (k + 1)");
  _N = N;
  _k = k;
}
TrlweCipher::~TrlweCipher() {}

} // namespace thesis
