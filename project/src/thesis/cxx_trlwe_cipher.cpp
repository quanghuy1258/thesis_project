#include "thesis/trlwe_cipher.h"

namespace thesis {

TrlweCipher::TrlweCipher(int N, int k, double sdError, double varError)
    : Cipher(N * (k + 1), sdError, varError) {
  if (N < 2 || (N & (N - 1)) || k <= 0)
    throw std::invalid_argument("N = 2^a with a > 0 ; k > 0");
  _N = N;
  _k = k;
}
TrlweCipher::TrlweCipher(TorusInteger *data, int size, int N, int k,
                         double sdError, double varError)
    : Cipher(data, size, sdError, varError) {
  if (N < 2 || (N & (N - 1)) || k <= 0 || size < N * (k + 1))
    throw std::invalid_argument(
        "N = 2^a with a > 0 ; k > 0 ; size >= N * (k + 1)");
  _N = N;
  _k = k;
}
TrlweCipher::~TrlweCipher() {}

} // namespace thesis
