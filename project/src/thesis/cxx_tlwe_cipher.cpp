#include "thesis/tlwe_cipher.h"

namespace thesis {

TlweCipher::TlweCipher(int n, double sdError, double varError)
    : Cipher(n + 1, sdError, varError) {
  if (n < 1)
    throw std::invalid_argument("n > 0");
  _n = n;
}
TlweCipher::TlweCipher(TorusInteger *data, int size, int n, double sdError,
                       double varError)
    : Cipher(data, size, sdError, varError) {
  if (n < 1 || size < n + 1)
    throw std::invalid_argument("n > 0 ; n + 1 >= size");
  _n = n;
}
TlweCipher::~TlweCipher() {}

} // namespace thesis
