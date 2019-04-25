#include "thesis/tlwe_cipher.h"

namespace thesis {

TlweCipher::TlweCipher(size_t n, double sdError, double varError)
    : Cipher(n + 1, sdError, varError) {
  if (!n)
    throw std::invalid_argument("n > 0");
  _n = n;
}
TlweCipher::TlweCipher(TorusInteger *data, size_t size, size_t n,
                       double sdError, double varError)
    : Cipher(data, size, sdError, varError) {
  if (!n || size < n + 1)
    throw std::invalid_argument("n > 0 ; n + 1 >= size");
  _n = n;
}
TlweCipher::~TlweCipher() {}

} // namespace thesis
