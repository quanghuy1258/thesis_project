#include "thesis/cipher.h"
#include "thesis/memory_management.h"

namespace thesis {

Cipher::Cipher(size_t size, double sdError, double varError) {
  if (!size || sdError < 0 || varError < 0)
    throw std::invalid_argument("size > 0 ; sdError >= 0 ; varError >= 0");
  _data =
      (TorusInteger *)MemoryManagement::mallocMM(size * sizeof(TorusInteger));
  _size = size;
  _isOwnData = true;
  _sdError = sdError;
  _varError = varError;
  if (!_data)
    throw std::runtime_error("Cannot malloc data");
}
Cipher::Cipher(TorusInteger *data, size_t size, double sdError,
               double varError) {
  if (!data || !size || sdError < 0 || varError < 0)
    throw std::invalid_argument(
        "data != NULL ; size > 0 ; sdError >= 0 ; varError >= 0");
  _data = data;
  _size = size;
  _isOwnData = false;
  _sdError = sdError;
  _varError = varError;
}
Cipher::~Cipher() {
  if (_isOwnData)
    MemoryManagement::freeMM(_data);
}

} // namespace thesis
