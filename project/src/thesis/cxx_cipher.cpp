#include "thesis/cipher.h"
#include "thesis/memory_management.h"

namespace thesis {

Cipher::Cipher(int size, double sdError, double varError) {
  if (size < 1 || sdError < 0 || varError < 0)
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
Cipher::Cipher(TorusInteger *data, int size, double sdError, double varError) {
  if (!data || size < 1 || sdError < 0 || varError < 0)
    throw std::invalid_argument(
        "data != NULL ; size > 0 ; sdError >= 0 ; varError >= 0");
  _data = data;
  _size = size;
  _isOwnData = false;
  _sdError = sdError;
  _varError = varError;
}
Cipher::Cipher(Cipher &&obj) {
  _data = obj._data;
  _size = obj._size;
  _isOwnData = obj._isOwnData;
  _sdError = obj._sdError;
  _varError = obj._varError;
  obj._data = nullptr;
  obj._size = 0;
  obj._isOwnData = false;
  obj._sdError = 0;
  obj._varError = 0;
}
Cipher &Cipher::operator=(Cipher &&obj) {
  if (_isOwnData)
    MemoryManagement::freeMM(_data);
  _data = obj._data;
  _size = obj._size;
  _isOwnData = obj._isOwnData;
  _sdError = obj._sdError;
  _varError = obj._varError;
  obj._data = nullptr;
  obj._size = 0;
  obj._isOwnData = false;
  obj._sdError = 0;
  obj._varError = 0;
  return *this;
}
Cipher::~Cipher() {
  if (_isOwnData)
    MemoryManagement::freeMM(_data);
}

} // namespace thesis
