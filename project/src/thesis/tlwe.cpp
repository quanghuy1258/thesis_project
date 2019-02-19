#include "thesis/tlwe.h"

namespace thesis {

INTEGER Tlwe::_n = 500;
REAL Tlwe::_sdError = std::sqrt(2. / CONST_PI) * pow(2., -15);
REAL Tlwe::_sdMaximumError = std::sqrt(2. / CONST_PI) * 9.e-9;

// Constructors
Tlwe::Tlwe() {}

// Destructor
Tlwe::~Tlwe() {}

// Get params
INTEGER Tlwe::get_n() { return _n; }
REAL Tlwe::get_sdError() { return _sdError; }
REAL Tlwe::get_sdMaximumError() { return _sdMaximumError; }

// Set attributes
void Tlwe::clear_s() { _s.clear(); }
void Tlwe::clear_ciphertexts() { _ciphertexts.clear(); }
void Tlwe::clear_plaintexts() { _plaintexts.clear(); }
bool Tlwe::set_s(const std::vector<INTEGER> &s) {
  if ((signed)s.size() != _n)
    return false;

  _s.resize(_n);
  for (INTEGER i = 0; i < _n; i++) {
    _s[i] = s[i];
  }

  return true;
}
bool Tlwe::addCiphertext(const std::vector<INTEGER> &cipher) {
  if ((signed)cipher.size() != _n + 1)
    return false;

  _ciphertexts.push_back(cipher);
  return true;
};
void Tlwe::addPlaintext(const bool &bit) { _plaintexts.push_back(bit); }

// Get attributes
bool Tlwe::get_s(std::vector<INTEGER> &s) const {
  if ((signed)_s.size() == 0) {
    return false;
  }

  s.resize(_n);
  for (INTEGER i = 0; i < _n; i++) {
    s[i] = _s[i];
  }
  return true;
}
void Tlwe::get_ciphertexts(
    std::vector<std::vector<INTEGER>> &ciphertexts) const {
  ciphertexts = _ciphertexts;
}
void Tlwe::get_plaintexts(std::vector<bool> &plaintexts) const {
  plaintexts = _plaintexts;
}

// Utilities
void Tlwe::encryptAll() { clear_ciphertexts(); }
void Tlwe::decryptAll() { clear_plaintexts(); }

} // namespace thesis
