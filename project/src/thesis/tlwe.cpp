#include <Eigen/Core>

#include "thesis/random.h"
#include "thesis/threadpool.h"
#include "thesis/tlwe.h"

namespace thesis {

int Tlwe::_n = 500;
double Tlwe::_stddevError = std::sqrt(2. / CONST_PI) * pow(2., -15);

// Constructors
Tlwe::Tlwe() {}

// Destructor
Tlwe::~Tlwe() {}

// Get params
int Tlwe::get_n() { return _n; }
double Tlwe::get_stddevError() { return _stddevError; }

// Set attributes
void Tlwe::clear_s() { _s.clear(); }
void Tlwe::clear_ciphertexts() { _ciphertexts.clear(); }
void Tlwe::clear_plaintexts() { _plaintexts.clear(); }
bool Tlwe::set_s(const std::vector<Integer> &s) {
  if ((signed)s.size() != _n)
    return false;

  _s.resize(_n);
  for (int i = 0; i < _n; i++) {
    _s[i] = s[i];
  }

  return true;
}
void Tlwe::generate_s() {
  _s.resize(_n);
  for (int i = 0; i < _n; i++) {
    _s[i] = Random::getUniformInteger();
  }
}
bool Tlwe::addCiphertext(const std::vector<Torus> &cipher) {
  if ((signed)cipher.size() != _n + 1)
    return false;

  _ciphertexts.push_back(cipher);
  return true;
};
void Tlwe::addPlaintext(const bool &bit) { _plaintexts.push_back(bit); }

// Get attributes
bool Tlwe::get_s(std::vector<Integer> &s) const {
  if (_s.empty())
    return false;

  s.resize(_n);
  for (int i = 0; i < _n; i++) {
    s[i] = _s[i];
  }
  return true;
}
void Tlwe::get_ciphertexts(std::vector<std::vector<Torus>> &ciphertexts) const {
  ciphertexts = _ciphertexts;
}
void Tlwe::get_plaintexts(std::vector<bool> &plaintexts) const {
  plaintexts = _plaintexts;
}

// Utilities
bool Tlwe::encryptAll() {
  if (_s.empty())
    return false;
  if (_plaintexts.empty()) {
    _ciphertexts.clear();
    return true;
  } else {
    _ciphertexts.resize(_plaintexts.size());
    for (int i = 0; i < (signed)_plaintexts.size(); i++) {
      _ciphertexts[i].resize(_n + 1);
    }
  }
#ifdef USING_GPU
#else
  for (int i = 0; i < (signed)_plaintexts.size(); i++) {
    for (int j = 0; j < _n; j++) {
      _ciphertexts[i][j] = Random::getUniformTorus();
    }
    _ciphertexts[i][_n] = Random::getNormalTorus(0, _stddevError);
  }
  int numberThreads = ThreadPool::get_numberThreads();
  Eigen::Barrier barrier(numberThreads);
  for (int i = 0; i < numberThreads; i++) {
    ThreadPool::get_threadPool().Schedule([&, i]() {
      int s = (_plaintexts.size() * i) / numberThreads,
          e = (_plaintexts.size() * (i + 1)) / numberThreads;
      for (int j = s; j < e; j++) {
        for (int k = 0; k < _n; k++) {
          _ciphertexts[j][_n] += _ciphertexts[j][k] * _s[k];
        }
        int shift = (signed)sizeof(Torus) * 8 - 1;
        shift = (shift < 0) ? 0 : shift;
        Torus bit = 1;
        bit = bit << (unsigned)shift;
        _ciphertexts[j][_n] += ((_plaintexts[j]) ? bit : 0);
      }
      barrier.Notify();
    });
  }
  barrier.Wait();
#endif
  return true;
}
bool Tlwe::decryptAll() {
  if (_s.empty())
    return false;
  if (_ciphertexts.empty()) {
    _plaintexts.clear();
    return true;
  } else {
    _plaintexts.resize(_ciphertexts.size());
  }
#ifdef USING_GPU
#else
  std::vector<Torus> rowvector_decrypts(_ciphertexts.size());
  int numberThreads = ThreadPool::get_numberThreads();
  Eigen::Barrier barrier(numberThreads);
  for (int i = 0; i < numberThreads; i++) {
    ThreadPool::get_threadPool().Schedule([&, i]() {
      int s = (_ciphertexts.size() * i) / numberThreads,
          e = (_ciphertexts.size() * (i + 1)) / numberThreads;
      for (int j = s; j < e; j++) {
        rowvector_decrypts[j] = _ciphertexts[j][_n];
        for (int k = 0; k < _n; k++) {
          rowvector_decrypts[j] -= _ciphertexts[j][k] * _s[k];
        }
      }
      barrier.Notify();
    });
  }
  barrier.Wait();
  for (int i = 0; i < (signed)_ciphertexts.size(); i++) {
    int shift = (signed)sizeof(Torus) * 8 - 2;
    shift = (shift < 0) ? 0 : shift;
    Torus code = (rowvector_decrypts[i] >> (unsigned)shift) & 3;
    _plaintexts[i] = (code == 1 || code == 2);
  }
#endif
  return true;
}

} // namespace thesis
