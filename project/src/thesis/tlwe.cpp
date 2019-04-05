#include "thesis/tlwe.h"
#include "thesis/random.h"
#include "thesis/threadpool.h"

namespace thesis {

static const double STDDEV_ERROR = std::sqrt(2. / CONST_PI) * pow(2., -15);

// Constructors
Tlwe::Tlwe() { _n = 500; }

// Destructor
Tlwe::~Tlwe() {}

// Get params
int Tlwe::get_n() const { return _n; }

// Set params
bool Tlwe::set_n(int n) {
  if (n < 1)
    return false;
  _n = n;
  return true;
}

// Set attributes
void Tlwe::clear_s() { _s.clear(); }
void Tlwe::clear_ciphertexts() {
  _ciphertexts.clear();
  _stddevErrors.clear();
  _varianceErrors.clear();
}
void Tlwe::clear_plaintexts() { _plaintexts.clear(); }
bool Tlwe::set_s(const std::vector<bool> &s) {
  if ((signed)s.size() != _n)
    return false;
  _s = s;
  return true;
}
void Tlwe::generate_s() {
  _s.resize(_n);
  for (int i = 0; i < _n; i++) {
    _s[i] = (Random::getUniformInteger() % 2 == 1);
  }
}
bool Tlwe::addCiphertext(const std::vector<Torus> &cipher, double stddevError,
                         double varianceError) {
  if ((signed)cipher.size() != _n + 1)
    return false;
  _ciphertexts.push_back(cipher);
  _stddevErrors.push_back(stddevError);
  _varianceErrors.push_back(varianceError);
  return true;
};
void Tlwe::addPlaintext(bool bit) { _plaintexts.push_back(bit); }

// Get attributes
const std::vector<bool> &Tlwe::get_s() const { return _s; }
const std::vector<std::vector<Torus>> &Tlwe::get_ciphertexts() const {
  return _ciphertexts;
}
const std::vector<double> &Tlwe::get_stddevErrors() const {
  return _stddevErrors;
}
const std::vector<double> &Tlwe::get_varianceErrors() const {
  return _varianceErrors;
}
const std::vector<bool> &Tlwe::get_plaintexts() const { return _plaintexts; }

// Utilities
bool Tlwe::encryptAll() {
  if (_s.empty())
    return false;
  if (_plaintexts.empty()) {
    clear_ciphertexts();
    return true;
  } else {
    _ciphertexts.resize(_plaintexts.size());
    _stddevErrors.resize(_plaintexts.size());
    _varianceErrors.resize(_plaintexts.size());
    for (int i = 0; i < (signed)_plaintexts.size(); i++) {
      _ciphertexts[i].resize(_n + 1);
      _stddevErrors[i] = STDDEV_ERROR;
      _varianceErrors[i] = STDDEV_ERROR * STDDEV_ERROR;
      for (int j = 0; j < _n; j++) {
        _ciphertexts[i][j] = Random::getUniformTorus();
      }
      _ciphertexts[i][_n] = Random::getNormalTorus(0, STDDEV_ERROR);
    }
  }
  int numberThreads = ThreadPool::get_numberThreads();
  Eigen::Barrier barrier(numberThreads);
  for (int i = 0; i < numberThreads; i++) {
    ThreadPool::get_threadPool().Schedule([&, i]() {
      int s = (_plaintexts.size() * i) / numberThreads,
          e = (_plaintexts.size() * (i + 1)) / numberThreads;
      int shift = sizeof(Torus) * 8 - 1;
      Torus bit = 1;
      bit <<= shift;
      for (int j = s; j < e; j++) {
        for (int k = 0; k < _n; k++) {
          if (_s[k])
            _ciphertexts[j][_n] += _ciphertexts[j][k];
        }
        _ciphertexts[j][_n] += ((_plaintexts[j]) ? bit : 0);
      }
      barrier.Notify();
    });
  }
  barrier.Wait();
  return true;
}
bool Tlwe::decryptAll() {
  if (_s.empty())
    return false;
  if (_ciphertexts.empty()) {
    clear_plaintexts();
    return true;
  } else {
    _plaintexts.resize(_ciphertexts.size());
  }
  std::vector<Torus> decrypts(_ciphertexts.size());
  int numberThreads = ThreadPool::get_numberThreads();
  Eigen::Barrier barrier(numberThreads);
  for (int i = 0; i < numberThreads; i++) {
    ThreadPool::get_threadPool().Schedule([&, i]() {
      int s = (_ciphertexts.size() * i) / numberThreads,
          e = (_ciphertexts.size() * (i + 1)) / numberThreads;
      for (int j = s; j < e; j++) {
        decrypts[j] = _ciphertexts[j][_n];
        for (int k = 0; k < _n; k++) {
          if (_s[k])
            decrypts[j] -= _ciphertexts[j][k];
        }
      }
      barrier.Notify();
    });
  }
  barrier.Wait();
  for (int i = 0; i < (signed)_ciphertexts.size(); i++) {
    int bits = sizeof(Torus) * 8 - 2;
    decrypts[i] = ((decrypts[i] >> bits) & 3);
    _plaintexts[i] = ((decrypts[i] == 1) || (decrypts[i] == 2));
  }
  return true;
}
bool Tlwe::getAllErrorsForDebugging(
    std::vector<double> &errors,
    const std::vector<bool> &expectedPlaintexts) const {
  if (_s.empty() || _ciphertexts.size() != expectedPlaintexts.size())
    return false;
  if (_ciphertexts.empty()) {
    errors.clear();
    return true;
  } else {
    errors.resize(_ciphertexts.size());
  }
  int numberThreads = ThreadPool::get_numberThreads();
  Eigen::Barrier barrier(numberThreads);
  for (int i = 0; i < numberThreads; i++) {
    ThreadPool::get_threadPool().Schedule([&, i]() {
      Torus decrypt;
      int shift = sizeof(Torus) * 8 - 1;
      Torus bit = 1;
      bit <<= shift;
      int s = (_ciphertexts.size() * i) / numberThreads,
          e = (_ciphertexts.size() * (i + 1)) / numberThreads;
      for (int j = s; j < e; j++) {
        decrypt = _ciphertexts[j][_n] - ((expectedPlaintexts[j]) ? bit : 0);
        for (int k = 0; k < _n; k++) {
          if (_s[k])
            decrypt -= _ciphertexts[j][k];
        }
        errors[j] = std::abs(decrypt / std::pow(2, sizeof(Torus) * 8));
      }
      barrier.Notify();
    });
  }
  barrier.Wait();
  return true;
}
bool Tlwe::initPublicKeySwitching(const std::vector<bool> &key, int t) {
  if (key.empty() || _s.empty() || t < 1 || t > (signed)sizeof(Torus) * 8)
    return false;
  _plaintexts.resize(key.size() * t);
  std::fill(_plaintexts.begin(), _plaintexts.end(), false);
  encryptAll();
  clear_plaintexts();
  for (int i = 0; i < (signed)key.size(); i++) {
    if (!key[i])
      continue;
    for (int j = 0; j < t; j++) {
      Torus bit = 1;
      bit <<= (sizeof(Torus) * 8 - j - 1);
      _ciphertexts[i * t + j][_n] += bit;
    }
  }
  return true;
}

} // namespace thesis
