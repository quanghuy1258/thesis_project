#include "thesis/tlwe.h"
#include "thesis/random.h"

namespace thesis {
/*
static const double STDDEV_ERROR = std::sqrt(2. / CONST_PI) * pow(2., -15);

// Constructors
Tlwe::Tlwe() { _n = 500; }

// Destructor
Tlwe::~Tlwe() {}

// Get params
int Tlwe::get_n() const { return _n; }

// Set params
bool Tlwe::set_n(int n, bool isForcedToCheck) {
  if (isForcedToCheck && n < 1)
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
bool Tlwe::set_s(const std::vector<bool> &s, bool isForcedToCheck) {
  if (isForcedToCheck) {
    const int s_size = s.size();
    if (s_size != _n)
      return false;
  }
  _s = s;
  return true;
}
bool Tlwe::moveTo_s(std::vector<bool> &s, bool isForcedToCheck) {
  if (isForcedToCheck) {
    const int s_size = s.size();
    if (s_size != _n)
      return false;
  }
  _s = std::move(s);
  return true;
}
bool Tlwe::generate_s(bool isForcedToCheck) {
  if (isForcedToCheck && _n < 1)
    return false;
  _s.resize(_n);
  for (int i = 0; i < _n; i++)
    _s[i] = Random::getUniformInteger() & 1;
  return true;
}
bool Tlwe::addCiphertext(const std::vector<Torus> &cipher, double stddevError,
                         double varianceError, bool isForcedToCheck) {
  if (isForcedToCheck) {
    const int cipher_size = cipher.size();
    if (cipher_size != _n + 1 || stddevError < 0 || varianceError < 0)
      return false;
  }
  _ciphertexts.push_back(cipher);
  _stddevErrors.push_back(stddevError);
  _varianceErrors.push_back(varianceError);
  return true;
};
bool Tlwe::moveCiphertext(std::vector<Torus> &cipher, double stddevError,
                          double varianceError, bool isForcedToCheck) {
  if (isForcedToCheck) {
    const int cipher_size = cipher.size();
    if (cipher_size != _n + 1 || stddevError < 0 || varianceError < 0)
      return false;
  }
  _ciphertexts.push_back(std::move(cipher));
  _stddevErrors.push_back(stddevError);
  _varianceErrors.push_back(varianceError);
  return true;
}
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
bool Tlwe::encryptAll(bool isForcedToCheck) {
  if (isForcedToCheck && _s.empty()) {
    return false;
  }
  const int _plaintexts_size = _plaintexts.size();
  if (_plaintexts_size == 0) {
    clear_ciphertexts();
    return true;
  } else {
    _ciphertexts.resize(_plaintexts_size);
    _stddevErrors.resize(_plaintexts_size);
    _varianceErrors.resize(_plaintexts_size);
    for (int i = 0; i < _plaintexts_size; i++) {
      _ciphertexts[i].resize(_n + 1);
      _stddevErrors[i] = STDDEV_ERROR;
      _varianceErrors[i] = STDDEV_ERROR * STDDEV_ERROR;
      for (int j = 0; j < _n; j++)
        _ciphertexts[i][j] = Random::getUniformTorus();
      _ciphertexts[i][_n] = Random::getNormalTorus(0, STDDEV_ERROR);
    }
  }
  const int numberThreads = ThreadPool::get_numberThreads();
  Eigen::Barrier barrier(numberThreads);
  std::vector<std::vector<Torus>> encrypts(numberThreads);
  for (int i = 0; i < numberThreads; i++) {
    ThreadPool::get_threadPool().Schedule(
        [this, i, &encrypts, _plaintexts_size, numberThreads, &barrier]() {
          encrypts[i].resize(_plaintexts_size, 0);
          Torus bit = 1;
          bit <<= sizeof(Torus) * 8 - 1;
          int s = (_plaintexts_size * (_n + 1) * i) / numberThreads,
              e = (_plaintexts_size * (_n + 1) * (i + 1)) / numberThreads;
          for (int it = s; it < e; it++) {
            int j = it / (_n + 1);
            int k = it % (_n + 1);
            if (k == _n) {
              encrypts[i][j] += _plaintexts[j] ? bit : 0;
            } else {
              encrypts[i][j] += _s[k] ? _ciphertexts[j][k] : 0;
            }
          }
          barrier.Notify();
        });
  }
  barrier.Wait();
  for (int i = 0; i < _plaintexts_size; i++) {
    for (int j = 0; j < numberThreads; j++)
      _ciphertexts[i][_n] += encrypts[j][i];
  }
  return true;
}
bool Tlwe::decryptAll(bool isForcedToCheck) {
  if (isForcedToCheck && _s.empty())
    return false;
  const int _ciphertexts_size = _ciphertexts.size();
  if (_ciphertexts_size == 0) {
    clear_plaintexts();
    return true;
  } else {
    _plaintexts.resize(_ciphertexts_size);
  }
  const int numberThreads = ThreadPool::get_numberThreads();
  Eigen::Barrier barrier(numberThreads);
  std::vector<std::vector<Torus>> decrypts(numberThreads);
  for (int i = 0; i < numberThreads; i++) {
    ThreadPool::get_threadPool().Schedule(
        [this, i, &decrypts, _ciphertexts_size, numberThreads, &barrier]() {
          decrypts[i].resize(_ciphertexts_size, 0);
          int s = (_ciphertexts_size * (_n + 1) * i) / numberThreads,
              e = (_ciphertexts_size * (_n + 1) * (i + 1)) / numberThreads;
          for (int it = s; it < e; it++) {
            int j = it / (_n + 1);
            int k = it % (_n + 1);
            if (k == _n) {
              decrypts[i][j] += _ciphertexts[j][_n];
            } else {
              decrypts[i][j] -= _s[k] ? _ciphertexts[j][k] : 0;
            }
          }
          barrier.Notify();
        });
  }
  barrier.Wait();
  for (int i = 0; i < _ciphertexts_size; i++) {
    Torus bit_torus = 0;
    for (int j = 0; j < numberThreads; j++)
      bit_torus += decrypts[j][i];
    double bit_double = bit_torus;
    _plaintexts[i] =
        std::llround(bit_double / std::pow(2, sizeof(Torus) * 8 - 1)) & 1;
  }
  return true;
}
bool Tlwe::getAllErrorsForDebugging(std::vector<double> &errors,
                                    const std::vector<bool> &expectedPlaintexts,
                                    bool isForcedToCheck) const {
  const int _ciphertexts_size = _ciphertexts.size();
  const int expectedPlaintexts_size = expectedPlaintexts.size();
  if (isForcedToCheck &&
      (_s.empty() || _ciphertexts_size != expectedPlaintexts_size))
    return false;
  if (_ciphertexts_size == 0) {
    errors.clear();
    return true;
  } else {
    errors.resize(_ciphertexts_size, 0);
  }
  const int numberThreads = ThreadPool::get_numberThreads();
  Eigen::Barrier barrier(numberThreads);
  std::vector<std::vector<Torus>> decrypt_errors(numberThreads);
  for (int i = 0; i < numberThreads; i++) {
    ThreadPool::get_threadPool().Schedule([this, i, &decrypt_errors,
                                           _ciphertexts_size, numberThreads,
                                           &expectedPlaintexts, &barrier]() {
      decrypt_errors[i].resize(_ciphertexts_size);
      Torus bit = 1;
      bit <<= sizeof(Torus) * 8 - 1;
      int s = (_ciphertexts_size * (_n + 1) * i) / numberThreads,
          e = (_ciphertexts_size * (_n + 1) * (i + 1)) / numberThreads;
      for (int it = s; it < e; it++) {
        int j = it / (_n + 1);
        int k = it % (_n + 1);
        if (k == _n) {
          decrypt_errors[i][j] += _ciphertexts[j][_n];
          decrypt_errors[i][j] -= (expectedPlaintexts[j]) ? bit : 0;
        } else {
          decrypt_errors[i][j] -= _s[k] ? _ciphertexts[j][k] : 0;
        }
      }
      barrier.Notify();
    });
  }
  barrier.Wait();
  for (int i = 0; i < _ciphertexts_size; i++) {
    Torus error = 0;
    for (int j = 0; j < numberThreads; j++)
      error += decrypt_errors[j][i];
    errors[i] = std::abs(error / std::pow(2, sizeof(Torus) * 8));
  }
  return true;
}
bool Tlwe::initPublicKeySwitching(const std::vector<bool> &key, int t,
                                  bool isForcedToCheck) {
  if (isForcedToCheck &&
      (key.empty() || _s.empty() || t < 1 || t > (signed)sizeof(Torus) * 8))
    return false;
  const int key_size = key.size();
  _plaintexts.resize(key_size * t);
  std::fill(_plaintexts.begin(), _plaintexts.end(), false);
  encryptAll(false);
  clear_plaintexts();
  for (int i = 0; i < key_size; i++) {
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
*/
} // namespace thesis
