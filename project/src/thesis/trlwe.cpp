#include "thesis/trlwe.h"
#include "thesis/fft.h"
#include "thesis/random.h"
#include "thesis/threadpool.h"

namespace thesis {

// Constructors
Trlwe::Trlwe() {
  _N = 1024; // pow(2, 10)
  _k = 1;
  _alpha = std::sqrt(2. / CONST_PI) * pow(2., -15);
}

// Destructor
Trlwe::~Trlwe() {}

// Get params
int Trlwe::get_N() const { return _N; }
int Trlwe::get_k() const { return _k; }
double Trlwe::get_alpha() const { return _alpha; }

// Set attributes
void Trlwe::clear_s() { _s.clear(); }
void Trlwe::clear_ciphertexts() { _ciphertexts.clear(); }
void Trlwe::clear_plaintexts() { _plaintexts.clear(); }
bool Trlwe::set_s(const std::vector<PolynomialBinary> &s) {
  if ((signed)s.size() != _k)
    return false;
  for (int i = 0; i < _k; i++) {
    if ((signed)s[i].size() != _N)
      return false;
  }
  _s = s;
  return true;
}
void Trlwe::generate_s() {
  _s.resize(_k);
  for (int i = 0; i < _k; i++) {
    _s[i].resize(_N);
    for (int j = 0; j < _N; j++) {
      _s[i][j] = (Random::getUniformInteger() % 2 == 1) ? true : false;
    }
  }
}
bool Trlwe::addCiphertext(const std::vector<PolynomialTorus> &cipher) {
  if ((signed)cipher.size() != _k + 1)
    return false;
  for (int i = 0; i <= _k; i++) {
    if ((signed)cipher[i].size() != _N)
      return false;
  }
  _ciphertexts.push_back(cipher);
  return true;
}
bool Trlwe::addPlaintext(const PolynomialBinary &plain) {
  if ((signed)plain.size() != _N)
    return false;
  _plaintexts.push_back(plain);
  return true;
}

// Get attributes
bool Trlwe::get_s(std::vector<PolynomialBinary> &s) const {
  if ((signed)_s.size() == 0)
    return false;
  s = _s;
  return true;
}
void Trlwe::get_ciphertexts(
    std::vector<std::vector<PolynomialTorus>> &ciphertexts) const {
  ciphertexts = _ciphertexts;
}
void Trlwe::get_plaintexts(std::vector<PolynomialBinary> &plaintexts) const {
  plaintexts = _plaintexts;
}

// Utilities
bool Trlwe::encryptAll() {
  if (_s.empty())
    return false;
  if (_plaintexts.empty()) {
    _ciphertexts.clear();
    return true;
  } else {
    _ciphertexts.resize(_plaintexts.size());
    for (int i = 0; i < (signed)_plaintexts.size(); i++) {
      _ciphertexts[i].resize(_k + 1);
      for (int j = 0; j <= _k; j++) {
        _ciphertexts[i][j].resize(_N);
        if (j != _k) {
          // _ciphertexts[i][j] uniform polynomial
          for (int k = 0; k < _N; k++) {
            _ciphertexts[i][j][k] = Random::getUniformTorus();
          }
        } else {
          // _ciphertexts[i][_k] Gaussian polynomial
          for (int k = 0; k < _N; k++) {
            _ciphertexts[i][_k][k] = Random::getNormalTorus(0, _alpha);
          }
        }
      }
    }
  }
#ifdef USING_GPU
#else
  int numberThreads = ThreadPool::get_numberThreads();
  Eigen::Barrier barrier(numberThreads);
  std::unique_ptr<FFT[]> fftCalculators(new FFT[numberThreads]);
  for (int i = 0; i < numberThreads; i++)
    fftCalculators[i].set_N(_N);
  for (int i = 0; i < numberThreads; i++) {
    ThreadPool::get_threadPool().Schedule([&, i]() {
      PolynomialTorus productTorusPolynomial;
      int s = (_plaintexts.size() * i) / numberThreads,
          e = (_plaintexts.size() * (i + 1)) / numberThreads;
      int shift = (signed)sizeof(Torus) * 8 - 1;
      shift = (shift < 0) ? 0 : shift;
      Torus bit = 1;
      bit = bit << (unsigned)shift;
      for (int j = s; j < e; j++) {
        for (int k = 0; k < _k; k++) {
          fftCalculators[i].torusPolynomialMultiplication(
              productTorusPolynomial, _s[k], _ciphertexts[j][k]);
          for (int l = 0; l < _N; l++) {
            _ciphertexts[j][_k][l] += productTorusPolynomial[l];
          }
        }
        for (int l = 0; l < _N; l++) {
          _ciphertexts[j][_k][l] += ((_plaintexts[j][l]) ? bit : 0);
        }
      }
      barrier.Notify();
    });
  }
  barrier.Wait();
#endif
  return true;
}
bool Trlwe::decryptAll() {
  if (_s.empty())
    return false;
  if (_ciphertexts.empty()) {
    _plaintexts.clear();
    return true;
  } else {
    _plaintexts.resize(_ciphertexts.size());
    for (int i = 0; i < (signed)_ciphertexts.size(); i++) {
      _plaintexts[i].resize(_N);
    }
  }
#ifdef USING_GPU
#else
  int numberThreads = ThreadPool::get_numberThreads();
  Eigen::Barrier barrier(numberThreads);
  std::unique_ptr<FFT[]> fftCalculators(new FFT[numberThreads]);
  for (int i = 0; i < numberThreads; i++)
    fftCalculators[i].set_N(_N);
  for (int i = 0; i < numberThreads; i++) {
    ThreadPool::get_threadPool().Schedule([&, i]() {
      PolynomialTorus productTorusPolynomial, decryptTorusPolynomial;
      int s = (_ciphertexts.size() * i) / numberThreads,
          e = (_ciphertexts.size() * (i + 1)) / numberThreads;
      int shift = (signed)sizeof(Torus) * 8 - 2;
      shift = (shift < 0) ? 0 : shift;
      for (int j = s; j < e; j++) {
        decryptTorusPolynomial = _ciphertexts[j][_k];
        for (int k = 0; k < _k; k++) {
          fftCalculators[i].torusPolynomialMultiplication(
              productTorusPolynomial, _s[k], _ciphertexts[j][k]);
          for (int l = 0; l < _N; l++) {
            decryptTorusPolynomial[l] -= productTorusPolynomial[l];
          }
        }
        for (int l = 0; l < _N; l++) {
          Torus code = (decryptTorusPolynomial[l] >> (unsigned)shift) & 3;
          _plaintexts[j][l] = (code == 1 || code == 2);
        }
      }
      barrier.Notify();
    });
  }
  barrier.Wait();
#endif
  return true;
}
bool Trlwe::tlweExtractAll(Tlwe &out) const {
  if (out.get_n() != (_N * _k))
    return false;
  for (int i = 0; i < (signed)_ciphertexts.size(); i++) {
    for (int j = 0; j < _N; j++) {
      std::vector<Torus> cipher(_N * _k + 1);
      for (int k = 0; k < _N * _k; k++) {
        cipher[k] = (j >= k % _N) ? (_ciphertexts[i][k / _N][j - k % _N]) : 0;
      }
      cipher[_N * _k] = _ciphertexts[i][_k][j];
      out.addCiphertext(cipher);
    }
  }
  return true;
}
bool Trlwe::tlweExtractOne(Tlwe &out, int p, int cipherID) const {
  if (out.get_n() != (_N * _k) || p < 0 || p >= _N || cipherID < 0 ||
      cipherID >= (signed)_ciphertexts.size())
    return false;
  std::vector<Torus> cipher(_N * _k + 1);
  for (int i = 0; i < _N * _k; i++) {
    cipher[i] =
        (p >= i % _N) ? (_ciphertexts[cipherID][i / _N][p - i % _N]) : 0;
  }
  cipher[_N * _k] = _ciphertexts[cipherID][_k][p];
  out.addCiphertext(cipher);
  return true;
}

} // namespace thesis
