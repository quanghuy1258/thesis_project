#include "thesis/trgsw.h"
#include "thesis/fft.h"
#include "thesis/random.h"
#include "thesis/threadpool.h"
#include "thesis/trlwe.h"

namespace thesis {

// Constructors
Trgsw::Trgsw() {
  _l = 2;
  _Bgbit = 10; // Bg = 1024
  // Similar to TRLWE
  _N = 1024;
  _k = 1;
  _alpha = std::sqrt(2. / CONST_PI) * pow(2., -15);
}

// Destructor
Trgsw::~Trgsw() {}

// Get params
int Trgsw::get_l() const { return _l; }
int Trgsw::get_Bgbit() const { return _Bgbit; }
int Trgsw::get_N() const { return _N; }
int Trgsw::get_k() const { return _k; }
int Trgsw::get_alpha() const { return _alpha; }

// Set attributes
void Trgsw::clear_s() { _s.clear(); }
void Trgsw::clear_ciphertexts() { _ciphertexts.clear(); }
void Trgsw::clear_plaintexts() { _plaintexts.clear(); }
bool Trgsw::set_s(const std::vector<PolynomialBinary> &s) {
  if ((signed)s.size() != _k)
    return false;
  for (int i = 0; i < _k; i++) {
    if ((signed)s[i].size() != _N)
      return false;
  }
  _s = s;
  return true;
}
void Trgsw::generate_s() {
  _s.resize(_k);
  for (int i = 0; i < _k; i++) {
    _s[i].resize(_N);
    for (int j = 0; j < _N; j++) {
      _s[i][j] = (Random::getUniformInteger() % 2 == 1) ? true : false;
    }
  }
}
bool Trgsw::addCiphertext(const std::vector<PolynomialTorus> &cipher) {
  if ((signed)cipher.size() != (_k + 1) * _l * (_k + 1))
    return false;
  for (int i = 0; i < (_k + 1) * _l * (_k + 1); i++) {
    if ((signed)cipher[i].size() != _N)
      return false;
  }
  _ciphertexts.push_back(cipher);
  return true;
}
bool Trgsw::addPlaintext(const PolynomialInteger &plain) {
  if ((signed)plain.size() != _N)
    return false;
  _plaintexts.push_back(plain);
  return true;
}

// Get attributes
bool Trgsw::get_s(std::vector<PolynomialBinary> &s) const {
  if ((signed)_s.size() == 0)
    return false;
  s = _s;
  return true;
}
void Trgsw::get_ciphertexts(
    std::vector<std::vector<PolynomialTorus>> &ciphertexts) const {
  ciphertexts = _ciphertexts;
}
void Trgsw::get_plaintexts(std::vector<PolynomialInteger> &plaintexts) const {
  plaintexts = _plaintexts;
}

// Utilities
bool Trgsw::encryptAll() {
  if (_s.empty())
    return false;
  if (_plaintexts.empty()) {
    _ciphertexts.clear();
    return true;
  } else {
    _ciphertexts.resize(_plaintexts.size());
    for (int i = 0; i < (signed)_plaintexts.size(); i++) {
      _ciphertexts[i].resize((_k + 1) * _l * (_k + 1));
      for (int j = 0; j < (_k + 1) * _l * (_k + 1); j++) {
        _ciphertexts[i][j].resize(_N);
        if (j % (_k + 1) != _k) {
          // _ciphertexts[i][j] uniform polynomial
          for (int k = 0; k < _N; k++) {
            _ciphertexts[i][j][k] = Random::getUniformTorus();
          }
        } else {
          // _ciphertexts[i][_k (mod _k+1)] Gaussian polynomial
          for (int k = 0; k < _N; k++) {
            _ciphertexts[i][j][k] = Random::getNormalTorus(0, _alpha);
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
      int s = (_plaintexts.size() * (_k + 1) * _l * i) / numberThreads,
          e = (_plaintexts.size() * (_k + 1) * _l * (i + 1)) / numberThreads;
      for (int j = s; j < e; j++) {
        int plainID = j / ((_k + 1) * _l);
        int rowID = j % ((_k + 1) * _l);
        int blockID = rowID / _l;
        int rowIdInBlock = rowID % _l;
        for (int k = 0; k < _k; k++) {
          fftCalculators[i].torusPolynomialMultiplication(
              productTorusPolynomial, _s[k],
              _ciphertexts[plainID][rowID * (_k + 1) + k]);
          for (int l = 0; l < _N; l++) {
            _ciphertexts[plainID][rowID * (_k + 1) + _k][l] +=
                productTorusPolynomial[l];
          }
        }
        for (int l = 0; l < _N; l++) {
          int bits = sizeof(Integer) * 8 - _Bgbit * (rowIdInBlock + 1);
          if (bits >= 0)
            _ciphertexts[plainID][rowID * (_k + 1) + blockID][l] +=
                _plaintexts[plainID][l] << bits;
          else
            _ciphertexts[plainID][rowID * (_k + 1) + blockID][l] +=
                _plaintexts[plainID][l] >> bits;
        }
      }
      barrier.Notify();
    });
  }
  barrier.Wait();
#endif
  return true;
}
bool Trgsw::decryptAll() {
  if (_s.empty())
    return false;
  if (_ciphertexts.empty()) {
    _plaintexts.clear();
    return true;
  } else {
    _plaintexts.resize(_ciphertexts.size());
    for (int i = 0; i < (signed)_ciphertexts.size(); i++) {
      _plaintexts[i].resize(_N);
      std::fill(_plaintexts[i].begin(), _plaintexts[i].end(), 0);
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
      int shift = sizeof(Integer) * 8 - _Bgbit - 1;
      shift = (shift < 0) ? 0 : shift;
#if defined(USING_32BIT)
      uint32_t decrypt_integer, mask = 1;
#else
      uint64_t decrypt_integer, mask = 1;
#endif
      mask = mask << _Bgbit;
      mask = mask - 1;
      PolynomialTorus productTorusPolynomial;
      std::vector<PolynomialTorus> decrypts;
      decrypts.resize(_l);
      int s = (_ciphertexts.size() * i) / numberThreads,
          e = (_ciphertexts.size() * (i + 1)) / numberThreads;
      for (int j = s; j < e; j++) {
        for (int k = 0; k < _l; k++) {
          decrypts[k] = _ciphertexts[j][(_k * _l + k) * (_k + 1) + _k];
          for (int l = 0; l < _k; l++) {
            fftCalculators[i].torusPolynomialMultiplication(
                productTorusPolynomial, _s[l],
                _ciphertexts[j][(_k * _l + k) * (_k + 1) + l]);
            for (int m = 0; m < _N; m++) {
              decrypts[k][m] -= productTorusPolynomial[m];
            }
          }
          int bits = sizeof(Integer) * 8 - _Bgbit * (k + 1);
          for (int l = 0; l < _N; l++) {
            if (bits >= 0)
              decrypts[k][l] -= _plaintexts[j][l] << bits;
            else
              decrypts[k][l] -= _plaintexts[j][l] >> (-bits);
            decrypt_integer = decrypts[k][l];
            decrypt_integer = (decrypt_integer >> shift);
            double decrypt_real = decrypt_integer;
            decrypt_real /= 2;
            decrypt_integer = std::llround(decrypt_real);
            decrypt_integer = decrypt_integer & mask;
            _plaintexts[j][l] += (decrypt_integer << (_Bgbit * k));
          }
        }
      }
      barrier.Notify();
    });
  }
  barrier.Wait();
#endif
  return true;
}
bool Trgsw::decompositeAll(std::vector<std::vector<PolynomialInteger>> &out,
                           const Trlwe &inp) const {
  if (_N != inp._N || _k != inp._k)
    return false;
  if (inp._ciphertexts.empty()) {
    out.clear();
    return true;
  } else {
    out.resize(inp._ciphertexts.size());
    for (int i = 0; i < (signed)inp._ciphertexts.size(); i++) {
      out[i].resize((_k + 1) * _l);
      for (int j = 0; j < (_k + 1) * _l; j++) {
        out[i][j].resize(_N);
      }
    }
  }
#ifdef USING_GPU
#else
  int numberThreads = ThreadPool::get_numberThreads();
  Eigen::Barrier barrier(numberThreads);
  for (int it = 0; it < numberThreads; it++) {
    ThreadPool::get_threadPool().Schedule([&, it]() {
      int s = (inp._ciphertexts.size() * (_k + 1) * _N * it) / numberThreads,
          e = (inp._ciphertexts.size() * (_k + 1) * _N * (it + 1)) /
              numberThreads;
      for (int newIt = s; newIt < e; newIt++) {
        int j = newIt % _N;
        int i = (newIt / _N) % (_k + 1);
        int cipherId = newIt / ((_k + 1) * _N);
        Torus value = inp._ciphertexts[cipherId][i][j], mask = 1;
        if ((signed)sizeof(Torus) * 8 > _Bgbit * _l) {
          mask <<= ((signed)sizeof(Torus) * 8 - _Bgbit * _l - 1);
          value += mask;
          mask = ~((mask << 1) - 1);
          value &= mask;
        }
        mask = 1;
        mask <<= _Bgbit;
        for (int p = _l - 1; p >= 0; p--) {
          Torus value_temp = value;
          int shift = sizeof(Torus) * 8 - _Bgbit * (p + 1);
          unsigned ushift = (shift < 0) ? (-shift) : shift;
          value_temp =
              (shift < 0) ? (value_temp << ushift) : (value_temp >> ushift);
          out[cipherId][i * _l + p][j] = value_temp % mask;
          if (out[cipherId][i * _l + p][j] < -mask / 2)
            out[cipherId][i * _l + p][j] += mask;
          if (out[cipherId][i * _l + p][j] >= mask / 2)
            out[cipherId][i * _l + p][j] -= mask;
          value -= (shift < 0) ? (out[cipherId][i * _l + p][j] >> ushift)
                               : (out[cipherId][i * _l + p][j] << ushift);
        }
      }
      barrier.Notify();
    });
  }
  barrier.Wait();
#endif
  return true;
}

} // namespace thesis
