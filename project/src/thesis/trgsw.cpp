#include "thesis/trgsw.h"
#include "thesis/fft.h"
#include "thesis/random.h"
#include "thesis/threadpool.h"
#include "thesis/trlwe.h"

namespace thesis {

static const double STDDEV_ERROR = std::sqrt(2. / CONST_PI) * pow(2., -30);

// Constructors
Trgsw::Trgsw() {
  _l = 3;
  _Bgbit = 10; // Bg = 1024
  // Similar to TRLWE
  _N = 1024;
  _k = 1;
}

// Destructor
Trgsw::~Trgsw() {}

// Get params
int Trgsw::get_l() const { return _l; }
int Trgsw::get_Bgbit() const { return _Bgbit; }
int Trgsw::get_N() const { return _N; }
int Trgsw::get_k() const { return _k; }

// Set attributes
void Trgsw::clear_s() { _s.clear(); }
void Trgsw::clear_ciphertexts() {
  _ciphertexts.clear();
  _stddevErrors.clear();
  _varianceErrors.clear();
}
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
      _s[i][j] = (Random::getUniformInteger() % 2 == 1);
    }
  }
}
bool Trgsw::addCiphertext(const std::vector<PolynomialTorus> &cipher,
                          double stddevError, double varianceError) {
  if ((signed)cipher.size() != (_k + 1) * _l * (_k + 1))
    return false;
  for (int i = 0; i < (_k + 1) * _l * (_k + 1); i++) {
    if ((signed)cipher[i].size() != _N)
      return false;
  }
  _ciphertexts.push_back(cipher);
  _stddevErrors.push_back(stddevError);
  _varianceErrors.push_back(varianceError);
  return true;
}
void Trgsw::addPlaintext(bool plain) { _plaintexts.push_back(plain); }

// Get attributes
const std::vector<PolynomialBinary> &Trgsw::get_s() const { return _s; }
const std::vector<std::vector<PolynomialTorus>> &
Trgsw::get_ciphertexts() const {
  return _ciphertexts;
}
const std::vector<double> &Trgsw::get_stddevErrors() const {
  return _stddevErrors;
}
const std::vector<double> &Trgsw::get_varianceErrors() const {
  return _varianceErrors;
}
const std::vector<bool> &Trgsw::get_plaintexts() const { return _plaintexts; }

// Utilities
bool Trgsw::encryptAll() {
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
      _ciphertexts[i].resize((_k + 1) * _l * (_k + 1));
      _stddevErrors[i] = STDDEV_ERROR;
      _varianceErrors[i] = STDDEV_ERROR * STDDEV_ERROR;
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
            _ciphertexts[i][j][k] = Random::getNormalTorus(0, STDDEV_ERROR);
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
      Torus bit = 1;
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
        if (_plaintexts[plainID] &&
            (8 * (signed)sizeof(Torus) >= _Bgbit * (rowIdInBlock + 1))) {
          _ciphertexts[plainID][rowID * (_k + 1) + blockID][0] +=
              (bit << (8 * sizeof(Torus) - _Bgbit * (rowIdInBlock + 1)));
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
    clear_plaintexts();
    return true;
  } else {
    _plaintexts.resize(_ciphertexts.size());
  }
#ifdef USING_GPU
#else
  std::vector<Torus> decrypts(_ciphertexts.size());
  int numberThreads = ThreadPool::get_numberThreads();
  Eigen::Barrier barrier(numberThreads);
  std::unique_ptr<FFT[]> fftCalculators(new FFT[numberThreads]);
  for (int i = 0; i < numberThreads; i++)
    fftCalculators[i].set_N(_N);
  for (int i = 0; i < numberThreads; i++) {
    ThreadPool::get_threadPool().Schedule([&, i]() {
      PolynomialTorus productTorusPolynomial;
      int s = (_ciphertexts.size() * i) / numberThreads,
          e = (_ciphertexts.size() * (i + 1)) / numberThreads;
      for (int j = s; j < e; j++) {
        decrypts[j] = _ciphertexts[j][_k * _l * (_k + 1) + _k][0];
        for (int k = 0; k < _k; k++) {
          decrypts[j] -= _ciphertexts[j][_k * _l * (_k + 1) + k][0] * _s[k][0];
          for (int l = 1; l < _N; l++) {
            decrypts[j] +=
                _ciphertexts[j][_k * _l * (_k + 1) + k][l] * _s[k][_N - l];
          }
        }
      }
      barrier.Notify();
    });
  }
  barrier.Wait();
  for (int i = 0; i < (signed)_ciphertexts.size(); i++) {
    int bits = sizeof(Torus) * 8 - _Bgbit - 1;
    decrypts[i] = ((decrypts[i] >> bits) & 3);
    _plaintexts[i] = ((decrypts[i] == 1) || (decrypts[i] == 2));
  }
#endif
  return true;
}
bool Trgsw::getAllErrorsForDebugging(
    std::vector<double> &errors,
    const std::vector<bool> &expectedPlaintexts) const {
  if (_s.empty() || _ciphertexts.size() != expectedPlaintexts.size())
    return false;
  if (_ciphertexts.empty()) {
    errors.clear();
    return true;
  } else {
    errors.resize(_ciphertexts.size());
    std::fill(errors.begin(), errors.end(), 0);
  }
#ifdef USING_GPU
#else
  std::vector<double> all_errors(_ciphertexts.size() * (_k + 1) * _l);
  std::fill(all_errors.begin(), all_errors.end(), 0);
  int numberThreads = ThreadPool::get_numberThreads();
  Eigen::Barrier barrier(numberThreads);
  std::unique_ptr<FFT[]> fftCalculators(new FFT[numberThreads]);
  for (int i = 0; i < numberThreads; i++)
    fftCalculators[i].set_N(_N);
  for (int i = 0; i < numberThreads; i++) {
    ThreadPool::get_threadPool().Schedule([&, i]() {
      PolynomialTorus productTorusPolynomial, cipherTorusPolynomial,
          decryptTorusPolynomial;
      Torus bit = 1;
      int s = (_ciphertexts.size() * (_k + 1) * _l * i) / numberThreads,
          e = (_ciphertexts.size() * (_k + 1) * _l * (i + 1)) / numberThreads;
      for (int j = s; j < e; j++) {
        int cipherID = j / ((_k + 1) * _l);
        int rowID = j % ((_k + 1) * _l);
        int blockID = rowID / _l;
        int rowIdInBlock = rowID % _l;
        decryptTorusPolynomial = _ciphertexts[cipherID][rowID * (_k + 1) + _k];
        if (expectedPlaintexts[cipherID] && (blockID == _k) &&
            (8 * (signed)sizeof(Torus) >= _Bgbit * (rowIdInBlock + 1))) {
          decryptTorusPolynomial[0] -=
              (bit << (8 * sizeof(Torus) - _Bgbit * (rowIdInBlock + 1)));
        }
        for (int k = 0; k < _k; k++) {
          if (expectedPlaintexts[cipherID] && (blockID == k) &&
              (8 * (signed)sizeof(Torus) >= _Bgbit * (rowIdInBlock + 1))) {
            cipherTorusPolynomial =
                _ciphertexts[cipherID][rowID * (_k + 1) + k];
            cipherTorusPolynomial[0] -=
                (bit << (8 * sizeof(Torus) - _Bgbit * (rowIdInBlock + 1)));
            fftCalculators[i].torusPolynomialMultiplication(
                productTorusPolynomial, _s[k], cipherTorusPolynomial);
          } else {
            fftCalculators[i].torusPolynomialMultiplication(
                productTorusPolynomial, _s[k],
                _ciphertexts[cipherID][rowID * (_k + 1) + k]);
          }
          for (int l = 0; l < _N; l++) {
            decryptTorusPolynomial[l] -= productTorusPolynomial[l];
          }
        }
        for (int k = 0; k < _N; k++) {
          all_errors[j] =
              std::max(all_errors[j], std::abs(decryptTorusPolynomial[k] /
                                               std::pow(2, sizeof(Torus) * 8)));
        }
      }
      barrier.Notify();
    });
  }
  barrier.Wait();
  for (int i = 0; i < (signed)_ciphertexts.size(); i++) {
    for (int j = 0; j < (_k + 1) * _l; j++) {
      errors[i] = std::max(errors[i], all_errors[i * (_k + 1) * _l + j]);
    }
  }
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
  int numberThreads = ThreadPool::get_numberThreads();
  Eigen::Barrier barrier(numberThreads);
  for (int it = 0; it < numberThreads; it++) {
    ThreadPool::get_threadPool().Schedule([&, it]() {
      int s = (inp._ciphertexts.size() * (_k + 1) * _N * it) / numberThreads,
          e = (inp._ciphertexts.size() * (_k + 1) * _N * (it + 1)) /
              numberThreads;
      for (int newIt = s; newIt < e; newIt++) {
        int i = (newIt / _N) % (_k + 1);
        int j = newIt % _N;
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
  return true;
}
bool Trgsw::decomposite(std::vector<std::vector<PolynomialInteger>> &out,
                        const Trlwe &inp,
                        const std::vector<int> &trlweCipherIds) const {
  if (_N != inp._N || _k != inp._k)
    return false;
  if (trlweCipherIds.empty()) {
    out.clear();
    return true;
  } else {
    for (int i = 0; i < (signed)trlweCipherIds.size(); i++) {
      if (trlweCipherIds[i] < 0 ||
          trlweCipherIds[i] >= (signed)inp._ciphertexts.size())
        return false;
    }
    out.resize(trlweCipherIds.size());
    for (int i = 0; i < (signed)trlweCipherIds.size(); i++) {
      out[i].resize((_k + 1) * _l);
      for (int j = 0; j < (_k + 1) * _l; j++) {
        out[i][j].resize(_N);
      }
    }
  }
  int numberThreads = ThreadPool::get_numberThreads();
  Eigen::Barrier barrier(numberThreads);
  for (int it = 0; it < numberThreads; it++) {
    ThreadPool::get_threadPool().Schedule([&, it]() {
      int s = (trlweCipherIds.size() * (_k + 1) * _N * it) / numberThreads,
          e = (trlweCipherIds.size() * (_k + 1) * _N * (it + 1)) /
              numberThreads;
      for (int newIt = s; newIt < e; newIt++) {
        int id = newIt / ((_k + 1) * _N);
        int cipherId = trlweCipherIds[id];
        int i = (newIt / _N) % (_k + 1);
        int j = newIt % _N;
        Torus value = inp._ciphertexts[cipherId][i][j], mask = 1;
        if ((signed)sizeof(Torus) * 8 > _Bgbit * _l) {
          mask <<= (sizeof(Torus) * 8 - _Bgbit * _l - 1);
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
          out[id][i * _l + p][j] = value_temp % mask;
          if (out[id][i * _l + p][j] < -mask / 2)
            out[id][i * _l + p][j] += mask;
          if (out[id][i * _l + p][j] >= mask / 2)
            out[id][i * _l + p][j] -= mask;
          value -= (shift < 0) ? (out[id][i * _l + p][j] >> ushift)
                               : (out[id][i * _l + p][j] << ushift);
        }
      }
      barrier.Notify();
    });
  }
  barrier.Wait();
  return true;
}
void Trgsw::setParamTo(Trlwe &obj) const {
  obj._N = _N;
  obj._k = _k;
  obj._s = _s;
  obj.clear_ciphertexts();
  obj.clear_plaintexts();
}
bool Trgsw::externalProduct(Trlwe &out, const Trlwe &inp,
                            const std::vector<int> &trlweCipherIds,
                            const std::vector<int> &trgswCipherIds) const {
  if (_N != inp._N || _k != inp._k ||
      trlweCipherIds.size() != trgswCipherIds.size())
    return false;
  int numberOfProducts = trgswCipherIds.size();
  for (int i = 0; i < numberOfProducts; i++) {
    if (trlweCipherIds[i] < 0 ||
        trlweCipherIds[i] >= (signed)inp._ciphertexts.size() ||
        trgswCipherIds[i] < 0 ||
        trgswCipherIds[i] >= (signed)_ciphertexts.size())
      return false;
  }
  if (!_s.empty() && !inp._s.empty()) {
    for (int i = 0; i < _k; i++) {
      for (int j = 0; j < _N; j++) {
        if (_s[i][j] != inp._s[i][j])
          return false;
      }
    }
  }
  out._N = _N;
  out._k = _k;
  if (!_s.empty())
    out._s = _s;
  else if (!inp._s.empty())
    out._s = inp._s;
  if (numberOfProducts) {
    out._ciphertexts.resize(numberOfProducts);
    out._stddevErrors.resize(numberOfProducts);
    out._varianceErrors.resize(numberOfProducts);
    for (int i = 0; i < numberOfProducts; i++) {
      out._ciphertexts[i].resize(_k + 1);
      out._stddevErrors[i] = inp._stddevErrors[trlweCipherIds[i]] +
                             (1 + _k * _N) * std::pow(2, -_l * _Bgbit - 1);
      out._varianceErrors[i] =
          inp._varianceErrors[trlweCipherIds[i]] +
          (1 + _k * _N) * std::pow(2, -_l * _Bgbit * 2 - 2);
      for (int j = 0; j <= _k; j++) {
        out._ciphertexts[i][j].resize(_N);
        std::fill(out._ciphertexts[i][j].begin(), out._ciphertexts[i][j].end(),
                  0);
      }
    }
  } else {
    out.clear_ciphertexts();
    return true;
  }
  std::vector<std::vector<PolynomialInteger>> decVecs;
  decomposite(decVecs, inp, trlweCipherIds);
  for (int i = 0; i < numberOfProducts; i++) {
    double s = 0;
    double s2 = 0;
    for (int j = 0; j < (_k + 1) * _l; j++) {
      for (int k = 0; k < _N; k++) {
        Integer temp = decVecs[trlweCipherIds[i]][j][k];
        s += std::abs(temp);
        s2 += temp * temp;
      }
    }
    out._stddevErrors[i] += s * _stddevErrors[trgswCipherIds[i]];
    out._varianceErrors[i] += s2 * _varianceErrors[trgswCipherIds[i]];
  }
#ifdef USING_GPU
#else
  int numberThreads = ThreadPool::get_numberThreads();
  Eigen::Barrier barrier(numberThreads);
  std::unique_ptr<FFT[]> fftCalculators(new FFT[numberThreads]);
  for (int i = 0; i < numberThreads; i++)
    fftCalculators[i].set_N(_N);
  for (int it = 0; it < numberThreads; it++) {
    ThreadPool::get_threadPool().Schedule([&, it]() {
      PolynomialTorus productTorusPolynomial;
      int s = (numberOfProducts * (_k + 1) * it) / numberThreads,
          e = (numberOfProducts * (_k + 1) * (it + 1)) / numberThreads;
      for (int newIt = s; newIt < e; newIt++) {
        int i = newIt % (_k + 1);
        int trlweCipherId = newIt / (_k + 1);
        int trgswCipherId = trgswCipherIds[newIt / (_k + 1)];
        for (int j = 0; j < (_k + 1) * _l; j++) {
          fftCalculators[it].torusPolynomialMultiplication(
              productTorusPolynomial, decVecs[trlweCipherId][j],
              _ciphertexts[trgswCipherId][j * (_k + 1) + i]);
          for (int k = 0; k < _N; k++) {
            out._ciphertexts[trlweCipherId][i][k] += productTorusPolynomial[k];
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
bool Trgsw::internalProduct(int &cipherIdResult, int cipherIdA, int cipherIdB) {
  if (cipherIdA < 0 || cipherIdA >= (signed)_ciphertexts.size() ||
      cipherIdB < 0 || cipherIdB >= (signed)_ciphertexts.size())
    return false;
  if (_stddevErrors[cipherIdA] > _stddevErrors[cipherIdB]) {
    std::swap(cipherIdA, cipherIdB);
  }
  Trlwe inp, out;
  setParamTo(inp);
  inp._ciphertexts.resize((_k + 1) * _l);
  inp._stddevErrors.resize((_k + 1) * _l);
  inp._varianceErrors.resize((_k + 1) * _l);
  std::vector<int> trlweCipherIds((_k + 1) * _l), trgswCipherIds((_k + 1) * _l);
  for (int i = 0; i < (_k + 1) * _l; i++) {
    inp._ciphertexts[i].resize(_k + 1);
    inp._stddevErrors[i] = _stddevErrors[cipherIdB];
    inp._varianceErrors[i] = _varianceErrors[cipherIdB];
    for (int j = 0; j <= _k; j++) {
      inp._ciphertexts[i][j].resize(_N);
      for (int k = 0; k < _N; k++) {
        inp._ciphertexts[i][j][k] =
            _ciphertexts[cipherIdB][i * (_k + 1) + j][k];
      }
    }
    trlweCipherIds[i] = i;
    trgswCipherIds[i] = cipherIdA;
  }
  externalProduct(out, inp, trlweCipherIds, trgswCipherIds);
  cipherIdResult = _ciphertexts.size();
  _ciphertexts.resize(cipherIdResult + 1);
  _stddevErrors.resize(cipherIdResult + 1);
  _varianceErrors.resize(cipherIdResult + 1);
  _ciphertexts[cipherIdResult].resize((_k + 1) * _l * (_k + 1));
  _stddevErrors[cipherIdResult] = 0;
  _varianceErrors[cipherIdResult] = 0;
  for (int i = 0; i < (_k + 1) * _l * (_k + 1); i++) {
    _ciphertexts[cipherIdResult][i].resize(_N);
    int c = i % (_k + 1);
    int r = i / (_k + 1);
    for (int j = 0; j < _N; j++) {
      _ciphertexts[cipherIdResult][i][j] = out._ciphertexts[r][c][j];
    }
    if (c == 0) {
      _stddevErrors[cipherIdResult] =
          std::max(_stddevErrors[cipherIdResult], out._stddevErrors[r]);
      _varianceErrors[cipherIdResult] =
          std::max(_varianceErrors[cipherIdResult], out._varianceErrors[r]);
    }
  }
  return true;
}
bool Trgsw::cMux(Trlwe &out, const Trlwe &inp,
                 const std::vector<int> &trlweCipherTrueIds,
                 const std::vector<int> &trlweCipherFalseIds,
                 const std::vector<int> &trgswCipherIds) const {
  if (_N != inp._N || _k != inp._k ||
      trgswCipherIds.size() != trlweCipherTrueIds.size() ||
      trgswCipherIds.size() != trlweCipherFalseIds.size())
    return false;
  int numberOfCMux = trgswCipherIds.size();
  for (int i = 0; i < numberOfCMux; i++) {
    if (trlweCipherTrueIds[i] < 0 ||
        trlweCipherTrueIds[i] >= (signed)inp._ciphertexts.size() ||
        trlweCipherFalseIds[i] < 0 ||
        trlweCipherFalseIds[i] >= (signed)inp._ciphertexts.size() ||
        trgswCipherIds[i] < 0 ||
        trgswCipherIds[i] >= (signed)_ciphertexts.size())
      return false;
  }
  if (!_s.empty() && !inp._s.empty()) {
    for (int i = 0; i < _k; i++) {
      for (int j = 0; j < _N; j++) {
        if (_s[i][j] != inp._s[i][j])
          return false;
      }
    }
  }
  setParamTo(out);
  if (!inp._s.empty())
    out._s = inp._s;
  if (numberOfCMux == 0)
    return true;
  std::vector<int> trlweCipherIds(numberOfCMux);
  Trlwe temp;
  setParamTo(temp);
  temp._ciphertexts.resize(numberOfCMux);
  temp._stddevErrors.resize(numberOfCMux);
  temp._varianceErrors.resize(numberOfCMux);
  for (int i = 0; i < numberOfCMux; i++) {
    temp._ciphertexts[i].resize(_k + 1);
    temp._stddevErrors[i] = std::max(inp._stddevErrors[trlweCipherTrueIds[i]],
                                     inp._stddevErrors[trlweCipherFalseIds[i]]);
    temp._varianceErrors[i] =
        std::max(inp._varianceErrors[trlweCipherTrueIds[i]],
                 inp._varianceErrors[trlweCipherFalseIds[i]]);
    for (int j = 0; j <= _k; j++) {
      temp._ciphertexts[i][j].resize(_N);
    }
    trlweCipherIds[i] = i;
  }
  {
    int numberThreads = ThreadPool::get_numberThreads();
    Eigen::Barrier barrier(numberThreads);
    for (int it = 0; it < numberThreads; it++) {
      ThreadPool::get_threadPool().Schedule([&, it]() {
        int s = (numberOfCMux * (_k + 1) * _N * it) / numberThreads,
            e = (numberOfCMux * (_k + 1) * _N * (it + 1)) / numberThreads;
        for (int newIt = s; newIt < e; newIt++) {
          int i = newIt / ((_k + 1) * _N);
          int j = (newIt / _N) % (_k + 1);
          int k = newIt % _N;
          temp._ciphertexts[i][j][k] =
              inp._ciphertexts[trlweCipherTrueIds[i]][j][k] -
              inp._ciphertexts[trlweCipherFalseIds[i]][j][k];
        }
        barrier.Notify();
      });
    }
    barrier.Wait();
  }
  externalProduct(out, temp, trlweCipherIds, trgswCipherIds);
  {
    int numberThreads = ThreadPool::get_numberThreads();
    Eigen::Barrier barrier(numberThreads);
    for (int it = 0; it < numberThreads; it++) {
      ThreadPool::get_threadPool().Schedule([&, it]() {
        int s = (numberOfCMux * (_k + 1) * _N * it) / numberThreads,
            e = (numberOfCMux * (_k + 1) * _N * (it + 1)) / numberThreads;
        for (int newIt = s; newIt < e; newIt++) {
          int i = newIt / ((_k + 1) * _N);
          int j = (newIt / _N) % (_k + 1);
          int k = newIt % _N;
          out._ciphertexts[i][j][k] +=
              inp._ciphertexts[trlweCipherFalseIds[i]][j][k];
        }
        barrier.Notify();
      });
    }
    barrier.Wait();
  }
  return true;
}
bool Trgsw::blindRotate(Trlwe &out, const Trlwe &inp,
                        const std::vector<int> &trlweCipherIds,
                        const std::vector<int> &coefficients,
                        const std::vector<int> &trgswCipherIds) const {
  if (_N != inp._N || _k != inp._k ||
      coefficients.size() != trgswCipherIds.size() + 1)
    return false;
  if (!_s.empty() && !inp._s.empty()) {
    for (int i = 0; i < _k; i++) {
      for (int j = 0; j < _N; j++) {
        if (_s[i][j] != inp._s[i][j])
          return false;
      }
    }
  }
  int p = trgswCipherIds.size();
  for (int i = 0; i < p; i++) {
    if (trgswCipherIds[i] < 0 ||
        trgswCipherIds[i] >= (signed)_ciphertexts.size())
      return false;
  }
  for (unsigned int i = 0; i < trlweCipherIds.size(); i++) {
    if (trlweCipherIds[i] < 0 ||
        trlweCipherIds[i] >= (signed)inp._ciphertexts.size())
      return false;
  }
  setParamTo(out);
  if (!inp._s.empty())
    out._s = inp._s;
  if (trlweCipherIds.empty())
    return true;
  std::vector<int> trlweCipherTrueIds(trlweCipherIds.size()),
      trlweCipherFalseIds(trlweCipherIds.size()),
      temp_trgswCipherIds(trlweCipherIds.size()), coeffs(coefficients.size());
  for (int i = 0; i <= p; i++) {
    coeffs[i] = (coefficients[i] % (_N << 1) + (_N << 1)) % (_N << 1);
  }
  Trlwe *temp_inp = new Trlwe();
  Trlwe *temp_out = new Trlwe();
  setParamTo(*temp_out);
  temp_out->_ciphertexts.resize(trlweCipherIds.size());
  temp_out->_stddevErrors.resize(trlweCipherIds.size());
  temp_out->_varianceErrors.resize(trlweCipherIds.size());
  for (int i = 0; i < (signed)trlweCipherIds.size(); i++) {
    temp_out->_ciphertexts[i].resize(_k + 1);
    temp_out->_stddevErrors[i] = inp._stddevErrors[trlweCipherIds[i]];
    temp_out->_varianceErrors[i] = inp._varianceErrors[trlweCipherIds[i]];
    for (int j = 0; j <= _k; j++) {
      temp_out->_ciphertexts[i][j].resize(_N);
    }
    trlweCipherFalseIds[i] = i;
    trlweCipherTrueIds[i] = i + trlweCipherIds.size();
  }
  {
    int numberThreads = ThreadPool::get_numberThreads();
    Eigen::Barrier barrier(numberThreads);
    for (int it = 0; it < numberThreads; it++) {
      ThreadPool::get_threadPool().Schedule([&, it]() {
        int s = (trlweCipherIds.size() * (_k + 1) * _N * it) / numberThreads,
            e = (trlweCipherIds.size() * (_k + 1) * _N * (it + 1)) /
                numberThreads;
        for (int newIt = s; newIt < e; newIt++) {
          int i = newIt / ((_k + 1) * _N);
          int k = (newIt / _N) % (_k + 1);
          int deg = newIt % _N;
          if ((deg + coeffs[p] >= _N) && (deg + coeffs[p] < 2 * _N)) {
            temp_out->_ciphertexts[i][k][deg] =
                -inp._ciphertexts[trlweCipherIds[i]][k][deg + coeffs[p] - _N];
          } else {
            temp_out->_ciphertexts[i][k][deg] =
                inp._ciphertexts[trlweCipherIds[i]][k][(deg + coeffs[p]) % _N];
          }
        }
        barrier.Notify();
      });
    }
    barrier.Wait();
  }
  for (int i = 0; i < p; i++) {
    std::swap(temp_inp, temp_out);
    temp_inp->_ciphertexts.resize(trlweCipherIds.size() << 1);
    temp_inp->_stddevErrors.resize(trlweCipherIds.size() << 1);
    temp_inp->_varianceErrors.resize(trlweCipherIds.size() << 1);
    for (int j = 0; j < (signed)trlweCipherIds.size(); j++) {
      temp_inp->_ciphertexts[j + trlweCipherIds.size()].resize(_k + 1);
      temp_inp->_stddevErrors[j + trlweCipherIds.size()] =
          temp_inp->_stddevErrors[j];
      temp_inp->_varianceErrors[j + trlweCipherIds.size()] =
          temp_inp->_varianceErrors[j];
      for (int k = 0; k <= _k; k++) {
        temp_inp->_ciphertexts[j + trlweCipherIds.size()][k].resize(_N);
      }
      temp_trgswCipherIds[j] = trgswCipherIds[i];
    }
    int numberThreads = ThreadPool::get_numberThreads();
    Eigen::Barrier barrier(numberThreads);
    for (int it = 0; it < numberThreads; it++) {
      ThreadPool::get_threadPool().Schedule([&, it]() {
        int s = (trlweCipherIds.size() * (_k + 1) * _N * it) / numberThreads,
            e = (trlweCipherIds.size() * (_k + 1) * _N * (it + 1)) /
                numberThreads;
        for (int newIt = s; newIt < e; newIt++) {
          int j = newIt / ((_k + 1) * _N);
          int k = (newIt / _N) % (_k + 1);
          int deg = newIt % _N;
          if ((deg - coeffs[i] >= -_N) && (deg - coeffs[i] < 0)) {
            temp_inp->_ciphertexts[j + trlweCipherIds.size()][k][deg] =
                -(temp_inp->_ciphertexts[j][k][deg - coeffs[i] + _N]);
          } else {
            temp_inp->_ciphertexts[j + trlweCipherIds.size()][k][deg] =
                temp_inp->_ciphertexts[j][k][(deg - coeffs[i] + 2 * _N) % _N];
          }
        }
        barrier.Notify();
      });
    }
    barrier.Wait();
    cMux(*temp_out, *temp_inp, trlweCipherTrueIds, trlweCipherFalseIds,
         temp_trgswCipherIds);
  }
  out._ciphertexts.resize(trlweCipherIds.size());
  out._stddevErrors.resize(trlweCipherIds.size());
  out._varianceErrors.resize(trlweCipherIds.size());
  for (int i = 0; i < (signed)trlweCipherIds.size(); i++) {
    out._ciphertexts[i].resize(_k + 1);
    out._stddevErrors[i] = temp_out->_stddevErrors[i];
    out._varianceErrors[i] = temp_out->_varianceErrors[i];
    for (int j = 0; j <= _k; j++) {
      out._ciphertexts[i][j].resize(_N);
    }
  }
  {
    int numberThreads = ThreadPool::get_numberThreads();
    Eigen::Barrier barrier(numberThreads);
    for (int it = 0; it < numberThreads; it++) {
      ThreadPool::get_threadPool().Schedule([&, it]() {
        int s = (trlweCipherIds.size() * (_k + 1) * _N * it) / numberThreads,
            e = (trlweCipherIds.size() * (_k + 1) * _N * (it + 1)) /
                numberThreads;
        for (int newIt = s; newIt < e; newIt++) {
          int i = newIt / ((_k + 1) * _N);
          int k = (newIt / _N) % (_k + 1);
          int deg = newIt % _N;
          out._ciphertexts[i][k][deg] = temp_out->_ciphertexts[i][k][deg];
        }
        barrier.Notify();
      });
    }
    barrier.Wait();
  }
  delete temp_inp;
  delete temp_out;
  return true;
}
bool Trgsw::bootstrapTLWE(Tlwe &out, const std::vector<Torus> &constants,
                          const Tlwe &inp, int tlweCipherId,
                          const std::vector<int> &trgswCipherIds) const {
  if (inp._n != (signed)trgswCipherIds.size() || tlweCipherId < 0 ||
      tlweCipherId >= (signed)inp._ciphertexts.size())
    return false;
  for (int i = 0; i < inp._n; i++) {
    if (trgswCipherIds[i] < 0 ||
        trgswCipherIds[i] >= (signed)trgswCipherIds.size())
      return false;
  }
  out._n = _N;
  out.clear_s();
  out.clear_ciphertexts();
  out.clear_plaintexts();
  if (constants.empty())
    return true;
  Trlwe trlwe_inp, trlwe_out;
  std::vector<int> trlweCipherIds, coefficients, ps;
  setParamTo(trlwe_inp);
  trlwe_inp._ciphertexts.resize(constants.size());
  trlwe_inp._stddevErrors.resize(constants.size());
  trlwe_inp._varianceErrors.resize(constants.size());
  trlweCipherIds.resize(constants.size());
  ps.resize(constants.size());
  for (int i = 0; i < (signed)constants.size(); i++) {
    trlwe_inp._ciphertexts[i].resize(_k + 1);
    trlwe_inp._stddevErrors[i] = 0;
    trlwe_inp._varianceErrors[i] = 0;
    for (int j = 0; j < _k; j++) {
      trlwe_inp._ciphertexts[i][j].resize(_N);
      std::fill(trlwe_inp._ciphertexts[i][j].begin(),
                trlwe_inp._ciphertexts[i][j].end(), 0);
    }
    trlwe_inp._ciphertexts[i][_k].resize(_N);
    Torus constant = (constants[i] >> 1);
    for (int j = 0; j < _N; j++) {
      trlwe_inp._ciphertexts[i][_k][j] =
          (j < (_N >> 1)) ? (-constant) : constant;
    }
    trlweCipherIds[i] = i;
    ps[i] = 0;
  }
  coefficients.resize(inp._n + 1);
  for (int i = 0; i <= inp._n; i++) {
    double number = inp._ciphertexts[tlweCipherId][i];
    number = (number * 2 * _N) / std::pow(2, sizeof(Torus) * 8);
    coefficients[i] = std::llround(number);
  }
  blindRotate(trlwe_out, trlwe_inp, trlweCipherIds, coefficients,
              trgswCipherIds);
  trlwe_out.tlweExtract(out, ps, trlweCipherIds);
  for (int i = 0; i < (signed)constants.size(); i++) {
    out._ciphertexts[i][_N] += (constants[i] >> 1);
  }
  return true;
}

} // namespace thesis
