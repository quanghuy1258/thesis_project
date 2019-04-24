#include "thesis/trgsw.h"

namespace thesis {

TrgswParams::TrgswParams(int N, int k, int l, int Bgbit, double sdError,
                         double varError) {
  if (N < 2 || (N & (N - 1)) || k < 1 || l < 1 || Bgbit < 1 || sdError <= 0 ||
      varError <= 0)
    throw std::invalid_argument("N = 2^k with k > 0 ; k > 0 ; l > 0 ; Bgbit > "
                                "1 ; sdError > 0 ; varError > 0");
  _N = N;
  _k = k;
  _l = l;
  _Bgbit = Bgbit;
  _sdError = sdError;
  _varError = _varError;
}
TrgswParams::~TrgswParams() {}

void TrgswParams::gateBootstrap(size_t count, void *tlweCiphers,
                                TorusInteger *constants, void *BKey,
                                void *KSKey) {
  if (!count)
    throw std::invalid_argument("count > 0");
}

/*
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
bool Trgsw::set_s(const std::vector<PolynomialBinary> &s,
                  bool isForcedToCheck) {
  if (isForcedToCheck) {
    const int s_size = s.size();
    if (s_size != _k)
      return false;
    for (int i = 0; i < _k; i++) {
      const int s_i_size = s[i].size();
      if (s_i_size != _N)
        return false;
    }
  }
  _s = s;
  return true;
}
bool Trgsw::move_s(std::vector<PolynomialBinary> &s, bool isForcedToCheck) {
  if (isForcedToCheck) {
    const int s_size = s.size();
    if (s_size != _k)
      return false;
    for (int i = 0; i < _k; i++) {
      const int s_i_size = s[i].size();
      if (s_i_size != _N)
        return false;
    }
  }
  _s = std::move(s);
  return true;
}
void Trgsw::generate_s() {
  _s.resize(_k);
  for (int i = 0; i < _k; i++) {
    _s[i].resize(_N);
    for (int j = 0; j < _N; j++) {
      _s[i][j] = Random::getUniformInteger() & 1;
    }
  }
}
bool Trgsw::addCiphertext(const std::vector<PolynomialTorus> &cipher,
                          double stddevError, double varianceError,
                          bool isForcedToCheck) {
  if (isForcedToCheck) {
    const int cipher_size = cipher.size();
    if (cipher_size != (_k + 1) * _l * (_k + 1) || stddevError < 0 ||
        varianceError < 0)
      return false;
    for (int i = 0; i < cipher_size; i++) {
      const int cipher_i_size = cipher[i].size();
      if (cipher_i_size != _N)
        return false;
    }
  }
  _ciphertexts.push_back(cipher);
  _stddevErrors.push_back(stddevError);
  _varianceErrors.push_back(varianceError);
  return true;
}
bool Trgsw::moveCiphertext(std::vector<PolynomialTorus> &cipher,
                           double stddevError, double varianceError,
                           bool isForcedToCheck) {
  if (isForcedToCheck) {
    const int cipher_size = cipher.size();
    if (cipher_size != (_k + 1) * _l * (_k + 1) || stddevError < 0 ||
        varianceError < 0)
      return false;
    for (int i = 0; i < cipher_size; i++) {
      const int cipher_i_size = cipher[i].size();
      if (cipher_i_size != _N)
        return false;
    }
  }
  _ciphertexts.push_back(std::move(cipher));
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
bool Trgsw::encryptAll(bool isForcedToCheck) {
  if (isForcedToCheck && _s.empty())
    return false;
  const int bitsize_Torus = 8 * sizeof(Torus);
  const Torus bit = 1;
  const int _plaintexts_size = _plaintexts.size();
  if (_plaintexts_size == 0) {
    clear_ciphertexts();
    return true;
  } else {
    _ciphertexts.resize(_plaintexts_size);
    _stddevErrors.resize(_plaintexts_size);
    _varianceErrors.resize(_plaintexts_size);
    for (int i = 0; i < _plaintexts_size; i++) {
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
  std::unique_ptr<BatchedFFT> ptr =
      BatchedFFT::createInstance(_N, _k * 2, _k, false);
  for (int i = 0; i < _k; i++)
    ptr->setBinaryInp(_s[i], _k + i, false);
  for (int i = 0; i < _plaintexts_size; i++) {
    for (int j = 0; j < (_k + 1) * _l; j++) {
      for (int k = 0; k < _k; k++)
        ptr->setTorusInp(_ciphertexts[i][j * (_k + 1) + k], k, false);
      for (int k = 0; k < _k; k++)
        ptr->setMulPair(k, _k + k, k, false);
      ptr->addAllOut(_ciphertexts[i][j * (_k + 1) + _k], false);
    }
  }
  ptr->waitAll();
  for (int i = 0; i < _plaintexts_size; i++) {
    if (!_plaintexts[i])
      continue;
    for (int j = 0; j <= _k; j++) {
      for (int k = 0; k < _l; k++) {
        if (bitsize_Torus < _Bgbit * (k + 1))
          break;
        _ciphertexts[i][j * _l * (_k + 1) + k * (_k + 1) + j][0] +=
            (bit << (bitsize_Torus - _Bgbit * (k + 1)));
      }
    }
  }
  return true;
}
bool Trgsw::decryptAll(bool isForcedToCheck) {
  if (isForcedToCheck && _s.empty())
    return false;
  const int bitsize_Torus = 8 * sizeof(Torus);
  const int _ciphertexts_size = _ciphertexts.size();
  if (_ciphertexts_size == 0) {
    clear_plaintexts();
    return true;
  } else {
    _plaintexts.resize(_ciphertexts_size);
  }
  std::vector<Torus> decrypts(_ciphertexts_size);
  int numberThreads = ThreadPool::get_numberThreads();
  Eigen::Barrier barrier(numberThreads);
  for (int i = 0; i < numberThreads; i++) {
    ThreadPool::get_threadPool().Schedule([&, i]() {
      int s = (_ciphertexts_size * i) / numberThreads,
          e = (_ciphertexts_size * (i + 1)) / numberThreads;
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
  const int bits = bitsize_Torus - _Bgbit - 1;
  for (int i = 0; i < _ciphertexts_size; i++) {
    decrypts[i] = (decrypts[i] >> bits) & 3;
    _plaintexts[i] = (decrypts[i] == 1 || decrypts[i] == 2);
  }
  return true;
}
bool Trgsw::getAllErrorsForDebugging(
    std::vector<double> &errors, const std::vector<bool> &expectedPlaintexts,
    bool isForcedToCheck) {
  const int bitsize_Torus = 8 * sizeof(Torus);
  const Torus bit = 1;
  const int _ciphertexts_size = _ciphertexts.size();
  const int expectedPlaintexts_size = expectedPlaintexts.size();
  if (isForcedToCheck &&
      (_s.empty() || _ciphertexts_size != expectedPlaintexts_size))
    return false;
  if (_ciphertexts_size == 0) {
    errors.clear();
    return true;
  } else {
    errors.resize(_ciphertexts_size);
    std::memset(errors.data(), 0, _ciphertexts_size * sizeof(double));
  }
  std::unique_ptr<BatchedFFT> ptr =
      BatchedFFT::createInstance(_N, _k * 2, _k, false);
  for (int i = 0; i < _k; i++)
    ptr->setBinaryInp(_s[i], _k + i, false);
  int old_i = -1, old_j = -1;
  PolynomialTorus decrypt;
  if (expectedPlaintexts[0] && bitsize_Torus >= _Bgbit)
    _ciphertexts[0][0][0] -= (bit << (bitsize_Torus - _Bgbit));
  for (int i = 0; i < _ciphertexts_size; i++) {
    for (int j = 0; j < (_k + 1) * _l; j++) {
      for (int k = 0; k < _k; k++)
        ptr->setTorusInp(_ciphertexts[i][j * (_k + 1) + k], k, false);
      for (int k = 0; k < _k; k++)
        ptr->setMulPair(k, _k + k, k, false);
      if (old_i >= 0 && old_j >= 0) {
        if (expectedPlaintexts[old_i] &&
            bitsize_Torus >= _Bgbit * (old_j % _l + 1)) {
          _ciphertexts[old_i][old_j * (_k + 1) + old_j / _l][0] +=
              (bit << (bitsize_Torus - _Bgbit * (old_j % _l + 1)));
        }
        for (int k = 0; k < _N; k++) {
          double bit_double = decrypt[k];
          errors[old_i] = std::max(
              errors[old_i], std::abs(bit_double / std::pow(2, bitsize_Torus)));
        }
      }
      old_i = i;
      old_j = j;
      int new_i = i, new_j = j + 1;
      if (new_j == (_k + 1) * _l) {
        new_i = i + 1;
        new_j = 0;
      }
      if (new_i == _ciphertexts_size) {
        new_i = -1;
        new_j = -1;
      }
      if (new_i >= 0 && new_j >= 0 && expectedPlaintexts[new_i] &&
          bitsize_Torus >= _Bgbit * (new_j % _l + 1)) {
        _ciphertexts[new_i][new_j * (_k + 1) + new_j / _l][0] -=
            (bit << (bitsize_Torus - _Bgbit * (new_j % _l + 1)));
      }
      decrypt = _ciphertexts[i][j * (_k + 1) + _k];
      ptr->subAllOut(decrypt, false);
    }
  }
  ptr->waitAll();
  if (expectedPlaintexts[_ciphertexts_size - 1] &&
      bitsize_Torus >= _Bgbit * _l) {
    _ciphertexts[_ciphertexts_size - 1][(_k + 1) * _l * (_k + 1) - 1][0] +=
        (bit << (bitsize_Torus - _Bgbit * _l));
  }
  return true;
}
bool Trgsw::decompositeAll(std::vector<std::vector<PolynomialInteger>> &out,
                           const Trlwe &inp, bool isForcedToCheck) const {
  if (isForcedToCheck && (_N != inp._N || _k != inp._k))
    return false;
  const int inp_ciphertexts_size = inp._ciphertexts.size();
  if (inp_ciphertexts_size == 0) {
    out.clear();
    return true;
  } else {
    out.resize(inp_ciphertexts_size);
    for (int i = 0; i < inp_ciphertexts_size; i++) {
      out[i].resize((_k + 1) * _l);
      for (int j = 0; j < (_k + 1) * _l; j++) {
        out[i][j].resize(_N);
      }
    }
  }
  const int numberThreads = ThreadPool::get_numberThreads();
  const int bitsize_Torus = 8 * sizeof(Torus);
  Eigen::Barrier barrier(numberThreads);
  for (int it = 0; it < numberThreads; it++) {
    ThreadPool::get_threadPool().Schedule([&, it]() {
      int s = (inp_ciphertexts_size * (_k + 1) * _N * it) / numberThreads,
          e = (inp_ciphertexts_size * (_k + 1) * _N * (it + 1)) / numberThreads;
      for (int newIt = s; newIt < e; newIt++) {
        int i = (newIt / _N) % (_k + 1);
        int j = newIt % _N;
        int cipherId = newIt / ((_k + 1) * _N);
        Torus value = inp._ciphertexts[cipherId][i][j], mask = 1;
        if (bitsize_Torus > _Bgbit * _l) {
          mask <<= (bitsize_Torus - _Bgbit * _l - 1);
          value += mask;
          mask = ~((mask << 1) - 1);
          value &= mask;
        }
        mask = 1;
        mask <<= _Bgbit;
        for (int p = _l - 1; p >= 0; p--) {
          Torus value_temp = value;
          int shift = bitsize_Torus - _Bgbit * (p + 1);
          int ushift = std::abs(shift);
          value_temp =
              (shift < 0) ? (value_temp << ushift) : (value_temp >> ushift);
          value_temp = value_temp % mask;
          if (value_temp < -mask / 2)
            value_temp += mask;
          if (value_temp >= mask / 2)
            value_temp -= mask;
          out[cipherId][i * _l + p][j] = value_temp;
          value -=
              (shift < 0) ? (value_temp >> ushift) : (value_temp << ushift);
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
                        const std::vector<int> &trlweCipherIds,
                        bool isForcedToCheck) const {
  const int trlweCipherIds_size = trlweCipherIds.size();
  const int inp_ciphertexts_size = inp._ciphertexts.size();
  if (isForcedToCheck) {
    if (_N != inp._N || _k != inp._k)
      return false;
    for (int i = 0; i < trlweCipherIds_size; i++) {
      if (trlweCipherIds[i] < 0 || trlweCipherIds[i] >= inp_ciphertexts_size)
        return false;
    }
  }
  if (trlweCipherIds_size == 0) {
    out.clear();
    return true;
  } else {
    out.resize(trlweCipherIds_size);
    for (int i = 0; i < trlweCipherIds_size; i++) {
      out[i].resize((_k + 1) * _l);
      for (int j = 0; j < (_k + 1) * _l; j++) {
        out[i][j].resize(_N);
      }
    }
  }
  const int numberThreads = ThreadPool::get_numberThreads();
  const int bitsize_Torus = 8 * sizeof(Torus);
  Eigen::Barrier barrier(numberThreads);
  for (int it = 0; it < numberThreads; it++) {
    ThreadPool::get_threadPool().Schedule([&, it]() {
      int s = (trlweCipherIds_size * (_k + 1) * _N * it) / numberThreads,
          e = (trlweCipherIds_size * (_k + 1) * _N * (it + 1)) / numberThreads;
      for (int newIt = s; newIt < e; newIt++) {
        int id = newIt / ((_k + 1) * _N);
        int cipherId = trlweCipherIds[id];
        int i = (newIt / _N) % (_k + 1);
        int j = newIt % _N;
        Torus value = inp._ciphertexts[cipherId][i][j], mask = 1;
        if (bitsize_Torus > _Bgbit * _l) {
          mask <<= (bitsize_Torus - _Bgbit * _l - 1);
          value += mask;
          mask = ~((mask << 1) - 1);
          value &= mask;
        }
        mask = 1;
        mask <<= _Bgbit;
        for (int p = _l - 1; p >= 0; p--) {
          Torus value_temp = value;
          int shift = bitsize_Torus - _Bgbit * (p + 1);
          int ushift = std::abs(shift);
          value_temp =
              (shift < 0) ? (value_temp << ushift) : (value_temp >> ushift);
          value_temp = value_temp % mask;
          if (value_temp < -mask / 2)
            value_temp += mask;
          if (value_temp >= mask / 2)
            value_temp -= mask;
          out[id][i * _l + p][j] = value_temp;
          value -=
              (shift < 0) ? (value_temp >> ushift) : (value_temp << ushift);
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
  obj._ptrEncDec.reset();
  obj.clear_ciphertexts();
  obj.clear_plaintexts();
}
bool Trgsw::_externalProduct(Trlwe &out, const Trlwe &inp,
                             const std::vector<int> &trlweCipherIds,
                             const std::vector<int> &trgswCipherIds,
                             std::unique_ptr<BatchedFFT> &ptr,
                             bool isForcedToCheck) const {
  const int numberProducts = trgswCipherIds.size();
  const int trlweCipherIds_size = trlweCipherIds.size();
  const int inp_ciphertexts_size = inp._ciphertexts.size();
  const int _ciphertexts_size = _ciphertexts.size();
  if (isForcedToCheck) {
    if (_N != inp._N || _k != inp._k || trlweCipherIds_size != numberProducts ||
        !ptr || ptr->get_N() != _N ||
        ptr->get_batch_inp() < (_k + 1) * _l * 2 ||
        ptr->get_batch_out() != (_k + 1) * _l)
      return false;
    for (int i = 0; i < numberProducts; i++) {
      if (trlweCipherIds[i] < 0 || trlweCipherIds[i] >= inp_ciphertexts_size ||
          trgswCipherIds[i] < 0 || trgswCipherIds[i] >= _ciphertexts_size)
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
  }
  setParamTo(out);
  if (!inp._s.empty())
    out._s = inp._s;
  if (numberProducts == 0) {
    return true;
  } else {
    out._ciphertexts.resize(numberProducts);
    out._stddevErrors.resize(numberProducts);
    out._varianceErrors.resize(numberProducts);
    for (int i = 0; i < numberProducts; i++) {
      out._ciphertexts[i].resize(_k + 1);
      out._stddevErrors[i] = inp._stddevErrors[trlweCipherIds[i]] +
                             (1 + _k * _N) * std::pow(2, -_l * _Bgbit - 1);
      out._varianceErrors[i] =
          inp._varianceErrors[trlweCipherIds[i]] +
          (1 + _k * _N) * std::pow(2, -_l * _Bgbit * 2 - 2);
      for (int j = 0; j <= _k; j++) {
        out._ciphertexts[i][j].resize(_N);
        std::memset(out._ciphertexts[i][j].data(), 0, _N * sizeof(Torus));
      }
    }
  }
  std::vector<std::vector<PolynomialInteger>> decVecs;
  decomposite(decVecs, inp, trlweCipherIds, false);
  for (int i = 0; i < numberProducts; i++) {
    double s = 0;
    double s2 = 0;
    for (int j = 0; j < (_k + 1) * _l; j++) {
      ptr->setIntegerInp(decVecs[i][j], (_k + 1) * _l + j, false);
      for (int k = 0; k < _N; k++) {
        double temp = decVecs[i][j][k];
        s += std::abs(temp);
        s2 += temp * temp;
      }
    }
    out._stddevErrors[i] += s * _stddevErrors[trgswCipherIds[i]];
    out._varianceErrors[i] += s2 * _varianceErrors[trgswCipherIds[i]];
    for (int k = 0; k <= _k; k++) {
      for (int j = 0; j < (_k + 1) * _l; j++)
        ptr->setTorusInp(_ciphertexts[trgswCipherIds[i]][j * (_k + 1) + k], j,
                         false);
      for (int j = 0; j < (_k + 1) * _l; j++)
        ptr->setMulPair(j, (_k + 1) * _l + j, j, false);
      ptr->addAllOut(out._ciphertexts[i][k], false);
    }
  }
  ptr->waitAll();
  return true;
}
bool Trgsw::externalProduct(Trlwe &out, const Trlwe &inp,
                            const std::vector<int> &trlweCipherIds,
                            const std::vector<int> &trgswCipherIds,
                            bool isForcedToCheck) const {
  std::unique_ptr<BatchedFFT> ptr =
      BatchedFFT::createInstance(_N, (_k + 1) * _l * 2, (_k + 1) * _l, false);
  return _externalProduct(out, inp, trlweCipherIds, trgswCipherIds, ptr,
                          isForcedToCheck);
}
bool Trgsw::_internalProduct(int &cipherIdResult, int cipherIdA, int cipherIdB,
                             std::unique_ptr<BatchedFFT> &ptr,
                             bool isForcedToCheck) {
  const int _ciphertexts_size = _ciphertexts.size();
  if (isForcedToCheck &&
      (cipherIdA < 0 || cipherIdA >= _ciphertexts_size || cipherIdB < 0 ||
       cipherIdB >= _ciphertexts_size || !ptr || ptr->get_N() != _N ||
       ptr->get_batch_inp() < (_k + 1) * _l * (_k + 2) ||
       ptr->get_batch_out() != (_k + 1) * _l))
    return false;
  if (_varianceErrors[cipherIdA] > _varianceErrors[cipherIdB])
    std::swap(cipherIdA, cipherIdB);
  cipherIdResult = _ciphertexts.size();
  _ciphertexts.resize(cipherIdResult + 1);
  _stddevErrors.resize(cipherIdResult + 1);
  _varianceErrors.resize(cipherIdResult + 1);
  _ciphertexts[cipherIdResult].resize((_k + 1) * _l * (_k + 1));
  _stddevErrors[cipherIdResult] = 0;
  _varianceErrors[cipherIdResult] = 0;
  for (int i = 0; i < (_k + 1) * _l * (_k + 1); i++) {
    _ciphertexts[cipherIdResult][i].resize(_N);
    std::memset(_ciphertexts[cipherIdResult][i].data(), 0, sizeof(Torus) * _N);
    ptr->setTorusInp(_ciphertexts[cipherIdA][i], i, false);
  }
  std::vector<std::vector<PolynomialInteger>> decVecs;
  Trlwe trlweObj;
  setParamTo(trlweObj);
  trlweObj._ciphertexts.resize(1);
  trlweObj._ciphertexts[0].resize(_k + 1);
  for (int i = 0; i <= _k; i++)
    trlweObj._ciphertexts[0][i] = std::move(_ciphertexts[cipherIdB][i]);
  for (int i = 0; i < (_k + 1) * _l; i++) {
    double s = 0;
    double s2 = 0;
    decompositeAll(decVecs, trlweObj, false);
    for (int j = 0; j < (_k + 1) * _l; j++) {
      ptr->setIntegerInp(decVecs[0][j], (_k + 1) * _l * (_k + 1) + j, false);
      for (int k = 0; k < _N; k++) {
        double temp = decVecs[0][j][k];
        s += std::abs(temp);
        s2 += temp * temp;
      }
    }
    _stddevErrors[cipherIdResult] = std::max(_stddevErrors[cipherIdResult], s);
    _varianceErrors[cipherIdResult] =
        std::max(_varianceErrors[cipherIdResult], s2);
    for (int k = 0; k <= _k; k++) {
      for (int j = 0; j < (_k + 1) * _l; j++)
        ptr->setMulPair((_k + 1) * _l * (_k + 1) + j, j * (_k + 1) + k, j, false);
      _ciphertexts[cipherIdB][i * (_k + 1) + k] =
          std::move(trlweObj._ciphertexts[0][k]);
      if (i + 1 < (_k + 1) * _l)
        trlweObj._ciphertexts[0][k] =
            std::move(_ciphertexts[cipherIdB][(i + 1) * (_k + 1) + k]);
      ptr->addAllOut(_ciphertexts[cipherIdResult][i * (_k + 1) + k], false);
    }
  }
  _stddevErrors[cipherIdResult] =
      _stddevErrors[cipherIdResult] * _stddevErrors[cipherIdA] +
      (1 + _k * _N) * std::pow(2, -_l * _Bgbit - 1) + _stddevErrors[cipherIdB];
  _varianceErrors[cipherIdResult] =
      _varianceErrors[cipherIdResult] * _varianceErrors[cipherIdA] +
      (1 + _k * _N) * std::pow(2, -_l * _Bgbit * 2 - 2) +
      _varianceErrors[cipherIdB];
  ptr->waitAll();
  return true;
}
bool Trgsw::internalProduct(int &cipherIdResult, int cipherIdA, int cipherIdB,
                            bool isForcedToCheck) {
  std::unique_ptr<BatchedFFT> ptr = BatchedFFT::createInstance(
      _N, (_k + 1) * _l * (_k + 2), (_k + 1) * _l, false);
  return _internalProduct(cipherIdResult, cipherIdA, cipherIdB, ptr,
                          isForcedToCheck);
}
bool Trgsw::_cMux(Trlwe &out, const Trlwe &inp,
                  const std::vector<int> &trlweCipherTrueIds,
                  const std::vector<int> &trlweCipherFalseIds,
                  const std::vector<int> &trgswCipherIds,
                  std::unique_ptr<BatchedFFT> &ptr,
                  bool isForcedToCheck) const {
  const int numberCMux = trgswCipherIds.size();
  const int inp_ciphertexts_size = inp._ciphertexts.size();
  const int _ciphertexts_size = _ciphertexts.size();
  const int trlweCipherTrueIds_size = trlweCipherTrueIds.size();
  const int trlweCipherFalseIds_size = trlweCipherFalseIds.size();
  if (isForcedToCheck) {
    if (_N != inp._N || _k != inp._k ||
        trlweCipherTrueIds_size != numberCMux ||
        trlweCipherFalseIds_size != numberCMux || !ptr ||
        ptr->get_N() != _N || ptr->get_batch_inp() < (_k + 1) * _l * 2 ||
        ptr->get_batch_out() != (_k + 1) * _l)
      return false;
    for (int i = 0; i < numberCMux; i++) {
      if (trlweCipherTrueIds[i] < 0 ||
          trlweCipherTrueIds[i] >= inp_ciphertexts_size ||
          trlweCipherFalseIds[i] < 0 ||
          trlweCipherFalseIds[i] >= inp_ciphertexts_size ||
          trgswCipherIds[i] < 0 || trgswCipherIds[i] >= _ciphertexts_size)
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
  }
  setParamTo(out);
  if (!inp._s.empty())
    out._s = inp._s;
  if (numberCMux == 0)
    return true;
  std::vector<int> trlweCipherIds(numberCMux);
  Trlwe temp;
  setParamTo(temp);
  temp._ciphertexts.resize(numberCMux);
  temp._stddevErrors.resize(numberCMux);
  temp._varianceErrors.resize(numberCMux);
  for (int i = 0; i < numberCMux; i++) {
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
        int s = (numberCMux * (_k + 1) * _N * it) / numberThreads,
            e = (numberCMux * (_k + 1) * _N * (it + 1)) / numberThreads;
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
  _externalProduct(out, temp, trlweCipherIds, trgswCipherIds, ptr, false);
  {
    int numberThreads = ThreadPool::get_numberThreads();
    Eigen::Barrier barrier(numberThreads);
    for (int it = 0; it < numberThreads; it++) {
      ThreadPool::get_threadPool().Schedule([&, it]() {
        int s = (numberCMux * (_k + 1) * _N * it) / numberThreads,
            e = (numberCMux * (_k + 1) * _N * (it + 1)) / numberThreads;
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
bool Trgsw::cMux(Trlwe &out, const Trlwe &inp,
                 const std::vector<int> &trlweCipherTrueIds,
                 const std::vector<int> &trlweCipherFalseIds,
                 const std::vector<int> &trgswCipherIds,
                 bool isForcedToCheck) const {
  std::unique_ptr<BatchedFFT> ptr =
      BatchedFFT::createInstance(_N, (_k + 1) * _l * 2, (_k + 1) * _l, false);
  return _cMux(out, inp, trlweCipherTrueIds, trlweCipherFalseIds, trgswCipherIds,
               ptr, isForcedToCheck);
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
      std::memset(trlwe_inp._ciphertexts[i][j].data(), 0, _N * sizeof(Torus));
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

bool Trgsw::gateBootstrap(Tlwe &out, const std::vector<Torus> &constants,
                          const Tlwe &inp, int tlweCipherId,
                          const std::vector<int> &trgswCipherIds,
                          const Tlwe &ks, int ks_t) const {
  if (ks_t < 1 || ks_t > (signed)sizeof(Torus) * 8 ||
      (signed)ks._ciphertexts.size() != ks_t * _k * _N)
    return false;
  Tlwe temp_out;
  if (!bootstrapTLWE(temp_out, constants, inp, tlweCipherId, trgswCipherIds))
    return false;
  out._n = ks._n;
  out._s = ks._s;
  out.clear_ciphertexts();
  out.clear_plaintexts();
  if (temp_out._ciphertexts.empty())
    return true;
  out._ciphertexts.resize(temp_out._ciphertexts.size());
  out._stddevErrors.resize(temp_out._ciphertexts.size());
  out._varianceErrors.resize(temp_out._ciphertexts.size());
  for (int i = 0; i < (signed)temp_out._ciphertexts.size(); i++) {
    out._ciphertexts[i].resize(ks._n + 1);
    std::memset(out._ciphertexts[i].data(), 0, (ks._n + 1) * sizeof(Torus));
    out._ciphertexts[i][ks._n] = temp_out._ciphertexts[i][_k * _N];
    out._stddevErrors[i] = temp_out._stddevErrors[i];
    out._varianceErrors[i] = temp_out._varianceErrors[i];
  }
  int numberThreads = ThreadPool::get_numberThreads();
  Eigen::Barrier barrier(numberThreads);
  for (int it = 0; it < numberThreads; it++) {
    ThreadPool::get_threadPool().Schedule([&, it]() {
      int s = (temp_out._ciphertexts.size() * (ks._n + 1) * it) / numberThreads,
          e = (temp_out._ciphertexts.size() * (ks._n + 1) * (it + 1)) /
              numberThreads;
      Torus round = 0;
      if ((signed)sizeof(Torus) * 8 > ks_t) {
        round = 1;
        round <<= (sizeof(Torus) * 8 - ks_t - 1);
      }
      for (int newIt = s; newIt < e; newIt++) {
        int i = newIt / (ks._n + 1);
        int j = newIt % (ks._n + 1);
        for (int k = 0; k < _k * _N; k++) {
          Torus temp_torus = temp_out._ciphertexts[i][k] + round;
          for (int l = 0; l < ks_t; l++) {
            Integer bit = (temp_torus >> (sizeof(Torus) * 8 - l - 1)) & 1;
            out._ciphertexts[i][j] -= bit * ks._ciphertexts[k * ks_t + l][j];
            if (j == ks._n) {
              out._stddevErrors[i] += ks._stddevErrors[k * ks_t + l];
              out._varianceErrors[i] += ks._varianceErrors[k * ks_t + l];
            }
          }
          if (j == ks._n) {
            out._stddevErrors[i] += std::pow(2, -(ks_t + 1));
            out._varianceErrors[i] += std::pow(2, -2 * (ks_t + 1));
          }
        }
      }
      barrier.Notify();
    });
  }
  barrier.Wait();
  return true;
}
*/
} // namespace thesis
