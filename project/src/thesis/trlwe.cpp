#include "thesis/trlwe.h"
#include "thesis/batched_fft.h"
#include "thesis/random.h"
#include "thesis/threadpool.h"

namespace thesis {

static const double STDDEV_ERROR = std::sqrt(2. / CONST_PI) * pow(2., -15);

// Constructors
Trlwe::Trlwe() {
  _N = 1024; // pow(2, 10)
  _k = 1;
}

// Destructor
Trlwe::~Trlwe() {}

// Get params
int Trlwe::get_N() const { return _N; }
int Trlwe::get_k() const { return _k; }

// Set attributes
void Trlwe::clear_s() { _s.clear(); }
void Trlwe::clear_ciphertexts() {
  _ciphertexts.clear();
  _stddevErrors.clear();
  _varianceErrors.clear();
}
void Trlwe::clear_plaintexts() { _plaintexts.clear(); }
bool Trlwe::set_s(const std::vector<PolynomialBinary> &s,
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
bool Trlwe::moveTo_s(std::vector<PolynomialBinary> &s, bool isForcedToCheck) {
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
void Trlwe::generate_s() {
  _s.resize(_k);
  for (int i = 0; i < _k; i++) {
    _s[i].resize(_N);
    for (int j = 0; j < _N; j++)
      _s[i][j] = Random::getUniformInteger() & 1;
  }
}
bool Trlwe::addCiphertext(const std::vector<PolynomialTorus> &cipher,
                          double stddevError, double varianceError,
                          bool isForcedToCheck) {
  if (isForcedToCheck) {
    if (stddevError < 0 || varianceError < 0)
      return false;
    const int cipher_size = cipher.size();
    if (cipher_size != _k + 1)
      return false;
    for (int i = 0; i <= _k; i++) {
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
bool Trlwe::moveCiphertext(std::vector<PolynomialTorus> &cipher,
                           double stddevError, double varianceError,
                           bool isForcedToCheck) {
  if (isForcedToCheck) {
    if (stddevError < 0 || varianceError < 0)
      return false;
    const int cipher_size = cipher.size();
    if (cipher_size != _k + 1)
      return false;
    for (int i = 0; i <= _k; i++) {
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
bool Trlwe::addPlaintext(const PolynomialBinary &plain, bool isForcedToCheck) {
  if (isForcedToCheck) {
    const int plain_size = plain.size();
    if (plain_size != _N)
      return false;
  }
  _plaintexts.push_back(plain);
  return true;
}
bool Trlwe::movePlaintext(PolynomialBinary &plain, bool isForcedToCheck) {
  if (isForcedToCheck) {
    const int plain_size = plain.size();
    if (plain_size != _N)
      return false;
  }
  _plaintexts.push_back(std::move(plain));
  return true;
}

// Get attributes
const std::vector<PolynomialBinary> &Trlwe::get_s() const { return _s; }
const std::vector<std::vector<PolynomialTorus>> &
Trlwe::get_ciphertexts() const {
  return _ciphertexts;
}
const std::vector<double> &Trlwe::get_stddevErrors() const {
  return _stddevErrors;
}
const std::vector<double> &Trlwe::get_varianceErrors() const {
  return _varianceErrors;
}
const std::vector<PolynomialBinary> &Trlwe::get_plaintexts() const {
  return _plaintexts;
}

// Utilities
bool Trlwe::encryptAll(bool isForcedToCheck) {
  if (isForcedToCheck && _s.empty())
    return false;
  if (_plaintexts.empty()) {
    clear_ciphertexts();
    return true;
  } else {
    const int _plaintexts_size = _plaintexts.size();
    _ciphertexts.resize(_plaintexts_size);
    _stddevErrors.resize(_plaintexts_size);
    _varianceErrors.resize(_plaintexts_size);
    for (int i = 0; i < _plaintexts_size; i++) {
      _ciphertexts[i].resize(_k + 1);
      _stddevErrors[i] = STDDEV_ERROR;
      _varianceErrors[i] = STDDEV_ERROR * STDDEV_ERROR;
      for (int j = 0; j <= _k; j++) {
        _ciphertexts[i][j].resize(_N);
        if (j != _k) {
          // _ciphertexts[i][j] uniform polynomial
          for (int k = 0; k < _N; k++)
            _ciphertexts[i][j][k] = Random::getUniformTorus();
        } else {
          // _ciphertexts[i][_k] Gaussian polynomial
          for (int k = 0; k < _N; k++)
            _ciphertexts[i][_k][k] = Random::getNormalTorus(0, STDDEV_ERROR);
        }
      }
    }
  }
  const int numberThreads = ThreadPool::get_numberThreads();
  const int _plaintexts_size = _plaintexts.size();
  std::unique_ptr<BatchedFFT> ptr =
      BatchedFFT::createInstance(_N, _k * numberThreads, _k, false);
  {
    Eigen::Barrier barrier(_k);
    for (int i = 0; i < _k; i++)
      ptr->setBinaryInput(_s[i], i, &barrier, false);
    barrier.Wait();
  }
  ptr->doFFT();
  {
    Eigen::Barrier barrier(_k);
    for (int i = 0; i < _k; i++)
      ptr->copyTo(i, ptr->get_batch() + i, &barrier, false);
    barrier.Wait();
  }
  for (int i = 0; i < numberThreads; i++) {
    for (int j = 0; j < _k; j++)
      ptr->setMultiplicationPair(i * _k + j, ptr->get_batch() + j, i * _k + j,
                                 false);
  }
  for (int i = 0; i < (_plaintexts_size + numberThreads - 1) / numberThreads;
       i++) {
    const int pos = i * numberThreads;
    const int n_plain = std::min(_plaintexts_size - pos, numberThreads);
    {
      Eigen::Barrier barrier(n_plain * _k);
      for (int j = 0; j < n_plain; j++) {
        for (int k = 0; k < _k; k++)
          ptr->setTorusInput(_ciphertexts[pos + j][k], j * _k + k, &barrier,
                             false);
      }
      barrier.Wait();
    }
    ptr->doFFT();
    ptr->doMultiplicationAndIFFT();
    for (int k = 0; k < _k; k++) {
      Eigen::Barrier barrier(n_plain);
      for (int j = 0; j < n_plain; j++)
        ptr->addOutput(_ciphertexts[pos + j][_k], j * _k + k, &barrier, false);
      barrier.Wait();
    }
  }
  Torus bit = 1;
  bit <<= sizeof(Torus) * 8 - 1;
  for (int i = 0; i < _plaintexts_size; i++) {
    for (int j = 0; j < _N; j++)
      _ciphertexts[i][_k][j] += _plaintexts[i][j] ? bit : 0;
  }
  return true;
}
bool Trlwe::decryptAll(bool isForcedToCheck) {
  if (isForcedToCheck && _s.empty())
    return false;
  if (_ciphertexts.empty()) {
    clear_plaintexts();
    return true;
  } else {
    const int _ciphertexts_size = _ciphertexts.size();
    _plaintexts.resize(_ciphertexts_size);
    for (int i = 0; i < _ciphertexts_size; i++)
      _plaintexts[i].resize(_N);
  }
  const int numberThreads = ThreadPool::get_numberThreads();
  const int _ciphertexts_size = _ciphertexts.size();
  std::unique_ptr<BatchedFFT> ptr =
      BatchedFFT::createInstance(_N, _k * numberThreads, _k, false);
  {
    Eigen::Barrier barrier(_k);
    for (int i = 0; i < _k; i++)
      ptr->setBinaryInput(_s[i], i, &barrier, false);
    barrier.Wait();
  }
  ptr->doFFT();
  {
    Eigen::Barrier barrier(_k);
    for (int i = 0; i < _k; i++)
      ptr->copyTo(i, ptr->get_batch() + i, &barrier, false);
    barrier.Wait();
  }
  for (int i = 0; i < numberThreads; i++) {
    for (int j = 0; j < _k; j++)
      ptr->setMultiplicationPair(i * _k + j, ptr->get_batch() + j, i * _k + j,
                                 false);
  }
  std::vector<PolynomialTorus> decrypts(numberThreads);
  for (int i = 0; i < (_ciphertexts_size + numberThreads - 1) / numberThreads;
       i++) {
    const int pos = i * numberThreads;
    const int n_plain = std::min(_ciphertexts_size - pos, numberThreads);
    {
      Eigen::Barrier barrier(n_plain * _k);
      for (int j = 0; j < n_plain; j++) {
        decrypts[j] = _ciphertexts[pos + j][_k];
        for (int k = 0; k < _k; k++)
          ptr->setTorusInput(_ciphertexts[pos + j][k], j * _k + k, &barrier,
                             false);
      }
      barrier.Wait();
    }
    ptr->doFFT();
    ptr->doMultiplicationAndIFFT();
    for (int k = 0; k < _k; k++) {
      Eigen::Barrier barrier(n_plain);
      for (int j = 0; j < n_plain; j++)
        ptr->subOutput(decrypts[j], j * _k + k, &barrier, false);
      barrier.Wait();
    }
    for (int j = 0; j < n_plain; j++) {
      for (int k = 0; k < _N; k++) {
        double bit_double = decrypts[j][k];
        _plaintexts[pos + j][k] =
            std::llround(bit_double / std::pow(2, sizeof(Torus) * 8 - 1)) & 1;
      }
    }
  }
  return true;
}
bool Trlwe::getAllErrorsForDebugging(
    std::vector<double> &errors,
    const std::vector<PolynomialBinary> &expectedPlaintexts,
    bool isForcedToCheck) const {
  if (isForcedToCheck) {
    if (_s.empty() || _ciphertexts.size() != expectedPlaintexts.size())
      return false;
    const int expectedPlaintexts_size = expectedPlaintexts.size();
    for (int i = 0; i < expectedPlaintexts_size; i++) {
      const int expectedPlaintexts_i_size = expectedPlaintexts[i].size();
      if (expectedPlaintexts_i_size != _N)
        return false;
    }
  }
  if (_ciphertexts.empty()) {
    errors.clear();
    return true;
  } else {
    errors.resize(_ciphertexts.size());
    std::fill(errors.begin(), errors.end(), 0);
  }
  const int numberThreads = ThreadPool::get_numberThreads();
  const int _ciphertexts_size = _ciphertexts.size();
  std::unique_ptr<BatchedFFT> ptr =
      BatchedFFT::createInstance(_N, _k * numberThreads, _k, false);
  {
    Eigen::Barrier barrier(_k);
    for (int i = 0; i < _k; i++)
      ptr->setBinaryInput(_s[i], i, &barrier, false);
    barrier.Wait();
  }
  ptr->doFFT();
  {
    Eigen::Barrier barrier(_k);
    for (int i = 0; i < _k; i++)
      ptr->copyTo(i, ptr->get_batch() + i, &barrier, false);
    barrier.Wait();
  }
  for (int i = 0; i < numberThreads; i++) {
    for (int j = 0; j < _k; j++)
      ptr->setMultiplicationPair(i * _k + j, ptr->get_batch() + j, i * _k + j,
                                 false);
  }
  std::vector<PolynomialTorus> decrypts(numberThreads);
  Torus bit = 1;
  bit <<= sizeof(Torus) * 8 - 1;
  for (int i = 0; i < (_ciphertexts_size + numberThreads - 1) / numberThreads;
       i++) {
    const int pos = i * numberThreads;
    const int n_plain = std::min(_ciphertexts_size - pos, numberThreads);
    {
      Eigen::Barrier barrier(n_plain * _k);
      for (int j = 0; j < n_plain; j++) {
        decrypts[j] = _ciphertexts[pos + j][_k];
        for (int k = 0; k < _k; k++)
          ptr->setTorusInput(_ciphertexts[pos + j][k], j * _k + k, &barrier,
                             false);
      }
      barrier.Wait();
    }
    ptr->doFFT();
    ptr->doMultiplicationAndIFFT();
    for (int k = 0; k < _k; k++) {
      Eigen::Barrier barrier(n_plain);
      for (int j = 0; j < n_plain; j++)
        ptr->subOutput(decrypts[j], j * _k + k, &barrier, false);
      barrier.Wait();
    }
    for (int j = 0; j < n_plain; j++) {
      for (int k = 0; k < _N; k++) {
        decrypts[j][k] -= expectedPlaintexts[pos + j][k] ? bit : 0;
        errors[pos + j] =
            std::max(errors[pos + j],
                     std::abs(decrypts[j][k] / std::pow(2, sizeof(Torus) * 8)));
      }
    }
  }
  return true;
}
void Trlwe::setParamTo(Tlwe &obj) const {
  obj._n = _N * _k;
  if (_s.empty()) {
    obj._s.empty();
  } else {
    obj._s.resize(obj._n);
    for (int i = 0; i < obj._n; i++) {
      obj._s[i] = _s[i / _N][i % _N];
    }
  }
  obj.clear_ciphertexts();
  obj.clear_plaintexts();
}
void Trlwe::tlweExtractAll(Tlwe &out) const {
  setParamTo(out);
  if (_ciphertexts.empty()) {
    return;
  } else {
    const int _ciphertexts_size = _ciphertexts.size();
    out._ciphertexts.resize(_ciphertexts_size * _N);
    out._stddevErrors.resize(_ciphertexts_size * _N);
    out._varianceErrors.resize(_ciphertexts_size * _N);
    for (int i = 0; i < _ciphertexts_size * _N; i++) {
      out._ciphertexts[i].resize(_N * _k + 1);
      out._stddevErrors[i] = _stddevErrors[i / _N];
      out._varianceErrors[i] = _varianceErrors[i / _N];
    }
  }
  int numberThreads = ThreadPool::get_numberThreads();
  Eigen::Barrier barrier(numberThreads);
  for (int it = 0; it < numberThreads; it++) {
    ThreadPool::get_threadPool().Schedule([&, it]() {
      int s = (_ciphertexts.size() * _N * (_N * _k + 1) * it) / numberThreads,
          e = (_ciphertexts.size() * _N * (_N * _k + 1) * (it + 1)) /
              numberThreads;
      for (int i = s; i < e; i++) {
        int cipherID = i / (_N * (_N * _k + 1));
        int p = (i / (_N * _k + 1)) % _N;
        int elementID = i % (_N * _k + 1);
        if (elementID == (_N * _k)) {
          out._ciphertexts[cipherID * _N + p][_N * _k] =
              _ciphertexts[cipherID][_k][p];
        } else if (p >= (elementID % _N)) {
          out._ciphertexts[cipherID * _N + p][elementID] =
              _ciphertexts[cipherID][elementID / _N][p - elementID % _N];
        } else {
          out._ciphertexts[cipherID * _N + p][elementID] =
              -_ciphertexts[cipherID][elementID / _N][p - elementID % _N + _N];
        }
      }
      barrier.Notify();
    });
  }
  barrier.Wait();
}
bool Trlwe::tlweExtract(Tlwe &out, const std::vector<int> &ps,
                        const std::vector<int> &cipherIDs,
                        bool isForcedToCheck) const {
  const int numberExtracts = ps.size();
  if (isForcedToCheck) {
    if (ps.size() != cipherIDs.size())
      return false;
    const int _ciphertexts_size = _ciphertexts.size();
    for (int i = 0; i < numberExtracts; i++) {
      if (ps[i] < 0 || ps[i] >= _N || cipherIDs[i] < 0 ||
          cipherIDs[i] >= _ciphertexts_size)
        return false;
    }
  }
  setParamTo(out);
  if (numberExtracts == 0) {
    return true;
  } else {
    out._ciphertexts.resize(numberExtracts);
    out._stddevErrors.resize(numberExtracts);
    out._varianceErrors.resize(numberExtracts);
    for (int i = 0; i < numberExtracts; i++) {
      out._ciphertexts[i].resize(_N * _k + 1);
      out._stddevErrors[i] = _stddevErrors[cipherIDs[i]];
      out._varianceErrors[i] = _varianceErrors[cipherIDs[i]];
    }
  }
  int numberThreads = ThreadPool::get_numberThreads();
  Eigen::Barrier barrier(numberThreads);
  for (int it = 0; it < numberThreads; it++) {
    ThreadPool::get_threadPool().Schedule([&, it]() {
      int s = (numberExtracts * (_N * _k + 1) * it) / numberThreads,
          e = (numberExtracts * (_N * _k + 1) * (it + 1)) / numberThreads;
      for (int i = s; i < e; i++) {
        int extractID = i / (_N * _k + 1);
        int elementID = i % (_N * _k + 1);
        int cipherID = cipherIDs[extractID];
        int p = ps[extractID];
        if (elementID == (_N * _k)) {
          out._ciphertexts[extractID][_N * _k] = _ciphertexts[cipherID][_k][p];
        } else if (p >= (elementID % _N)) {
          out._ciphertexts[extractID][elementID] =
              _ciphertexts[cipherID][elementID / _N][p - elementID % _N];
        } else {
          out._ciphertexts[extractID][elementID] =
              -_ciphertexts[cipherID][elementID / _N][p - elementID % _N + _N];
        }
      }
      barrier.Notify();
    });
  }
  barrier.Wait();
  return true;
}

} // namespace thesis
