#include "thesis/decomposition.h"
#include "thesis/memory_management.h"
#include "thesis/random.h"
#include "thesis/stream.h"
#include "thesis/torus_utility.h"
#include "thesis/trgsw_cipher.h"
#include "thesis/trgsw_function.h"
#include "thesis/trlwe_cipher.h"
#include "thesis/trlwe_function.h"

#include "mpc_application.h"

using namespace thesis;

MpcApplication::MpcApplication(int numParty, int partyId, int N, int m, int l,
                               double sdFresh)
    : _fft_privkey(N, 2, 1), _fft_pubkey(N, 2, m),
      _fft_preExpand(N, m * (numParty - 1), l),
      _fft_preExpandRandom(N, numParty - 1, m) {
  if (numParty < 2 || partyId < 0 || partyId >= numParty || N < 2 ||
      (N & (N - 1)) || m < 1 || l < 1 || sdFresh <= 0)
    throw std::invalid_argument("numParty > 1 ; 0 <= partyId < numParty ; N = "
                                "2^a with a > 0 ; m > 0 ; l > 0 ; sdFresh > 0");
  _numParty = numParty;
  _partyId = partyId;
  _N = N;
  _m = m;
  _l = l;
  _sdFresh = sdFresh;
  _privkey = (TorusInteger *)MemoryManagement::mallocMM(getSizePrivkey());
  _pubkey.resize(m, nullptr);
  TorusInteger *mem_pubkey =
      (TorusInteger *)MemoryManagement::mallocMM(getSizePubkey());
  for (int i = 0; i < _m; i++)
    _pubkey[i] = new TrlweCipher(mem_pubkey + 2 * N * i, 2 * N, N, 1, sdFresh,
                                 sdFresh * sdFresh);
  int numStream = 0; // TODO: Always check maximum size when use
  numStream = std::max(numStream, 2 * _l * _m + 2);
  numStream = std::max(numStream, _numParty);
  _stream.resize(numStream, nullptr);
  for (int i = 0; i < numStream; i++)
    _stream[i] = Stream::createS();
}
MpcApplication::~MpcApplication() {
  MemoryManagement::freeMM(_privkey);
  MemoryManagement::freeMM(_pubkey[0]->_data);
  for (int i = 0; i < _m; i++)
    delete _pubkey[i];
  int numStream = 0; // TODO: Always check maximum size when use
  numStream = std::max(numStream, 2 * _l * _m + 2);
  numStream = std::max(numStream, _numParty);
  for (int i = 0; i < numStream; i++)
    Stream::destroyS(_stream[i]);
}
void MpcApplication::createPrivkey() {
  TrlweFunction::genkey(_privkey, _N, 1);
  TrlweFunction::keyToFFT(_privkey, _N, 1, &_fft_privkey);
}
void MpcApplication::importPrivkey(void *hPrivkey) {
  if (!hPrivkey)
    return;
  MemoryManagement::memcpyMM_h2d(_privkey, hPrivkey, getSizePrivkey());
  TrlweFunction::keyToFFT(_privkey, _N, 1, &_fft_privkey);
}
void MpcApplication::exportPrivkey(void *hPrivkey) {
  if (!hPrivkey)
    return;
  MemoryManagement::memcpyMM_d2h(hPrivkey, _privkey, getSizePrivkey());
}
int MpcApplication::getSizePrivkey() { return _N * sizeof(TorusInteger); }
void MpcApplication::createPubkey() {
  for (int i = 0; i < _m; i++) {
    TrlweFunction::createSample(&_fft_privkey, i & 1, _pubkey[i]);
    _fft_privkey.waitOut(i & 1);
    _fft_pubkey.setInp(_pubkey[i]->get_pol_data(0), 0, i);
    _fft_pubkey.setInp(_pubkey[i]->get_pol_data(1), 1, i);
  }
}
void MpcApplication::importPubkey(void *hPubkey) {
  if (!hPubkey)
    return;
  MemoryManagement::memcpyMM_h2d(_pubkey[0]->_data, hPubkey, getSizePubkey());
  for (int i = 0; i < _m; i++) {
    _fft_pubkey.setInp(_pubkey[i]->get_pol_data(0), 0, i);
    _fft_pubkey.setInp(_pubkey[i]->get_pol_data(1), 1, i);
  }
}
void MpcApplication::exportPubkey(void *hPubkey) {
  if (!hPubkey)
    return;
  MemoryManagement::memcpyMM_d2h(hPubkey, _pubkey[0]->_data, getSizePubkey());
}
int MpcApplication::getSizePubkey() {
  return _N * 2 * sizeof(TorusInteger) * _m;
}
void MpcApplication::encrypt(bool msg, void *hMainCipher, void *hRandCipher,
                             void *hRandom) {
  if (!hMainCipher)
    return;
  // Create random
  TorusInteger *random =
      (TorusInteger *)MemoryManagement::mallocMM(getSizeRandom());
  Random::setUniform(random, 2 * _l * _m * _N,
                     [](TorusInteger x) -> TorusInteger { return x & 1; });
  // Copy random from device to host (if possible)
  if (hRandom)
    MemoryManagement::memcpyMM_d2h(hRandom, random, getSizeRandom(),
                                   _stream[1]);
  // Create and copy random cipher (if posible)
  std::vector<TrgswCipher *> randCipher(2 * _l * _m, nullptr);
  if (hRandCipher) {
    for (int i = 0; i < 2 * _l * _m; i++)
      randCipher[i] =
          new TrgswCipher(_N, 1, _l, 1, _sdFresh, _sdFresh * _sdFresh);
    for (int i = 0; i < 2 * _l * _m; i++) {
      for (int j = 0; j < _l; j++)
        TrlweFunction::createSample(&_fft_privkey, (_l * i + j) & 1,
                                    randCipher[i]->get_trlwe_data(_l + j), _N,
                                    1, _sdFresh);
    }
    _fft_privkey.waitAllOut();
    for (int i = 0; i < 2 * _l * _m; i++)
      TrgswFunction::addMuGadget(random + _N * i, randCipher[i],
                                 _stream[i + 2]);
    TorusInteger *ptr = (TorusInteger *)hRandCipher;
    for (int i = 0; i < 2 * _l * _m; i++)
      MemoryManagement::memcpyMM_d2h(
          ptr + 2 * _l * _N * i, randCipher[i]->get_trlwe_data(_l),
          2 * _l * _N * sizeof(TorusInteger), _stream[i + 2]);
  }
  // Create main cipher
  //   sdError = m * N * sdFresh + sdFresh
  //   varError = m * N * (sdFresh ^ 2) + (sdFresh ^ 2)
  TrgswCipher mainCipher(_N, 1, _l, 1, _sdFresh * (_m * _N + 1),
                         _sdFresh * _sdFresh * (_m * _N + 1));
  mainCipher.clear_trgsw_data();
  for (int i = 0; i < 2 * _l; i++)
    Random::setNormalTorus(mainCipher.get_pol_data(i, 1), _N, _sdFresh);
  // main cipher += random * pubkey
  for (int i = 0; i < 2 * _l; i++) {
    for (int j = 0; j < _m; j++)
      _fft_pubkey.setInp(random + (i * _m + j) * _N, j);
    for (int j = 0; j < _m; j++) {
      _fft_pubkey.setMul(0, j);
      _fft_pubkey.setMul(1, j);
    }
    _fft_pubkey.addAllOut(mainCipher.get_pol_data(i, 0), 0);
    _fft_pubkey.addAllOut(mainCipher.get_pol_data(i, 1), 1);
  }
  _fft_pubkey.waitAllOut();
  // main cipher += msg*G;
  if (msg)
    TrgswFunction::addMuGadget(1, &mainCipher, _stream[0]);
  // Copy main cipher from device to host
  MemoryManagement::memcpyMM_d2h(hMainCipher, mainCipher._data,
                                 getSizeMainCipher(), _stream[0]);
  // Wait all streams
  for (int i = 0; i < 2 * _l * _m + 2; i++)
    Stream::synchronizeS(_stream[i]);
  // Delete randCipher
  if (hRandCipher) {
    for (int i = 0; i < 2 * _l * _m; i++)
      delete randCipher[i];
  }
}
int MpcApplication::getSizeMainCipher() {
  return 4 * _l * _N * sizeof(TorusInteger);
}
int MpcApplication::getSizeRandCipher() {
  // (2 * l) x 4 random ciphers
  // --> each cipher: matrix 2 x l torus polynomials
  // --> each polynomial: deg = N
  return 4 * _l * _m * _l * _N * sizeof(TorusInteger);
}
int MpcApplication::getSizeRandom() {
  return 2 * _l * _m * _N * sizeof(TorusInteger);
}
void MpcApplication::preExpand(void *hPubkey, void *hPreExpand) {
  if (!hPubkey || !hPreExpand)
    return;
  // Init pubkey and preExpand memory on VRAM
  TorusInteger *pubkey_ptr =
      (TorusInteger *)MemoryManagement::mallocMM(getSizePubkey());
  MemoryManagement::memcpyMM_h2d(pubkey_ptr, hPubkey, getSizePubkey());
  TorusInteger *preExpand_ptr =
      (TorusInteger *)MemoryManagement::mallocMM(getSizePreExpand());
  Random::setNormalTorus(preExpand_ptr, _m * _N, _sdFresh);
  // preExpand_ptr -= hPubkey * privkey
  for (int i = 0; i < _m; i++) {
    _fft_privkey.setInp(pubkey_ptr + 2 * _N * i, i & 1, 0);
    _fft_privkey.setMul(i & 1, 0);
    TorusUtility::subVector(preExpand_ptr + _N * i,
                            pubkey_ptr + (2 * i + 1) * _N, _N);
    _fft_privkey.addAllOut(preExpand_ptr + _N * i, i & 1);
  }
  _fft_privkey.waitAllOut();
  // Copy preExpand_ptr from device to host
  MemoryManagement::memcpyMM_d2h(hPreExpand, preExpand_ptr, getSizePreExpand());
  // Delete pubkey and preExpand
  MemoryManagement::freeMM(pubkey_ptr);
  MemoryManagement::freeMM(preExpand_ptr);
}
int MpcApplication::getSizePreExpand() {
  return _m * _N * sizeof(TorusInteger);
}
TorusInteger *MpcApplication::_decompPreExpand(void *hPreExpand, int id) {
  // preExpandCipher for only decomposition
  //   -> choosing any value for sd and var is ok
  TrlweCipher preExpandCipher(_N, _m, 1, 1);
  // param for only decomposition
  //   -> no need allocate memory and calculate value for sd and var
  TrgswCipher param(preExpandCipher._data, _N * (_m + 1) * (_m + 1) * _l, _N,
                    _m, _l, 1, 1, 1);
  MemoryManagement::memcpyMM_h2d(preExpandCipher._data, hPreExpand,
                                 getSizePreExpand());
  TorusInteger *decompPreExpand = (TorusInteger *)MemoryManagement::mallocMM(
      (_m + 1) * _N * sizeof(TorusInteger) * _l);
  Decomposition::onlyDecomp(&preExpandCipher, &param, decompPreExpand);
  for (int i = 0; i < _m; i++) {
    for (int j = 0; j < _l; j++)
      _fft_preExpand.setInp(decompPreExpand + (i * _l + j) * _N, id * _m + i,
                            j);
  }
  return decompPreExpand;
}
TrgswCipher *MpcApplication::_extend(void *hPreExpand, int partyId,
                                     TorusInteger *cipher) {
  // Create output trgsw cipher
  double e_dec = std::pow(2, -_l - 1);
  // sdError = m * l * N * sdFresh + N * e_dec * m + m * N * sdFresh
  // varError = m * l * N * (sdFresh ^ 2) + N * (e_dec ^ 2) * m +
  //            m * N * (sdFresh ^ 2)
  TrgswCipher *out = new TrgswCipher(
      _N, 1, _l, 1, _m * _N * ((_l + 1) * _sdFresh + e_dec),
      _m * _N * ((_l + 1) * _sdFresh * _sdFresh + e_dec * e_dec));
  // Determine the position of partyId in _fft_with_preExpand
  int id = (partyId < _partyId) ? partyId : (partyId - 1);
  // Do decomposition if hPreExpand is not null
  TorusInteger *decompPreExpand = nullptr;
  if (hPreExpand)
    decompPreExpand = _decompPreExpand(hPreExpand, id);
  // Clear data of out
  out->clear_trgsw_data();
  // Calculate extend cipher
  for (int j = 0; j < _m; j++) {
    for (int i = 0; i < 2 * _l; i++) {
      // First column
      for (int k = 0; k < _l; k++)
        _fft_preExpand.setInp(cipher + ((i * _m + j) * 2 * _l + 2 * k) * _N, k);
      for (int k = 0; k < _l; k++)
        _fft_preExpand.setMul(id * _m + j, k);
      if (j > 0)
        _fft_preExpand.waitOut(id * _m + j - 1);
      _fft_preExpand.addAllOut(out->get_pol_data(i, 0), id * _m + j);
      // Second column
      for (int k = 0; k < _l; k++)
        _fft_preExpand.setInp(cipher + ((i * _m + j) * 2 * _l + 2 * k + 1) * _N,
                              k);
      for (int k = 0; k < _l; k++)
        _fft_preExpand.setMul(id * _m + j, k);
      if (j > 0)
        _fft_preExpand.waitOut(id * _m + j - 1);
      _fft_preExpand.addAllOut(out->get_pol_data(i, 1), id * _m + j);
    }
  }
  _fft_preExpand.waitOut(id * _m + _m - 1);
  // Free all allocated memory
  if (hPreExpand)
    MemoryManagement::freeMM(decompPreExpand);
  return out;
}
TrgswCipher *MpcApplication::_extendWithPlainRandom(void *hPreExpand,
                                                    int partyId,
                                                    TorusInteger *random) {
  // Create output trgsw cipher
  TrgswCipher *out = new TrgswCipher(_N, 1, _l, 1, (1 + _m * _N) * _sdFresh,
                                     (1 + _m * _N) * _sdFresh * _sdFresh);
  // Determine the position of partyId in _fft_with_preExpand
  int id = (partyId < _partyId) ? partyId : (partyId - 1);
  // Move preExpand from host to device if possible
  TorusInteger *preExpandCipher = nullptr;
  if (hPreExpand) {
    preExpandCipher =
        (TorusInteger *)MemoryManagement::mallocMM(getSizePreExpand());
    MemoryManagement::memcpyMM_h2d(preExpandCipher, hPreExpand,
                                   getSizePreExpand());
    for (int i = 0; i < _m; i++)
      _fft_preExpandRandom.setInp(preExpandCipher + i * _N, id, i);
  }
  // Clear data of out
  out->clear_trgsw_data();
  // Calculate extend cipher
  for (int i = 0; i < 2 * _l; i++)
    TrlweFunction::createSample(&_fft_privkey, i & 1, out->get_trlwe_data(i),
                                _N, 1, _sdFresh);
  _fft_privkey.waitAllOut();
  for (int i = 0; i < 2 * _l; i++) {
    for (int j = 0; j < _m; j++)
      _fft_preExpandRandom.setInp(random + (i * _m + j) * _N, j);
    for (int j = 0; j < _m; j++)
      _fft_preExpandRandom.setMul(id, j);
    _fft_preExpandRandom.addAllOut(out->get_pol_data(i, 1), id);
  }
  _fft_preExpandRandom.waitAllOut();
  // Free all allocated memory
  if (hPreExpand)
    MemoryManagement::freeMM(preExpandCipher);
  return out;
}
std::vector<TrgswCipher *>
MpcApplication::expand(std::vector<void *> &hPreExpand,
                       std::function<void(void *)> freeFnPreExpand, int partyId,
                       void *hMainCipher, void *hRandCipher) {
  std::vector<TrgswCipher *> out(_numParty * _numParty, nullptr);
  if (partyId < 0 || partyId >= _numParty || !hMainCipher || !hRandCipher)
    return out;
  // Copy cipher from host to device
  TorusInteger *randCipher =
      (TorusInteger *)MemoryManagement::mallocMM(getSizeRandCipher());
  MemoryManagement::memcpyMM_h2d(randCipher, hRandCipher, getSizeRandCipher());
  // Copy main cipher
  for (int i = 0; i < _numParty; i++) {
    out[i * _numParty + i] = new TrgswCipher(_N, 1, _l, 1, _m * _N * _sdFresh,
                                             _m * _N * _sdFresh * _sdFresh);
    MemoryManagement::memcpyMM_h2d(out[i * _numParty + i]->_data, hMainCipher,
                                   getSizeMainCipher(), _stream[i]);
  }
  // Create extend cipher
  int sizePreExpandVec = hPreExpand.size();
  for (int i = 0; i < _numParty; i++) {
    if (i == partyId)
      continue;
    void *hPreExpandPtr = nullptr;
    if (i < sizePreExpandVec)
      hPreExpandPtr = hPreExpand[i];
    out[i * _numParty + partyId] = _extend(hPreExpandPtr, i, randCipher);
    if (hPreExpandPtr != nullptr && freeFnPreExpand != nullptr) {
      freeFnPreExpand(hPreExpand[i]);
      hPreExpand[i] = nullptr;
    }
  }
  // Free all allocated memory
  MemoryManagement::freeMM(randCipher);
  // Wait all streams
  for (int i = 0; i < _numParty; i++)
    Stream::synchronizeS(_stream[i]);
  return out;
}
std::vector<TrgswCipher *> MpcApplication::expandWithPlainRandom(
    std::vector<void *> &hPreExpand,
    std::function<void(void *)> freeFnPreExpand, int partyId, void *hMainCipher,
    void *hRandom) {
  std::vector<TrgswCipher *> out(_numParty * _numParty, nullptr);
  if (partyId < 0 || partyId >= _numParty || !hMainCipher || !hRandom)
    return out;
  // Copy random from host to device
  TorusInteger *random =
      (TorusInteger *)MemoryManagement::mallocMM(getSizeRandom());
  MemoryManagement::memcpyMM_h2d(random, hRandom, getSizeRandom());
  // Copy main cipher
  for (int i = 0; i < _numParty; i++) {
    out[i * _numParty + i] = new TrgswCipher(_N, 1, _l, 1, _m * _N * _sdFresh,
                                             _m * _N * _sdFresh * _sdFresh);
    MemoryManagement::memcpyMM_h2d(out[i * _numParty + i]->_data, hMainCipher,
                                   getSizeMainCipher(), _stream[i]);
  }
  // Create extend cipher
  int sizePreExpandVec = hPreExpand.size();
  for (int i = 0; i < _numParty; i++) {
    if (i == partyId)
      continue;
    void *hPreExpandPtr = nullptr;
    if (i < sizePreExpandVec)
      hPreExpandPtr = hPreExpand[i];
    out[i * _numParty + partyId] =
        _extendWithPlainRandom(hPreExpandPtr, i, random);
    if (hPreExpandPtr != nullptr && freeFnPreExpand != nullptr) {
      freeFnPreExpand(hPreExpand[i]);
      hPreExpand[i] = nullptr;
    }
  }
  // Free all allocated memory
  MemoryManagement::freeMM(random);
  // Wait all streams
  for (int i = 0; i < _numParty; i++)
    Stream::synchronizeS(_stream[i]);
  return out;
}
TorusInteger MpcApplication::partDec(std::vector<TrgswCipher *> &cipher) {
  int sizeCipher = cipher.size();
  TorusInteger out = 0;
  if (sizeCipher != _numParty * _numParty ||
      !cipher[_numParty * (_numParty - 1) + _partyId])
    return out;
  // Decrypt: get raw plain + error
  TorusInteger *plainWithError =
      (TorusInteger *)MemoryManagement::mallocMM(_N * sizeof(TorusInteger));
  TrlweFunction::getPlain(
      &_fft_privkey, 0,
      cipher[_numParty * (_numParty - 1) + _partyId]->get_trlwe_data(_l), _N, 1,
      plainWithError);
  _fft_privkey.waitOut(0);
  // Move raw plain + error from device to host
  MemoryManagement::memcpyMM_d2h(&out, plainWithError, sizeof(TorusInteger));
  // Free all allocated memory
  MemoryManagement::freeMM(plainWithError);
  return out;
}
bool MpcApplication::finDec(TorusInteger partDecPlain[], size_t numParty,
                            double *outError) {
  TorusInteger x = 0;
  for (size_t i = 0; i < numParty; i++)
    x += partDecPlain[i];
  double y = std::abs(x / std::pow(2, 8 * sizeof(TorusInteger)));
  if (outError)
    *outError = (y < 0.25) ? y : (0.5 - y);
  return (y >= 0.25);
}
