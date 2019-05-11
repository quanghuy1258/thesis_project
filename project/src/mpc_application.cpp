#include "thesis/batched_fft.h"
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
  _stream.resize(2 * _l * _m + 1,
                 nullptr); // TODO: Always check maximum size when use
  for (int i = 0; i <= 2 * _l * _m; i++)
    _stream[i] = Stream::createS();
}
MpcApplication::~MpcApplication() {
  MemoryManagement::freeMM(_privkey);
  MemoryManagement::freeMM(_pubkey[0]->_data);
  for (int i = 0; i < _m; i++)
    delete _pubkey[i];
  for (int i = 0; i <= 2 * _l * _m; i++)
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
void MpcApplication::encrypt(bool msg, void *hCipher, void *hRandom) {
  TorusInteger *ptr = (TorusInteger *)hCipher;
  // Construct cipher
  std::vector<TrgswCipher *> cipher(2 * _l * _m + 1);
  for (int i = 0; i <= 2 * _l * _m; i++)
    cipher[i] = new TrgswCipher(_N, 1, _l, 1, _sdFresh, _sdFresh * _sdFresh);
  // Create random
  TorusInteger *random_ptr =
      (TorusInteger *)MemoryManagement::mallocMM(getSizeRandom());
  Random::setUniform(random_ptr, 2 * _l * _m * _N,
                     [](TorusInteger x) -> TorusInteger { return x & 1; });
  // Init main cipher
  cipher[2 * _l * _m]->clear_trgsw_data();
  for (int i = 0; i < 2 * _l; i++)
    Random::setNormalTorus(cipher[2 * _l * _m]->get_pol_data(i, 1), _N,
                           _sdFresh);
  // main cipher += random * pubkey
  for (int i = 0; i < 2 * _l; i++) {
    for (int j = 0; j < _m; j++)
      _fft_pubkey.setInp(random_ptr + (i * _m + j) * _N, j);
    for (int j = 0; j < _m; j++) {
      _fft_pubkey.setMul(0, j);
      _fft_pubkey.setMul(1, j);
    }
    _fft_pubkey.addAllOut(cipher[2 * _l * _m]->get_pol_data(i, 0), 0);
    _fft_pubkey.addAllOut(cipher[2 * _l * _m]->get_pol_data(i, 1), 1);
  }
  _fft_pubkey.waitAllOut();
  // main cipher += msg*G;
  if (msg)
    TrgswFunction::addMuGadget(1, cipher[2 * _l * _m], _stream[2 * _l * _m]);
  // Create cipher for random
  for (int i = 0; i < 2 * _l * _m; i++) {
    for (int j = 0; j < _l; j++)
      TrlweFunction::createSample(&_fft_privkey, (_l * i + j) & 1,
                                  cipher[i]->get_trlwe_data(_l + j), _N, 1,
                                  _sdFresh);
  }
  _fft_privkey.waitAllOut();
  for (int i = 0; i < 2 * _l * _m; i++)
    TrgswFunction::addMuGadget(random_ptr + _N * i, cipher[i], _stream[i]);
  // Copy from device to host
  for (int i = 0; i < 2 * _l * _m; i++)
    MemoryManagement::memcpyMM_d2h(
        ptr + 2 * _l * _N * i, cipher[i]->get_trlwe_data(_l),
        2 * _l * _N * sizeof(TorusInteger), _stream[i]);
  MemoryManagement::memcpyMM_d2h(
      ptr + 2 * _l * _N * 2 * _l * _m, cipher[2 * _l * _m]->_data,
      4 * _l * _N * sizeof(TorusInteger), _stream[2 * _l * _m]);
  // Copy random if possible
  if (hRandom)
    MemoryManagement::memcpyMM_d2h(hRandom, random_ptr, getSizeRandom());
  // Wait all streams
  for (int i = 0; i <= 2 * _l * _m; i++)
    Stream::synchronizeS(_stream[i]);
  // Destroy random
  MemoryManagement::freeMM(random_ptr);
  // Destruct cipher
  for (int i = 0; i <= 2 * _l * _m; i++)
    delete cipher[i];
}
int MpcApplication::getSizeCipher() {
  return (2 * _l * _m * 2 + 4) * _l * _N * sizeof(TorusInteger);
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
    _fft_pubkey.setInp(pubkey_ptr + 2 * _N * i, i & 1, 0);
    _fft_pubkey.setMul(i & 1, 0);
    TorusUtility::subVector(preExpand_ptr + _N * i,
                            pubkey_ptr + (2 * i + 1) * _N, _N);
    _fft_pubkey.addAllOut(preExpand_ptr + _N * i, i & 1);
  }
  _fft_pubkey.waitAllOut();
  // Copy preExpand_ptr from device to host
  MemoryManagement::memcpyMM_d2h(hPreExpand, preExpand_ptr, getSizePreExpand());
  // Delete pubkey and preExpand
  MemoryManagement::freeMM(pubkey_ptr);
  MemoryManagement::freeMM(preExpand_ptr);
}
int MpcApplication::getSizePreExpand() {
  return _m * _N * sizeof(TorusInteger);
}
TorusInteger *MpcApplication::_decompPreExpand(void *hPreExpand, int id,
                                               TrgswCipher *param) {
  // preExpandCipher for only decomposition -> choosing any value for sd and var
  // is ok
  TrlweCipher preExpandCipher(_N, _m, 1, 1);
  MemoryManagement::memcpyMM_h2d(preExpandCipher._data, hPreExpand,
                                 getSizePreExpand());
  TorusInteger *decompPreExpand = (TorusInteger *)MemoryManagement::mallocMM(
      (_m + 1) * _N * sizeof(TorusInteger) * _l);
  Decomposition::onlyDecomp(&preExpandCipher, param, decompPreExpand);
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
  TrgswCipher *out = new TrgswCipher(
      _N, 1, _l, 1, _m * ((_l + 1) * _N * _sdFresh + _N * (1 + _N) * e_dec),
      _m * ((_l + 1) * _N * _sdFresh * _sdFresh +
            _N * (1 + _N) * e_dec * e_dec));
  // Determine the position of partyId in _fft_with_preExpand
  int id = (partyId < _partyId) ? partyId : (partyId - 1);
  // Do decomposition if hPreExpand is not null
  TorusInteger *decompPreExpand = nullptr;
  if (hPreExpand)
    decompPreExpand = _decompPreExpand(hPreExpand, id, out);
  // Clear data of out
  out->clear_trgsw_data();
  // Calculate extend cipher
  for (int i = 0; i < 2 * _l; i++) {
    for (int j = 0; j < _m; j++) {
      for (int k = 0; k < _l; k++)
        _fft_preExpand.setInp(cipher + ((i * _m + j) * 2 * _l + 2 * k) * _N, k);
      for (int k = 0; k < _l; k++)
        _fft_preExpand.setMul(id * _m + j, k);
      _fft_preExpand.addAllOut(out->get_pol_data(i, 0), id * _m + j);
      for (int k = 0; k < _l; k++)
        _fft_preExpand.setInp(cipher + ((i * _m + j) * 2 * _l + 2 * k + 1) * _N,
                              k);
      for (int k = 0; k < _l; k++)
        _fft_preExpand.setMul(id * _m + j, k);
      _fft_preExpand.addAllOut(out->get_pol_data(i, 1), id * _m + j);
    }
  }
  _fft_preExpand.waitAllOut();
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
                       void *hCipher) {
  std::vector<TrgswCipher *> out(_numParty * _numParty, nullptr);
  if (partyId < 0 || partyId >= _numParty || !hCipher)
    return out;
  // Copy cipher from host to device
  TorusInteger *cipher =
      (TorusInteger *)MemoryManagement::mallocMM(getSizeCipher());
  MemoryManagement::memcpyMM_h2d(cipher, hCipher, getSizeCipher());
  // Copy main cipher
  for (int i = 0; i < _numParty; i++) {
    out[i * _numParty + i] = new TrgswCipher(_N, 1, _l, 1, _m * _N * _sdFresh,
                                             _m * _N * _sdFresh * _sdFresh);
    MemoryManagement::memcpyMM_d2d(out[i * _numParty + i]->_data,
                                   cipher + 2 * _l * _m * 2 * _l * _N,
                                   4 * _l * _N * sizeof(TorusInteger));
  }
  // Create extend cipher
  int sizePreExpandVec = hPreExpand.size();
  for (int i = 0; i < _numParty; i++) {
    if (i == partyId)
      continue;
    void *hPreExpandPtr = nullptr;
    if (i < sizePreExpandVec)
      hPreExpandPtr = hPreExpand[i];
    out[i * _numParty + partyId] = _extend(hPreExpandPtr, partyId, cipher);
    if (hPreExpandPtr != nullptr && freeFnPreExpand != nullptr) {
      freeFnPreExpand(hPreExpand[i]);
      hPreExpand[i] = nullptr;
    }
  }
  // Free all allocated memory
  MemoryManagement::freeMM(cipher);
  return out;
}
std::vector<TrgswCipher *>
MpcApplication::expand(std::vector<void *> &hPreExpand,
                       std::function<void(void *)> freeFnPreExpand, int partyId,
                       void *hCipher, void *hRandom) {
  std::vector<TrgswCipher *> out(_numParty * _numParty, nullptr);
  if (partyId < 0 || partyId >= _numParty || !hCipher || !hRandom)
    return out;
  // Copy random from host to device
  TorusInteger *random =
      (TorusInteger *)MemoryManagement::mallocMM(getSizeRandom());
  MemoryManagement::memcpyMM_h2d(random, hRandom, getSizeRandom());
  // Copy main cipher
  TorusInteger *cipher =
      (TorusInteger *)MemoryManagement::mallocMM(getSizeCipher());
  MemoryManagement::memcpyMM_h2d(cipher, hCipher, getSizeCipher());
  for (int i = 0; i < _numParty; i++) {
    out[i * _numParty + i] = new TrgswCipher(_N, 1, _l, 1, _m * _N * _sdFresh,
                                             _m * _N * _sdFresh * _sdFresh);
    MemoryManagement::memcpyMM_d2d(out[i * _numParty + i]->_data,
                                   cipher + 2 * _l * _m * 2 * _l * _N,
                                   4 * _l * _N * sizeof(TorusInteger));
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
        _extendWithPlainRandom(hPreExpandPtr, partyId, random);
    if (hPreExpandPtr != nullptr && freeFnPreExpand != nullptr) {
      freeFnPreExpand(hPreExpand[i]);
      hPreExpand[i] = nullptr;
    }
  }
  // Free all allocated memory
  MemoryManagement::freeMM(random);
  MemoryManagement::freeMM(cipher);
  return out;
}
TorusInteger MpcApplication::partDec(std::vector<TrgswCipher *> &cipher) {
  int sizeCipher = cipher.size();
  TorusInteger out;
  if (sizeCipher != _numParty * _numParty ||
      !cipher[_numParty * (_numParty - 1) + _partyId])
    return 0;
  // Decrypt: get raw plain + error
  TorusInteger *plainWithError =
      (TorusInteger *)MemoryManagement::mallocMM(_N * sizeof(TorusInteger));
  TrlweFunction::getPlain(&_fft_privkey, 0,
                          cipher[_numParty * (_numParty - 1) + _partyId]->_data,
                          _N, 1, plainWithError);
  _fft_privkey.waitOut(0);
  // Move raw plain + error from device to host
  MemoryManagement::memcpyMM_d2h(&out, plainWithError, sizeof(TorusInteger));
  // Free all allocated memory
  MemoryManagement::freeMM(plainWithError);
  return out;
}
bool MpcApplication::finDec(std::vector<TorusInteger> partDecPlain,
                            double *outError) {
  TorusInteger x = 0;
  for (auto p : partDecPlain)
    x += p;
  double y = std::abs(x / std::pow(2, 8 * sizeof(TorusInteger)));
  if (outError)
    *outError = (y < 0.25) ? y : (0.5 - y);
  return (y >= 0.25);
}
