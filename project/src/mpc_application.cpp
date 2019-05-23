#include "thesis/decomposition.h"
#include "thesis/memory_management.h"
#include "thesis/random.h"
#include "thesis/stream.h"
#include "thesis/torus_utility.h"
#include "thesis/trgsw_function.h"
#include "thesis/trlwe_function.h"

#include "mpc_application.h"

#define WARNING_CERR(msg) std::cerr << "WARNING: " msg << std::endl

using namespace thesis;

MpcApplication::MpcApplication(int numParty, int partyId, int N, int m, int l,
                               double sdFresh) {
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
  _fft_privkey = nullptr;
  _fft_pubkey = nullptr;
  _fft_preExpand = nullptr;
  _fft_preExpandRandom = nullptr;
  _fft_mul = nullptr;
  _privkey = (TorusInteger *)MemoryManagement::mallocMM(getSizePrivkey());
  _pubkey.resize(m, nullptr);
  TorusInteger *mem_pubkey =
      (TorusInteger *)MemoryManagement::mallocMM(getSizePubkey());
  for (int i = 0; i < _m; i++)
    _pubkey[i] = new TrlweCipher(mem_pubkey + 2 * N * i, 2 * N, N, 1, sdFresh,
                                 sdFresh * sdFresh);
  int numStream = 0; // TODO: Always check maximum size when use
  numStream = std::max(numStream, 2 * _l * _m + 2);
  numStream = std::max(numStream, 2 * _l * _numParty + 1);
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
  numStream = std::max(numStream, 2 * _l * _numParty + 1);
  for (int i = 0; i < numStream; i++)
    Stream::destroyS(_stream[i]);
  if (_fft_privkey)
    delete _fft_privkey;
  if (_fft_pubkey)
    delete _fft_pubkey;
  if (_fft_preExpand)
    delete _fft_preExpand;
  if (_fft_preExpandRandom)
    delete _fft_preExpandRandom;
  if (_fft_mul)
    delete _fft_mul;
}
void MpcApplication::createPrivkey() {
  if (!_fft_privkey)
    _fft_privkey = new BatchedFFT(_N, 2, 1);
  TrlweFunction::genkey(_privkey, _N, 1);
  TrlweFunction::keyToFFT(_privkey, _N, 1, _fft_privkey);
}
void MpcApplication::importPrivkey(void *hPrivkey) {
  if (!hPrivkey) {
    WARNING_CERR("hPrivkey is not NULL");
    return;
  }
  if (!_fft_privkey)
    _fft_privkey = new BatchedFFT(_N, 2, 1);
  MemoryManagement::memcpyMM_h2d(_privkey, hPrivkey, getSizePrivkey());
  TrlweFunction::keyToFFT(_privkey, _N, 1, _fft_privkey);
}
void MpcApplication::exportPrivkey(void *hPrivkey) {
  if (!hPrivkey) {
    WARNING_CERR("hPrivkey is not NULL");
    return;
  }
  MemoryManagement::memcpyMM_d2h(hPrivkey, _privkey, getSizePrivkey());
}
int MpcApplication::getSizePrivkey() { return _N * sizeof(TorusInteger); }
void MpcApplication::createPubkey() {
  if (!_fft_privkey)
    throw std::runtime_error("ERROR: Must create or import private key");
  if (!_fft_pubkey)
    _fft_pubkey = new BatchedFFT(_N, 2, _m);
  for (int i = 0; i < _m; i++) {
    TrlweFunction::createSample(_fft_privkey, i & 1, _pubkey[i]);
    _fft_privkey->waitOut(i & 1);
    _fft_pubkey->setInp(_pubkey[i]->get_pol_data(0), 0, i);
    _fft_pubkey->setInp(_pubkey[i]->get_pol_data(1), 1, i);
  }
}
void MpcApplication::importPubkey(void *hPubkey) {
  if (!hPubkey) {
    WARNING_CERR("hPubkey is not NULL");
    return;
  }
  if (!_fft_pubkey)
    _fft_pubkey = new BatchedFFT(_N, 2, _m);
  MemoryManagement::memcpyMM_h2d(_pubkey[0]->_data, hPubkey, getSizePubkey());
  for (int i = 0; i < _m; i++) {
    _fft_pubkey->setInp(_pubkey[i]->get_pol_data(0), 0, i);
    _fft_pubkey->setInp(_pubkey[i]->get_pol_data(1), 1, i);
  }
}
void MpcApplication::exportPubkey(void *hPubkey) {
  if (!hPubkey) {
    WARNING_CERR("hPubkey is not NULL");
    return;
  }
  MemoryManagement::memcpyMM_d2h(hPubkey, _pubkey[0]->_data, getSizePubkey());
}
int MpcApplication::getSizePubkey() {
  return _N * 2 * sizeof(TorusInteger) * _m;
}
void MpcApplication::encrypt(bool msg, void *hMainCipher, void *hRandCipher,
                             void *hRandom) {
  if (!hMainCipher) {
    WARNING_CERR("hMainCipher is not NULL");
    return;
  }
  if (!_fft_pubkey)
    throw std::runtime_error("ERROR: Must create or import public key");
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
    if (!_fft_privkey)
      throw std::runtime_error("ERROR: Must create or import private key");
    for (int i = 0; i < 2 * _l * _m; i++)
      randCipher[i] =
          new TrgswCipher(_N, 1, _l, 1, _sdFresh, _sdFresh * _sdFresh);
    for (int i = 0; i < 2 * _l * _m; i++) {
      for (int j = 0; j < _l; j++)
        TrlweFunction::createSample(_fft_privkey, (_l * i + j) & 1,
                                    randCipher[i]->get_trlwe_data(_l + j), _N,
                                    1, _sdFresh);
    }
    _fft_privkey->waitAllOut();
    for (int i = 0; i < 2 * _l * _m; i++)
      TrgswFunction::addMuGadget(random + _N * i, randCipher[i],
                                 _stream[i + 2]);
    TorusInteger *ptr = (TorusInteger *)hRandCipher;
    for (int i = 0; i < 2 * _l * _m; i++)
      MemoryManagement::memcpyMM_d2h(
          ptr + 2 * _l * _N * i, randCipher[i]->get_trlwe_data(_l),
          getSizeRandCipher() / (2 * _l * _m), _stream[i + 2]);
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
      _fft_pubkey->setInp(random + (i * _m + j) * _N, j);
    for (int j = 0; j < _m; j++) {
      _fft_pubkey->setMul(0, j);
      _fft_pubkey->setMul(1, j);
    }
    _fft_pubkey->addAllOut(mainCipher.get_pol_data(i, 0), 0);
    _fft_pubkey->addAllOut(mainCipher.get_pol_data(i, 1), 1);
  }
  _fft_pubkey->waitAllOut();
  // main cipher += msg*G;
  if (msg)
    TrgswFunction::addMuGadget(1, &mainCipher, _stream[0]);
  // Copy main cipher from device to host
  MemoryManagement::memcpyMM_d2h(hMainCipher, mainCipher._data,
                                 getSizeMainCipher(), _stream[0]);
  // Wait all streams
  for (int i = 0; i < 2 * _l * _m + 2; i++)
    Stream::synchronizeS(_stream[i]);
  // Delete all
  MemoryManagement::freeMM(random);
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
  if (!hPubkey || !hPreExpand) {
    WARNING_CERR("hPubkey and hPreExpand is not NULL");
    return;
  }
  if (!_fft_privkey)
    throw std::runtime_error("ERROR: Must create or import private key");
  // Init pubkey and preExpand memory on VRAM
  TorusInteger *pubkey_ptr =
      (TorusInteger *)MemoryManagement::mallocMM(getSizePubkey());
  MemoryManagement::memcpyMM_h2d(pubkey_ptr, hPubkey, getSizePubkey());
  TorusInteger *preExpand_ptr =
      (TorusInteger *)MemoryManagement::mallocMM(getSizePreExpand());
  Random::setNormalTorus(preExpand_ptr, _m * _N, _sdFresh);
  // preExpand_ptr -= hPubkey * privkey
  for (int i = 0; i < _m; i++) {
    _fft_privkey->setInp(pubkey_ptr + 2 * _N * i, i & 1, 0);
    _fft_privkey->setMul(i & 1, 0);
    TorusUtility::subVector(preExpand_ptr + _N * i,
                            pubkey_ptr + (2 * i + 1) * _N, _N);
    _fft_privkey->addAllOut(preExpand_ptr + _N * i, i & 1);
  }
  _fft_privkey->waitAllOut();
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
  if (!_fft_preExpand)
    _fft_preExpand = new BatchedFFT(_N, _m * (_numParty - 1), _l);
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
      _fft_preExpand->setInp(decompPreExpand + (i * _l + j) * _N, id * _m + i,
                             j);
  }
  return decompPreExpand;
}
void MpcApplication::_extend(void *hPreExpand, int partyId,
                             TorusInteger *cipher, int mainPartyId,
                             TrgswCipher *out) {
  if (!_fft_preExpand)
    _fft_preExpand = new BatchedFFT(_N, _m * (_numParty - 1), _l);
  // Determine the position of partyId in _fft_with_preExpand
  int id = (partyId < _partyId) ? partyId : (partyId - 1);
  // Do decomposition if hPreExpand is not null
  TorusInteger *decompPreExpand = nullptr;
  if (hPreExpand)
    decompPreExpand = _decompPreExpand(hPreExpand, id);
  // Calculate extend cipher
  for (int j = 0; j < _m; j++) {
    for (int i = 0; i < 2 * _l; i++) {
      // First column
      for (int k = 0; k < _l; k++)
        _fft_preExpand->setInp(cipher + ((i * _m + j) * 2 * _l + 2 * k) * _N,
                               k);
      for (int k = 0; k < _l; k++)
        _fft_preExpand->setMul(id * _m + j, k);
      if (j > 0)
        _fft_preExpand->waitOut(id * _m + j - 1);
      _fft_preExpand->addAllOut(
          out->get_pol_data(partyId * 2 * _l + i, 2 * mainPartyId),
          id * _m + j);
      // Second column
      for (int k = 0; k < _l; k++)
        _fft_preExpand->setInp(
            cipher + ((i * _m + j) * 2 * _l + 2 * k + 1) * _N, k);
      for (int k = 0; k < _l; k++)
        _fft_preExpand->setMul(id * _m + j, k);
      if (j > 0)
        _fft_preExpand->waitOut(id * _m + j - 1);
      _fft_preExpand->addAllOut(
          out->get_pol_data(partyId * 2 * _l + i, 2 * mainPartyId + 1),
          id * _m + j);
    }
  }
  _fft_preExpand->waitOut(id * _m + _m - 1);
  // Free all allocated memory
  if (hPreExpand)
    MemoryManagement::freeMM(decompPreExpand);
}
void MpcApplication::_extendWithPlainRandom(void *hPreExpand, int partyId,
                                            TorusInteger *random,
                                            int mainPartyId, TrgswCipher *out) {
  if (!_fft_privkey)
    throw std::runtime_error("ERROR: Must create or import private key");
  if (!_fft_preExpandRandom)
    _fft_preExpandRandom = new BatchedFFT(_N, _numParty - 1, _m);
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
      _fft_preExpandRandom->setInp(preExpandCipher + i * _N, id, i);
  }
  // Calculate extend cipher
  for (int i = 0; i < 2 * _l; i++)
    TrlweFunction::createSample(
        _fft_privkey, i & 1,
        out->get_pol_data(partyId * 2 * _l + i, 2 * mainPartyId), _N, 1,
        _sdFresh);
  _fft_privkey->waitAllOut();
  for (int i = 0; i < 2 * _l; i++) {
    for (int j = 0; j < _m; j++)
      _fft_preExpandRandom->setInp(random + (i * _m + j) * _N, j);
    for (int j = 0; j < _m; j++)
      _fft_preExpandRandom->setMul(id, j);
    _fft_preExpandRandom->addAllOut(
        out->get_pol_data(partyId * 2 * _l + i, 2 * mainPartyId + 1), id);
  }
  _fft_preExpandRandom->waitAllOut();
  // Free all allocated memory
  if (hPreExpand)
    MemoryManagement::freeMM(preExpandCipher);
}
TrgswCipher *MpcApplication::expand(std::vector<void *> &hPreExpand,
                                    std::function<void(void *)> freeFnPreExpand,
                                    int partyId, void *hMainCipher,
                                    void *hRandCipher) {
  if (partyId < 0 || partyId >= _numParty || !hMainCipher || !hRandCipher) {
    WARNING_CERR(
        "0 <= partyId < _numParty ; hMainCipher and hRandCipher is not NULL");
    return nullptr;
  }
  // Copy cipher from host to device
  TorusInteger *randCipher =
      (TorusInteger *)MemoryManagement::mallocMM(getSizeRandCipher());
  MemoryManagement::memcpyMM_h2d(randCipher, hRandCipher, getSizeRandCipher(),
                                 _stream[0]);
  // Init expanded cipher
  double e_dec = std::pow(2, -_l - 1);
  //   sdError = m * l * N * sdFresh + N * e_dec * m + m * N * sdFresh
  //   varError = m * l * N * (sdFresh ^ 2) + N * (e_dec ^ 2) * m +
  //              m * N * (sdFresh ^ 2)
  TrgswCipher *out = new TrgswCipher(
      _N, 2 * _numParty - 1, _l, 1, _m * _N * ((_l + 1) * _sdFresh + e_dec),
      _m * _N * ((_l + 1) * _sdFresh * _sdFresh + e_dec * e_dec));
  out->clear_trgsw_data();
  // Copy main cipher
  TrgswCipher mainCipher((TorusInteger *)hMainCipher, 4 * _l * _N, _N, 1, _l, 1,
                         _sdFresh, _sdFresh * _sdFresh);
  for (int i = 0; i < _numParty; i++) {
    for (int j = 0; j < 2 * _l; j++)
      MemoryManagement::memcpyMM_h2d(out->get_pol_data(i * 2 * _l + j, 2 * i),
                                     mainCipher.get_trlwe_data(j),
                                     2 * _N * sizeof(TorusInteger),
                                     _stream[i * 2 * _l + j + 1]);
  }
  // Create extend cipher
  Stream::synchronizeS(_stream[0]);
  int sizePreExpandVec = hPreExpand.size();
  for (int i = 0; i < _numParty; i++) {
    if (i == partyId)
      continue;
    void *hPreExpandPtr = nullptr;
    if (i < sizePreExpandVec)
      hPreExpandPtr = hPreExpand[i];
    _extend(hPreExpandPtr, i, randCipher, partyId, out);
    if (hPreExpandPtr != nullptr && freeFnPreExpand != nullptr) {
      freeFnPreExpand(hPreExpand[i]);
      hPreExpand[i] = nullptr;
    }
  }
  // Free all allocated memory
  MemoryManagement::freeMM(randCipher);
  // Wait all streams
  for (int i = 1; i < 2 * _l * _numParty + 1; i++)
    Stream::synchronizeS(_stream[i]);
  return out;
}
TrgswCipher *MpcApplication::expandWithPlainRandom(
    std::vector<void *> &hPreExpand,
    std::function<void(void *)> freeFnPreExpand, int partyId, void *hMainCipher,
    void *hRandom) {
  if (partyId < 0 || partyId >= _numParty || !hMainCipher || !hRandom) {
    WARNING_CERR(
        "0 <= partyId < _numParty ; hMainCipher and hRandCipher is not NULL");
    return nullptr;
  }
  // Copy random from host to device
  TorusInteger *random =
      (TorusInteger *)MemoryManagement::mallocMM(getSizeRandom());
  MemoryManagement::memcpyMM_h2d(random, hRandom, getSizeRandom(), _stream[0]);
  // Init expanded cipher
  //   sdError = (1 + m * N) * sdFresh
  //   varError = (1 + m * N) * (SdFresh ^ 2)
  TrgswCipher *out =
      new TrgswCipher(_N, 2 * _numParty - 1, _l, 1, (1 + _m * _N) * _sdFresh,
                      (1 + _m * _N) * _sdFresh * _sdFresh);
  out->clear_trgsw_data();
  // Copy main cipher
  TrgswCipher mainCipher((TorusInteger *)hMainCipher, 4 * _l * _N, _N, 1, _l, 1,
                         _sdFresh, _sdFresh * _sdFresh);
  for (int i = 0; i < _numParty; i++) {
    for (int j = 0; j < 2 * _l; j++)
      MemoryManagement::memcpyMM_h2d(out->get_pol_data(i * 2 * _l + j, 2 * i),
                                     mainCipher.get_trlwe_data(j),
                                     2 * _N * sizeof(TorusInteger),
                                     _stream[i * 2 * _l + j + 1]);
  }
  // Create extend cipher
  Stream::synchronizeS(_stream[0]);
  int sizePreExpandVec = hPreExpand.size();
  for (int i = 0; i < _numParty; i++) {
    if (i == partyId)
      continue;
    void *hPreExpandPtr = nullptr;
    if (i < sizePreExpandVec)
      hPreExpandPtr = hPreExpand[i];
    _extendWithPlainRandom(hPreExpandPtr, i, random, partyId, out);
    if (hPreExpandPtr != nullptr && freeFnPreExpand != nullptr) {
      freeFnPreExpand(hPreExpand[i]);
      hPreExpand[i] = nullptr;
    }
  }
  // Free all allocated memory
  MemoryManagement::freeMM(random);
  // Wait all streams
  for (int i = 1; i < 2 * _l * _numParty + 1; i++)
    Stream::synchronizeS(_stream[i]);
  return out;
}
TorusInteger MpcApplication::partDec(TrgswCipher *cipher) {
  if (!_fft_privkey)
    throw std::runtime_error("ERROR: Must create or import private key");
  // Init out value
  TorusInteger out = 0;
  if (!cipher || cipher->_N != _N || cipher->_k != 2 * _numParty - 1 ||
      cipher->_l != _l || cipher->_Bgbit != 1 || cipher->_sdError < 0 ||
      cipher->_varError < 0) {
    WARNING_CERR("Cannot part decrypt");
    return out;
  }
  // Decrypt: get raw plain + error
  TorusInteger *plainWithError =
      (TorusInteger *)MemoryManagement::mallocMM(_N * sizeof(TorusInteger));
  TrlweFunction::getPlain(
      _fft_privkey, 0,
      cipher->get_pol_data((2 * _numParty - 1) * _l, 2 * _partyId), _N, 1,
      plainWithError);
  _fft_privkey->waitOut(0);
  // Move raw plain + error from device to host
  MemoryManagement::memcpyMM_d2h(&out, plainWithError, sizeof(TorusInteger));
  // Free all allocated memory
  MemoryManagement::freeMM(plainWithError);
  return out;
}
bool MpcApplication::finDec(TorusInteger partDecPlain[], double *outError) {
  TorusInteger x = 0;
  for (int i = 0; i < _numParty; i++)
    x += partDecPlain[i];
  double y = std::abs(x / std::pow(2, 8 * sizeof(TorusInteger)));
  if (outError)
    *outError = (y < 0.25) ? y : (0.5 - y);
  return (y >= 0.25);
}
TrgswCipher *MpcApplication::importExpandedCipher(void *inp) {
  double *ptr = (double *)inp;
  if (ptr[0] < 0 || ptr[1] < 0) {
    WARNING_CERR("sdError, varError >= 0");
    return nullptr;
  }
  TrgswCipher *out =
      new TrgswCipher(_N, 2 * _numParty - 1, _l, 1, ptr[0], ptr[1]);
  MemoryManagement::memcpyMM_h2d(out->_data, ptr + 2,
                                 _numParty * _numParty * getSizeMainCipher());
  return out;
}
void MpcApplication::exportExpandedCipher(thesis::TrgswCipher *inp, void *out) {
  double *ptr = (double *)out;
  ptr[0] = inp->_sdError;
  ptr[1] = inp->_varError;
  MemoryManagement::memcpyMM_d2h(ptr + 2, inp->_data,
                                 _numParty * _numParty * getSizeMainCipher());
}
int MpcApplication::getSizeExpandedCipher() {
  // sdError | varError | data
  return 2 * sizeof(double) + _numParty * _numParty * getSizeMainCipher();
}
TrgswCipher *MpcApplication::addOp(TrgswCipher *inp_1, TrgswCipher *inp_2) {
  if (!inp_1 || !inp_2) {
    WARNING_CERR("inp_1 and inp_2 is not NULL");
    return nullptr;
  }
  TrgswCipher *out = new TrgswCipher(_N, 2 * _numParty - 1, _l, 1,
                                     inp_1->_sdError + inp_2->_sdError,
                                     inp_1->_varError + inp_2->_varError);
  MemoryManagement::memcpyMM_d2d(out->_data, inp_1->_data,
                                 _numParty * _numParty * getSizeMainCipher());
  TorusUtility::addVector(out->_data, inp_2->_data,
                          _numParty * _numParty * 4 * _l * _N);
  return out;
}
TrgswCipher *MpcApplication::subOp(TrgswCipher *inp_1, TrgswCipher *inp_2) {
  if (!inp_1 || !inp_2) {
    WARNING_CERR("inp_1 and inp_2 is not NULL");
    return nullptr;
  }
  TrgswCipher *out = new TrgswCipher(_N, 2 * _numParty - 1, _l, 1,
                                     inp_1->_sdError + inp_2->_sdError,
                                     inp_1->_varError + inp_2->_varError);
  MemoryManagement::memcpyMM_d2d(out->_data, inp_1->_data,
                                 _numParty * _numParty * getSizeMainCipher());
  TorusUtility::subVector(out->_data, inp_2->_data,
                          _numParty * _numParty * 4 * _l * _N);
  return out;
}
TrgswCipher *MpcApplication::notOp(TrgswCipher *inp) {
  if (!inp) {
    WARNING_CERR("inp is not NULL");
    return nullptr;
  }
  TrgswCipher *out = new TrgswCipher(_N, 2 * _numParty - 1, _l, 1,
                                     inp->_sdError, inp->_varError);
  out->clear_trgsw_data();
  TrgswFunction::addMuGadget(1, out);
  TorusUtility::subVector(out->_data, inp->_data,
                          _numParty * _numParty * 4 * _l * _N);
  return out;
}
TrgswCipher *MpcApplication::notXorOp(TrgswCipher *inp_1, TrgswCipher *inp_2) {
  if (!inp_1 || !inp_2) {
    WARNING_CERR("inp_1 and inp_2 is not NULL");
    return nullptr;
  }
  TrgswCipher *out = new TrgswCipher(_N, 2 * _numParty - 1, _l, 1,
                                     inp_1->_sdError + inp_2->_sdError,
                                     inp_1->_varError + inp_2->_varError);
  out->clear_trgsw_data();
  TrgswFunction::addMuGadget(1, out);
  TorusUtility::subVector(out->_data, inp_1->_data,
                          _numParty * _numParty * 4 * _l * _N);
  TorusUtility::subVector(out->_data, inp_2->_data,
                          _numParty * _numParty * 4 * _l * _N);
  return out;
}
TrgswCipher *MpcApplication::mulOp(TrgswCipher *inp_1, TrgswCipher *inp_2) {
  if (!inp_1 || !inp_2) {
    WARNING_CERR("inp_1 and inp_2 is not NULL");
    return nullptr;
  }
  if (!_fft_mul)
    _fft_mul = new BatchedFFT(_N, 2 * _numParty, 2 * _l * _numParty);
  // Choose input which will be decomposed
  TrgswCipher *inp_decomp = inp_1;
  TrgswCipher *inp_ori = inp_2;
  if (inp_1->_varError < inp_2->_varError) {
    inp_decomp = inp_2;
    inp_ori = inp_1;
  }
  // Prepare FFT for multiplication
  for (int i = 0; i < 2 * _l * _numParty; i++) {
    for (int j = 0; j < 2 * _numParty; j++)
      _fft_mul->setInp(inp_ori->get_pol_data(i, j), j, i);
  }
  // Init output
  double e_dec = std::pow(2, -_l - 1);
  //   A: ori
  //   B: decomp
  //   sdError = 2 * l * numParty * N * sdA + numParty * (N + 1) * e_dec
  //             + sdB
  //   varError = 2 * l * numParty * N * varA + numParty * (N + 1) * (e_dec ^ 2)
  //              + varB
  TrgswCipher *out = new TrgswCipher(
      _N, 2 * _numParty - 1, _l, 1,
      2 * _l * _numParty * _N * inp_ori->_sdError +
          _numParty * (_N + 1) * e_dec + inp_decomp->_sdError,
      2 * _l * _numParty * _N * inp_ori->_varError +
          _numParty * (_N + 1) * e_dec * e_dec + inp_decomp->_varError);
  out->clear_trgsw_data();
  // Decomposition and multiplication
  TorusInteger *decomp_ptr = (TorusInteger *)MemoryManagement::mallocMM(
      2 * _l * _numParty * _N * sizeof(TorusInteger));
  for (int i = 0; i < 2 * _l * _numParty; i++) {
    TrlweCipher row(inp_decomp->get_trlwe_data(i), 2 * _numParty * _N, _N,
                    2 * _numParty - 1, inp_decomp->_sdError,
                    inp_decomp->_varError);
    Decomposition::onlyDecomp(&row, out, decomp_ptr);
    for (int j = 0; j < 2 * _l * _numParty; j++)
      _fft_mul->setInp(decomp_ptr + j * _N, j);
    for (int j = 0; j < 2 * _numParty; j++) {
      for (int k = 0; k < 2 * _l * _numParty; k++)
        _fft_mul->setMul(j, k);
    }
    for (int j = 0; j < 2 * _numParty; j++)
      _fft_mul->addAllOut(out->get_pol_data(i, j), j);
  }
  // Wait all
  _fft_mul->waitAllOut();
  // Free all
  MemoryManagement::freeMM(decomp_ptr);
  return out;
}
TrlweCipher *MpcApplication::reduce(TrgswCipher *inp) {
  if (!inp) {
    WARNING_CERR("inp is not NULL");
    return nullptr;
  }
  TrlweCipher *out =
      new TrlweCipher(_N, 2 * _numParty - 1, inp->_sdError, inp->_varError);
  MemoryManagement::memcpyMM_d2d(out->_data,
                                 inp->get_trlwe_data((2 * _numParty - 1) * _l),
                                 getSizeReducedCipher() - 2 * sizeof(double));
  return out;
}
TorusInteger MpcApplication::partDec(TrlweCipher *cipher) {
  if (!_fft_privkey)
    throw std::runtime_error("ERROR: Must create or import private key");
  // Init out value
  TorusInteger out = 0;
  if (!cipher || cipher->_N != _N || cipher->_k != 2 * _numParty - 1 ||
      cipher->_sdError < 0 || cipher->_varError < 0) {
    WARNING_CERR("Cannot part decrypt");
    return out;
  }
  // Decrypt: get raw plain + error
  TorusInteger *plainWithError =
      (TorusInteger *)MemoryManagement::mallocMM(_N * sizeof(TorusInteger));
  TrlweFunction::getPlain(_fft_privkey, 0, cipher->get_pol_data(2 * _partyId),
                          _N, 1, plainWithError);
  _fft_privkey->waitOut(0);
  // Move raw plain + error from device to host
  MemoryManagement::memcpyMM_d2h(&out, plainWithError, sizeof(TorusInteger));
  // Free all allocated memory
  MemoryManagement::freeMM(plainWithError);
  return out;
}
TrlweCipher *MpcApplication::importReducedCipher(void *inp) {
  double *ptr = (double *)inp;
  if (ptr[0] < 0 || ptr[1] < 0) {
    WARNING_CERR("sdError, varError >= 0");
    return nullptr;
  }
  TrlweCipher *out = new TrlweCipher(_N, 2 * _numParty - 1, ptr[0], ptr[1]);
  MemoryManagement::memcpyMM_h2d(out->_data, ptr + 2,
                                 getSizeReducedCipher() - 2 * sizeof(double));
  return out;
}
void MpcApplication::exportReducedCipher(TrlweCipher *inp, void *out) {
  double *ptr = (double *)out;
  ptr[0] = inp->_sdError;
  ptr[1] = inp->_varError;
  MemoryManagement::memcpyMM_d2h(ptr + 2, inp->_data,
                                 getSizeReducedCipher() - 2 * sizeof(double));
}
int MpcApplication::getSizeReducedCipher() {
  return 2 * sizeof(double) + 2 * _numParty * _N * sizeof(TorusInteger);
}
TrlweCipher *MpcApplication::addOp(TrlweCipher *inp_1, TrlweCipher *inp_2) {
  if (!inp_1 || !inp_2) {
    WARNING_CERR("inp_1 and inp_2 is not NULL");
    return nullptr;
  }
  TrlweCipher *out =
      new TrlweCipher(_N, 2 * _numParty - 1, inp_1->_sdError + inp_2->_sdError,
                      inp_1->_varError + inp_2->_varError);
  MemoryManagement::memcpyMM_d2d(out->_data, inp_1->_data,
                                 getSizeReducedCipher() - 2 * sizeof(double));
  TorusUtility::addVector(out->_data, inp_2->_data, 2 * _numParty * _N);
  return out;
}
TrlweCipher *MpcApplication::subOp(TrlweCipher *inp_1, TrlweCipher *inp_2) {
  if (!inp_1 || !inp_2) {
    WARNING_CERR("inp_1 and inp_2 is not NULL");
    return nullptr;
  }
  TrlweCipher *out =
      new TrlweCipher(_N, 2 * _numParty - 1, inp_1->_sdError + inp_2->_sdError,
                      inp_1->_varError + inp_2->_varError);
  MemoryManagement::memcpyMM_d2d(out->_data, inp_1->_data,
                                 getSizeReducedCipher() - 2 * sizeof(double));
  TorusUtility::subVector(out->_data, inp_2->_data, 2 * _numParty * _N);
  return out;
}
TrlweCipher *MpcApplication::notOp(TrlweCipher *inp) {
  if (!inp) {
    WARNING_CERR("inp is not NULL");
    return nullptr;
  }
  TrlweCipher *out =
      new TrlweCipher(_N, 2 * _numParty - 1, inp->_sdError, inp->_varError);
  out->clear_trlwe_data();
  TrlweFunction::putPlain(out, 1);
  TorusUtility::subVector(out->_data, inp->_data, 2 * _numParty * _N);
  return out;
}
TrlweCipher *MpcApplication::notXorOp(TrlweCipher *inp_1, TrlweCipher *inp_2) {
  if (!inp_1 || !inp_2) {
    WARNING_CERR("inp_1 and inp_2 is not NULL");
    return nullptr;
  }
  TrlweCipher *out =
      new TrlweCipher(_N, 2 * _numParty - 1, inp_1->_sdError + inp_2->_sdError,
                      inp_1->_varError + inp_2->_varError);
  out->clear_trlwe_data();
  TrlweFunction::putPlain(out, 1);
  TorusUtility::subVector(out->_data, inp_1->_data, 2 * _numParty * _N);
  TorusUtility::subVector(out->_data, inp_2->_data, 2 * _numParty * _N);
  return out;
}
TrlweCipher *MpcApplication::mulOp(TrlweCipher *inp_1, TrgswCipher *inp_2) {
  if (!inp_1 || !inp_2) {
    WARNING_CERR("inp_1 and inp_2 is not NULL");
    return nullptr;
  }
  if (!_fft_mul)
    _fft_mul = new BatchedFFT(_N, 2 * _numParty, 2 * _l * _numParty);
  // Prepare FFT for multiplication
  for (int i = 0; i < 2 * _l * _numParty; i++) {
    for (int j = 0; j < 2 * _numParty; j++)
      _fft_mul->setInp(inp_2->get_pol_data(i, j), j, i);
  }
  // Init output
  double e_dec = std::pow(2, -_l - 1);
  //   sdError = 2 * l * numParty * N * sd2 + numParty * (N + 1) * e_dec
  //             + sd1
  //   varError = 2 * l * numParty * N * var2 + numParty * (N + 1) * (e_dec ^ 2)
  //              + var1
  TrlweCipher *out = new TrlweCipher(
      _N, 2 * _numParty - 1,
      2 * _l * _numParty * _N * inp_2->_sdError + _numParty * (_N + 1) * e_dec +
          inp_1->_sdError,
      2 * _l * _numParty * _N * inp_2->_varError +
          _numParty * (_N + 1) * e_dec * e_dec + inp_1->_varError);
  out->clear_trlwe_data();
  // Decomposition and multiplication
  TorusInteger *decomp_ptr = (TorusInteger *)MemoryManagement::mallocMM(
      2 * _l * _numParty * _N * sizeof(TorusInteger));
  Decomposition::onlyDecomp(inp_1, inp_2, decomp_ptr);
  for (int i = 0; i < 2 * _l * _numParty; i++)
    _fft_mul->setInp(decomp_ptr + i * _N, i);
  for (int i = 0; i < 2 * _numParty; i++) {
    for (int j = 0; j < 2 * _l * _numParty; j++)
      _fft_mul->setMul(i, j);
  }
  for (int i = 0; i < 2 * _numParty; i++)
    _fft_mul->addAllOut(out->get_pol_data(i), i);
  // Wait all
  _fft_mul->waitAllOut();
  // Free all
  MemoryManagement::freeMM(decomp_ptr);
  return out;
}
