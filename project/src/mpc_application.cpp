#include "thesis/batched_fft.h"
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
    : _fft_with_privkey(N, 2, 1), _fft_with_pubkey(N, 2, m),
      _fft_with_preExpand(N, m * numParty, l) {
  if (numParty < 1 || partyId < 0 || partyId >= numParty || N < 2 ||
      (N & (N - 1)) || m < 1 || l < 1 || sdFresh <= 0)
    throw std::invalid_argument("numParty > 0 ; 0 <= partyId < numParty ; N = "
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
  _stream.resize(1 + 2 * _l * _m,
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
  TrlweFunction::keyToFFT(_privkey, _N, 1, &_fft_with_privkey);
}
void MpcApplication::importPrivkey(void *hPrivkey) {
  if (!hPrivkey)
    return;
  MemoryManagement::memcpyMM_h2d(_privkey, hPrivkey, getSizePrivkey());
  TrlweFunction::keyToFFT(_privkey, _N, 1, &_fft_with_privkey);
}
void MpcApplication::exportPrivkey(void *hPrivkey) {
  if (!hPrivkey)
    return;
  MemoryManagement::memcpyMM_d2h(hPrivkey, _privkey, getSizePrivkey());
}
size_t MpcApplication::getSizePrivkey() { return _N * sizeof(TorusInteger); }
void MpcApplication::createPubkey() {
  for (int i = 0; i < _m; i++) {
    TrlweFunction::createSample(&_fft_with_privkey, i & 1, _pubkey[i]);
    _fft_with_privkey.waitOut(i & 1);
    _fft_with_pubkey.setInp(_pubkey[i]->get_pol_data(0), 0, i);
    _fft_with_pubkey.setInp(_pubkey[i]->get_pol_data(1), 1, i);
  }
}
void MpcApplication::importPubkey(void *hPubkey) {
  if (!hPubkey)
    return;
  MemoryManagement::memcpyMM_h2d(_pubkey[0]->_data, hPubkey, getSizePubkey());
  for (int i = 0; i < _m; i++) {
    _fft_with_pubkey.setInp(_pubkey[i]->get_pol_data(0), 0, i);
    _fft_with_pubkey.setInp(_pubkey[i]->get_pol_data(1), 1, i);
  }
}
void MpcApplication::exportPubkey(void *hPubkey) {
  if (!hPubkey)
    return;
  MemoryManagement::memcpyMM_d2h(hPubkey, _pubkey[0]->_data, getSizePubkey());
}
size_t MpcApplication::getSizePubkey() {
  return _N * 2 * sizeof(TorusInteger) * _m;
}
void MpcApplication::encrypt(bool msg, void *hCipher) {
  TorusInteger *ptr = (TorusInteger *)hCipher;
  // Construct cipher
  std::vector<TrgswCipher *> cipher(2 * _l * _m + 1);
  for (int i = 0; i <= 2 * _l * _m; i++)
    cipher[i] = new TrgswCipher(_N, 1, _l, 1, _sdFresh, _sdFresh * _sdFresh);
  // Create random
  TorusInteger *random_ptr = (TorusInteger *)MemoryManagement::mallocMM(
      2 * _l * _m * _N * sizeof(TorusInteger));
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
      _fft_with_pubkey.setInp(random_ptr + (i * _m + j) * _N, j);
    for (int j = 0; j < _m; j++) {
      _fft_with_pubkey.setMul(0, j);
      _fft_with_pubkey.setMul(1, j);
    }
    _fft_with_pubkey.addAllOut(cipher[2 * _l * _m]->get_pol_data(i, 0), 0);
    _fft_with_pubkey.addAllOut(cipher[2 * _l * _m]->get_pol_data(i, 1), 1);
  }
  _fft_with_pubkey.waitAllOut();
  // main cipher += msg*G;
  if (msg)
    TrgswFunction::addMuGadget(1, cipher[2 * _l * _m], _stream[2 * _l * _m]);
  // Create cipher for random
  for (int i = 0; i < 2 * _l * _m; i++) {
    for (int j = 0; j < _l; j++)
      TrlweFunction::createSample(&_fft_with_privkey, (_l * i + j) & 1,
                                  cipher[i]->get_trlwe_data(_l + j), _N, 1,
                                  _sdFresh);
  }
  _fft_with_privkey.waitAllOut();
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
  // Wait all streams
  for (int i = 0; i <= 2 * _l * _m; i++)
    Stream::synchronizeS(_stream[i]);
  // Destroy random
  MemoryManagement::freeMM(random_ptr);
  // Destruct cipher
  for (int i = 0; i <= 2 * _l * _m; i++)
    delete cipher[i];
}
size_t MpcApplication::getSizeCipher() {
  return (2 * _l * _m * 2 + 4) * _l * _N * sizeof(TorusInteger);
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
    _fft_with_pubkey.setInp(pubkey_ptr + 2 * _N * i, i & 1, 0);
    _fft_with_pubkey.setMul(i & 1, 0);
    TorusUtility::subVector(preExpand_ptr + _N * i,
                            pubkey_ptr + (2 * i + 1) * _N, _N);
    _fft_with_pubkey.addAllOut(preExpand_ptr + _N * i, i & 1);
  }
  _fft_with_pubkey.waitAllOut();
  // Copy preExpand_ptr from device to host
  MemoryManagement::memcpyMM_d2h(hPreExpand, preExpand_ptr, getSizePreExpand());
  // Delete pubkey and preExpand
  MemoryManagement::freeMM(pubkey_ptr);
  MemoryManagement::freeMM(preExpand_ptr);
}
size_t MpcApplication::getSizePreExpand() {
  return _m * _N * sizeof(TorusInteger);
}
