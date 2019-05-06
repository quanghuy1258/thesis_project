#include "thesis/batched_fft.h"
#include "thesis/memory_management.h"
#include "thesis/trlwe_cipher.h"
#include "thesis/trlwe_function.h"

#include "mpc_application.h"

MpcApplication::MpcApplication(int numParty, int partyId, int N, int m, int l,
                               double sdFresh)
    : _fft_pubkey(N, 2, 1) {
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
  _privkey = (thesis::TorusInteger *)thesis::MemoryManagement::mallocMM(
      N * sizeof(thesis::TorusInteger));
  _pubkey.resize(m, nullptr);
  for (int i = 0; i < _m; i++)
    _pubkey[i] = new thesis::TrlweCipher(N, 1, sdFresh, sdFresh * sdFresh);
}
MpcApplication::~MpcApplication() {
  thesis::MemoryManagement::freeMM(_privkey);
  for (int i = 0; i < _m; i++) {
    if (!_pubkey[i])
      continue;
    delete _pubkey[i];
  }
}
void MpcApplication::createPrivkey() {
  thesis::TrlweFunction::genkey(_privkey, _N, 1);
  thesis::TrlweFunction::keyToFFT(_privkey, _N, 1, &_fft_pubkey);
}
void MpcApplication::importPrivkey(void *hPrivkey) {
  if (!hPrivkey)
    return;
  thesis::MemoryManagement::memcpyMM_h2d(_privkey, hPrivkey, getSizePrivkey());
  thesis::TrlweFunction::keyToFFT(_privkey, _N, 1, &_fft_pubkey);
}
void MpcApplication::exportPrivkey(void *hPrivkey) {
  if (!hPrivkey)
    return;
  thesis::MemoryManagement::memcpyMM_d2h(hPrivkey, _privkey, getSizePrivkey());
}
size_t MpcApplication::getSizePrivkey() {
  return _N * sizeof(thesis::TorusInteger);
}
void MpcApplication::createPubkey() {
  for (int i = 0; i < _m; i++) {
    if (!_pubkey[i])
      continue;
    thesis::TrlweFunction::createSample(&_fft_pubkey, i & 1, _pubkey[i]);
  }
}
void MpcApplication::importPubkey(void *hPubkey) {
  if (!hPubkey)
    return;
  thesis::TorusInteger *ptr = (thesis::TorusInteger *)hPubkey;
  for (int i = 0; i < _m; i++) {
    if (!_pubkey[i])
      continue;
    thesis::MemoryManagement::memcpyMM_h2d(_pubkey[i]->_data, ptr + _N * 2 * i,
                                           _N * 2 *
                                               sizeof(thesis::TorusInteger));
  }
}
void MpcApplication::exportPubkey(void *hPubkey) {
  if (!hPubkey)
    return;
  thesis::TorusInteger *ptr = (thesis::TorusInteger *)hPubkey;
  for (int i = 0; i < _m; i++) {
    if (!_pubkey[i])
      continue;
    thesis::MemoryManagement::memcpyMM_d2h(ptr + _N * 2 * i, _pubkey[i]->_data,
                                           _N * 2 *
                                               sizeof(thesis::TorusInteger));
  }
}
size_t MpcApplication::getSizePubkey() {
  return _N * 2 * sizeof(thesis::TorusInteger) * _m;
}
