#include "mpc_application.h"
#include "thesis/memory_management.h"
#include "thesis/trgsw_cipher.h"

MpcApplication::MpcApplication(int numParty, int partyId, int N, int m) {
  if (numParty < 1 || partyId < 0 || partyId >= numParty || N < 2 ||
      (N & (N - 1)) || m < 1)
    throw std::invalid_argument(
        "numParty > 0 ; 0 <= partyId < numParty ; N = 2^a with a > 0 ; m > 0");
  _numParty = numParty;
  _partyId = partyId;
  _N = N;
  _m = m;
  _privkey = (thesis::TorusInteger *)thesis::MemoryManagement::mallocMM(
      N * sizeof(thesis::TorusInteger));
  _pubkey.resize(m, nullptr);
}

MpcApplication::~MpcApplication() {
  thesis::MemoryManagement::freeMM(_privkey);
  for (int i = 0; i < _m; i++) {
    if (_pubkey[i])
      delete _pubkey[i];
  }
}
