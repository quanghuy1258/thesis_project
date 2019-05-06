#ifndef MPC_APPLICATION_H
#define MPC_APPLICATION_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

class MpcApplication {
public:
  int _numParty; // number of parties
  int _partyId;  // id of current party
  int _N;
  // int _k; --> k = 1 (fixed)
  int _m; // pubkey: number of samples

  thesis::TorusInteger *_privkey;
  std::vector<thesis::TrgswCipher *> _pubkey;

  MpcApplication() = delete;
  MpcApplication(const MpcApplication &) = delete;
  MpcApplication(int numParty, int partyId, int N, int m);

  MpcApplication &operator=(const MpcApplication &) = delete;

  ~MpcApplication();

  void createPrivkey();
  void importPrivkey(
      void *hPrivkey); // hPrivkey: private key pointer in host memory (RAM)
  void createPubkey(); // throw exception if private key is null
  void importPubkey(
      void *hPubkey); // hPubkey: public key pointer in host memory (RAM)
};

#endif
