#ifndef MPC_APPLICATION_H
#define MPC_APPLICATION_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

class MpcApplication {
private:
  int _numParty; // number of parties
  int _partyId;  // id of current party
  int _N;
  // int _k; --> k = 1 (fixed)
  int _m; // pubkey: number of samples
  int _l;
  double _sdFresh;

  thesis::TorusInteger *_privkey;
  std::vector<thesis::TrlweCipher *> _pubkey;
  thesis::BatchedFFT _fft_with_privkey;
  thesis::BatchedFFT _fft_with_pubkey;
  thesis::BatchedFFT _fft_with_preExpand;
  std::vector<void *> _stream;

public:
  MpcApplication() = delete;
  MpcApplication(const MpcApplication &) = delete;
  MpcApplication(int numParty, int partyId, int N, int m, int l,
                 double sdFresh);

  MpcApplication &operator=(const MpcApplication &) = delete;

  ~MpcApplication();

  // Private key
  void createPrivkey();
  void importPrivkey(
      void *hPrivkey); // hPrivkey: private key pointer in host memory (RAM)
  void exportPrivkey(
      void *hPrivkey); // hPrivkey: private key pointer in host memory (RAM)
  int getSizePrivkey();

  // Public key
  void createPubkey(); // throw exception if private key is null
  void importPubkey(
      void *hPubkey); // hPubkey: public key pointer in host memory (RAM)
  void exportPubkey(
      void *hPubkey); // hPubkey: public key pointer in host memory (RAM)
  int getSizePubkey();

  // Encrypt
  void
  encrypt(bool msg,
          void *hCipher); // hCipher: ciphertext pointer in host memory (RAM)
  int getSizeCipher();

  // Expand
  void preExpand(void *hPubkey,     // hPubkey: public key of another party
                 void *hPreExpand); // hPreExpand: pre expand ciphertext pointer
                                    // in host memory (RAM)
  int getSizePreExpand();
};

#endif
