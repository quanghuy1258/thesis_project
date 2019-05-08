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
  thesis::BatchedFFT _fft_privkey;
  thesis::BatchedFFT _fft_pubkey;
  thesis::BatchedFFT _fft_preExpand;
  thesis::BatchedFFT _fft_preExpandRandom;
  std::vector<void *> _stream;

  thesis::TorusInteger *_decompPreExpand(void *hPreExpand, int id,
                                         thesis::TrgswCipher *param);

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
          void *hCipher, // hCipher: ciphertext pointer in host memory (RAM)
          void *hRandom = nullptr); // hRandom: output random in hCipher
  int getSizeCipher();

  // Expand
  void preExpand(void *hPubkey,     // hPubkey: public key of another party
                 void *hPreExpand); // hPreExpand: pre expand ciphertext pointer
                                    // in host memory (RAM)
  int getSizePreExpand();
  void extend(void *hPreExpand, // hPreExpand: pointer to pre expand ciphertext
                                // in host memory (RAM), reuse it if null
              int partyId,      // partyId: id of party
              void *hCipher,    // hCipher: cipher associated with id of party
              thesis::TrgswCipher *out);
  void extendWithPlainRandom(
      void *hPreExpand, // hPreExpand: pointer to pre expand ciphertext
      int partyId,      // partyId: id of party
      void *hRandom,    // hRandom: cipher associated with id of party
      thesis::TrgswCipher *out);
};

#endif
