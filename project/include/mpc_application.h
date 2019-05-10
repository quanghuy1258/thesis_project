#ifndef MPC_APPLICATION_H
#define MPC_APPLICATION_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

class MpcApplication {
private:
  /**
   * @attribute _numParty: number of parties
   * @attribute _partyId: id of current party
   * @attribute _N: R[X] / (X^N + 1)
   * @attribute _k: _k = 1 (fixed)
   * @attribute _m: number of samples (pubkey)
   * @attribute _l: decomp length
   * @attribute _sdFresh: sd of noise in fresh cipher
   */
  int _numParty;
  int _partyId;
  int _N;
  int _m;
  int _l;
  double _sdFresh;

  thesis::TorusInteger *_privkey;
  std::vector<thesis::TrlweCipher *> _pubkey;
  thesis::BatchedFFT _fft_privkey;
  thesis::BatchedFFT _fft_pubkey;
  thesis::BatchedFFT _fft_preExpand;
  thesis::BatchedFFT _fft_preExpandRandom;
  std::vector<void *> _stream;

  // Extend
  thesis::TorusInteger *_decompPreExpand(void *hPreExpand, int id,
                                         thesis::TrgswCipher *param);
  /**
   * @param hPreExpand: pointer to pre expand ciphertext in host memory (RAM),
   *                    reuse it if null
   * @param partyId: id of party
   * @param cipher: cipher associated with id of party in device memory (VRAM)
   */
  thesis::TrgswCipher *_extend(void *hPreExpand, int partyId,
                               thesis::TorusInteger *cipher);
  /**
   * @param hPreExpand: pointer to pre expand ciphertext in host memory (RAM),
   *                    reuse it if null
   * @param partyId: id of party
   * @param random: random associated with id of party in device memory (VRAM)
   */
  thesis::TrgswCipher *_extendWithPlainRandom(void *hPreExpand, int partyId,
                                              thesis::TorusInteger *random);

public:
  MpcApplication() = delete;
  MpcApplication(const MpcApplication &) = delete;
  MpcApplication(int numParty, int partyId, int N, int m, int l,
                 double sdFresh);

  MpcApplication &operator=(const MpcApplication &) = delete;

  ~MpcApplication();

  // Private key
  void createPrivkey();
  /**
   * @param hPrivkey: private key pointer in host memory (RAM)
   */
  void importPrivkey(void *hPrivkey);
  /**
   * @param hPrivkey: private key pointer in host memory (RAM)
   */
  void exportPrivkey(void *hPrivkey);
  int getSizePrivkey();

  // Public key
  /**
   * throw exception if private key is null
   */
  void createPubkey();
  /**
   * @param hPubkey: public key pointer in host memory (RAM)
   */
  void importPubkey(void *hPubkey);
  /**
   * @param hPubkey: public key pointer in host memory (RAM)
   */
  void exportPubkey(void *hPubkey);
  int getSizePubkey();

  // Encrypt
  /**
   * @param msg: one bit message
   * @param hCipher: ciphertext pointer in host memory (RAM)
   * @param hRandom: output random in hCipher if necessary
   */
  void encrypt(bool msg, void *hCipher, void *hRandom = nullptr);
  int getSizeCipher();
  int getSizeRandom();

  // Expand
  /**
   * @param hPubkey: public key of another party
   * @param hPreExpand: pre expand ciphertext pointer in host memory (RAM)
   */
  void preExpand(void *hPubkey, void *hPreExpand);
  int getSizePreExpand();
  /**
   * @param hPreExpand: pointer to pre expand ciphertext in host memory (RAM)
   * @param freeFnPreExpand: free hPreExpand after use because hPreExpand will
   *                         be cached
   * @param partyId: id of party
   * @param hCipher: cipher associated with id of party
   */
  std::vector<thesis::TrgswCipher *>
  expand(std::vector<void *> &hPreExpand,
         std::function<void(void *)> freeFnPreExpand, int partyId,
         void *hCipher);
  /**
   * @param hPreExpand: pointer to pre expand ciphertext in host memory (RAM)
   * @param freeFnPreExpand: free hPreExpand after use because hPreExpand will
   *                         be cached
   * @param partyId: id of party
   * @param hCipher: cipher associated with id of party
   * @param hRandom: plain random associated with cipher
   */
  std::vector<thesis::TrgswCipher *>
  expand(std::vector<void *> &hPreExpand,
         std::function<void(void *)> freeFnPreExpand, int partyId,
         void *hCipher, void *hRandom);
};

#endif
