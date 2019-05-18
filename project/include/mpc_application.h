#ifndef MPC_APPLICATION_H
#define MPC_APPLICATION_H

#include "thesis/batched_fft.h"
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
  thesis::TorusInteger *_decompPreExpand(void *hPreExpand, int id);
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
  [[deprecated]]
  void encrypt(bool msg, void *hCipher, void *hRandom = nullptr);
  /**
   * @param msg: one bit message
   * @param hMainCipher: main ciphertext in RAM (Not NULL)
   * @param hRandCipher: random ciphertext in RAM (NULL = not get)
   * @param hRandom: random plaintext in RAM (NULL = not get)
   */
  void encrypt(bool msg, void *hMainCipher, void *hRandCipher, void *hRandom);
  [[deprecated]]
  int getSizeCipher();
  int getSizeMainCipher();
  int getSizeRandCipher();
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

  // Decrypt
  /**
   * @param cipher: expanded cipher
   */
  thesis::TorusInteger partDec(std::vector<thesis::TrgswCipher *> &cipher);
  /**
   * @param partDecPlain: array of outputs of partDec
   * @param numParty: number of parties
   * @param outError: pointer absolute value of error (null if not want to get
   *                  error)
   */
  bool finDec(thesis::TorusInteger partDecPlain[], size_t numParty,
              double *outError);
};

#endif
