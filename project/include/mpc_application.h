#ifndef MPC_APPLICATION_H
#define MPC_APPLICATION_H

#include "thesis/batched_fft.h"
#include "thesis/declarations.h"
#include "thesis/load_lib.h"
#include "thesis/trgsw_cipher.h"
#include "thesis/trlwe_cipher.h"

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
  thesis::BatchedFFT *_fft_privkey;
  thesis::BatchedFFT *_fft_pubkey;
  thesis::BatchedFFT *_fft_preExpand;
  thesis::BatchedFFT *_fft_preExpandRandom;
  thesis::BatchedFFT *_fft_mul;
  std::vector<void *> _stream;

  // Extend
  thesis::TorusInteger *_decompPreExpand(void *hPreExpand, int id);
  /**
   * @param hPreExpand: pointer to pre expand ciphertext in host memory (RAM),
   *                    reuse it if null
   * @param partyId: id of party
   * @param cipher: cipher associated with id of party in device memory (VRAM)
   * @param mainPartyId: id of party associated with main cipher
   * @param out: expanded cipher
   */
  void _extend(void *hPreExpand, int partyId, thesis::TorusInteger *cipher,
               int mainPartyId, thesis::TrgswCipher *out);
  /**
   * @param hPreExpand: pointer to pre expand ciphertext in host memory (RAM),
   *                    reuse it if null
   * @param partyId: id of party
   * @param random: random associated with id of party in device memory (VRAM)
   * @param mainPartyId: id of party associated with main cipher
   * @param out: expanded cipher
   */
  void _extendWithPlainRandom(void *hPreExpand, int partyId,
                              thesis::TorusInteger *random, int mainPartyId,
                              thesis::TrgswCipher *out);

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
   * @param hMainCipher: main ciphertext in RAM (Not NULL)
   * @param hRandCipher: random ciphertext in RAM (NULL = not get)
   * @param hRandom: random plaintext in RAM (NULL = not get)
   */
  void encrypt(bool msg, void *hMainCipher, void *hRandCipher, void *hRandom);
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
   * @param hMainCipher: main cipher associated with id of party
   * @param hRandCipher: rand cipher associated with id of party
   */
  thesis::TrgswCipher *expand(std::vector<void *> &hPreExpand,
                              std::function<void(void *)> freeFnPreExpand,
                              int partyId, void *hMainCipher,
                              void *hRandCipher);
  /**
   * @param hPreExpand: pointer to pre expand ciphertext in host memory (RAM)
   * @param freeFnPreExpand: free hPreExpand after use because hPreExpand will
   *                         be cached
   * @param partyId: id of party
   * @param hMainCipher: main cipher associated with id of party
   * @param hRandom: plain random associated with cipher
   */
  thesis::TrgswCipher *
  expandWithPlainRandom(std::vector<void *> &hPreExpand,
                        std::function<void(void *)> freeFnPreExpand,
                        int partyId, void *hMainCipher, void *hRandom);

  // Decrypt
  /**
   * @param cipher: expanded cipher
   */
  thesis::TorusInteger partDec(thesis::TrgswCipher *cipher);
  /**
   * @param partDecPlain: array of outputs of partDec
   * @param outError: pointer absolute value of error (null if not want to get
   *                  error)
   */
  bool finDec(thesis::TorusInteger partDecPlain[], double *outError);

  // Evaluation
  thesis::TrgswCipher *importExpandedCipher(void *inp);
  void exportExpandedCipher(thesis::TrgswCipher *inp, void *out);
  int getSizeExpandedCipher();
  thesis::TrgswCipher *addOp(thesis::TrgswCipher *inp_1,
                             thesis::TrgswCipher *inp_2);
  thesis::TrgswCipher *subOp(thesis::TrgswCipher *inp_1,
                             thesis::TrgswCipher *inp_2);
  thesis::TrgswCipher *notOp(thesis::TrgswCipher *inp);
  thesis::TrgswCipher *notXorOp(thesis::TrgswCipher *inp_1,
                                thesis::TrgswCipher *inp_2);
  thesis::TrgswCipher *mulOp(thesis::TrgswCipher *inp_1,
                             thesis::TrgswCipher *inp_2);

  // Reduction
  //   Reduce
  thesis::TrlweCipher *reduce(thesis::TrgswCipher *inp);
  //  Decrypt
  thesis::TorusInteger partDec(thesis::TrlweCipher *cipher);
  //  Import & Export
  thesis::TrlweCipher *importReducedCipher(void *inp);
  void exportReducedCipher(thesis::TrlweCipher *inp, void *out);
  int getSizeReducedCipher();
  //  Evaluate
  thesis::TrlweCipher *addOp(thesis::TrlweCipher *inp_1,
                             thesis::TrlweCipher *inp_2);
  thesis::TrlweCipher *subOp(thesis::TrlweCipher *inp_1,
                             thesis::TrlweCipher *inp_2);
  thesis::TrlweCipher *notOp(thesis::TrlweCipher *inp);
  thesis::TrlweCipher *notXorOp(thesis::TrlweCipher *inp_1,
                                thesis::TrlweCipher *inp_2);
  thesis::TrlweCipher *mulOp(thesis::TrlweCipher *inp_1,
                             thesis::TrgswCipher *inp_2);

  // Packing
  thesis::TrlweCipher *pseudoCipher(bool msgScalar);
  thesis::TrlweCipher *pseudoCipher(bool msgPol[]);
  //   output = (C) ? d_1 : d_0
  thesis::TrlweCipher *cMux(thesis::TrgswCipher *C, thesis::TrlweCipher *d_1,
                            thesis::TrlweCipher *d_0,
                            bool update_fft_mul = true);
  //   output = input * X^deg
  thesis::TrlweCipher *rotate(thesis::TrlweCipher *inp, int deg);
  //   output = (cond) ? (input * X^deg) : input
  thesis::TrlweCipher *blindRotate(thesis::TrlweCipher *inp,
                                   thesis::TrgswCipher *cond, int deg,
                                   bool update_fft_mul = true);
};

#endif
