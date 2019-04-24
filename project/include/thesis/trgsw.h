#ifndef THESIS_TRGSW_H
#define THESIS_TRGSW_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class TrgswParams {
public:
  int _N;
  int _k;
  int _l;
  int _Bgbit;
  double _sdError;
  double _varError;

  TrgswParams() = delete;
  TrgswParams(const TrgswParams &) = delete;
  TrgswParams(int N, int k, int l, int Bgbit, double sdError, double varError);

  TrgswParams &operator=(const TrgswParams &) = delete;

  ~TrgswParams();

  void gateBootstrap(size_t count, void *tlweCiphers, TorusInteger *constants,
                     void *BKey, void *KSKey);
};

/*
class Trgsw {
private:
  int _l;
  int _Bgbit;
  int _N;
  int _k;

  std::vector<PolynomialBinary> _s;
  std::vector<std::vector<PolynomialTorus>> _ciphertexts;
  std::vector<double> _stddevErrors;
  std::vector<double> _varianceErrors;
  std::vector<bool> _plaintexts;

public:
  // Constructors
  Trgsw();
  Trgsw(const Trgsw &obj) = delete;

  // Destructor
  ~Trgsw();

  // Copy assignment operator
  Trgsw &operator=(const Trgsw &obj) = delete;

  // Get params
  int get_l() const;
  int get_Bgbit() const;
  int get_N() const;
  int get_k() const;

  // Set attributes
  void clear_s();
  void clear_ciphertexts();
  void clear_plaintexts();
  bool set_s(const std::vector<PolynomialBinary> &s,
             bool isForcedToCheck = true);
  bool move_s(std::vector<PolynomialBinary> &s, bool isForcedToCheck = true);
  void generate_s();
  bool addCiphertext(const std::vector<PolynomialTorus> &cipher,
                     double stddevError, double varianceError,
                     bool isForcedToCheck = true);
  bool moveCiphertext(std::vector<PolynomialTorus> &cipher, double stddevError,
                      double varianceError, bool isForcedToCheck = true);
  void addPlaintext(bool plain);

  // Get attributes
  const std::vector<PolynomialBinary> &get_s() const;
  const std::vector<std::vector<PolynomialTorus>> &get_ciphertexts() const;
  const std::vector<double> &get_stddevErrors() const;
  const std::vector<double> &get_varianceErrors() const;
  const std::vector<bool> &get_plaintexts() const;

  // Utilities
  bool encryptAll(bool isForcedToCheck = true);
  bool decryptAll(bool isForcedToCheck = true);
  bool getAllErrorsForDebugging(std::vector<double> &errors,
                                const std::vector<bool> &expectedPlaintexts,
                                bool isForcedToCheck = true);
  bool decompositeAll(std::vector<std::vector<PolynomialInteger>> &out,
                      const Trlwe &inp, bool isForcedToCheck = true) const;
  bool decomposite(std::vector<std::vector<PolynomialInteger>> &out,
                   const Trlwe &inp, const std::vector<int> &trlweCipherIds,
                   bool isForcedToCheck = true) const;
  void setParamTo(Trlwe &obj) const;
  bool _externalProduct(Trlwe &out, const Trlwe &inp,
                        const std::vector<int> &trlweCipherIds,
                        const std::vector<int> &trgswCipherIds,
                        std::unique_ptr<BatchedFFT> &ptr,
                        bool isForcedToCheck = true) const;
  bool externalProduct(Trlwe &out, const Trlwe &inp,
                       const std::vector<int> &trlweCipherIds,
                       const std::vector<int> &trgswCipherIds,
                       bool isForcedToCheck = true) const;
  bool _internalProduct(int &cipherIdResult, int cipherIdA, int cipherIdB,
                        std::unique_ptr<BatchedFFT> &ptr,
                        bool isForcedToCheck = true);
  bool internalProduct(int &cipherIdResult, int cipherIdA, int cipherIdB,
                       bool isForcedToCheck = true);
  bool _cMux(Trlwe &out, const Trlwe &inp,
             const std::vector<int> &trlweCipherTrueIds,
             const std::vector<int> &trlweCipherFalseIds,
             const std::vector<int> &trgswCipherIds,
             std::unique_ptr<BatchedFFT> &ptr,
             bool isForcedToCheck = true) const;
  bool cMux(Trlwe &out, const Trlwe &inp,
            const std::vector<int> &trlweCipherTrueIds,
            const std::vector<int> &trlweCipherFalseIds,
            const std::vector<int> &trgswCipherIds,
            bool isForcedToCheck = true) const;
  bool blindRotate(Trlwe &out, const Trlwe &inp,
                   const std::vector<int> &trlweCipherIds,
                   const std::vector<int> &coefficients,
                   const std::vector<int> &trgswCipherIds) const;
  bool bootstrapTLWE(Tlwe &out, const std::vector<Torus> &constants,
                     const Tlwe &inp, int tlweCipherId,
                     const std::vector<int> &trgswCipherIds) const;
  bool gateBootstrap(Tlwe &out, const std::vector<Torus> &constants,
                     const Tlwe &inp, int tlweCipherId,
                     const std::vector<int> &trgswCipherIds, const Tlwe &ks,
                     int ks_t) const;
};
*/
} // namespace thesis

#endif
