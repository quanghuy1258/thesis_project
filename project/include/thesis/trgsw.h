#ifndef THESIS_TRGSW_H
#define THESIS_TRGSW_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class Trgsw {
private:
  int _l;
  int _Bgbit;
  int _N;
  int _k;

  std::vector<PolynomialBinary> _s;
  std::vector<std::vector<PolynomialTorus>> _ciphertexts;
  std::vector<double> _stddevErrors;
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
  bool set_s(const std::vector<PolynomialBinary> &s);
  void generate_s();
  bool addCiphertext(const std::vector<PolynomialTorus> &cipher,
                     double stddevError);
  void addPlaintext(bool plain);

  // Get attributes
  const std::vector<PolynomialBinary> &get_s() const;
  const std::vector<std::vector<PolynomialTorus>> &get_ciphertexts() const;
  const std::vector<double> &get_stddevErrors() const;
  const std::vector<bool> &get_plaintexts() const;

  // Utilities
  bool encryptAll();
  bool decryptAll();
  bool
  getAllErrorsForDebugging(std::vector<double> &errors,
                           const std::vector<bool> &expectedPlaintexts) const;
  bool decompositeAll(std::vector<std::vector<PolynomialInteger>> &out,
                      const Trlwe &inp) const;
  void setParamTo(Trlwe &obj) const;
  bool externalProduct(Trlwe &out, const Trlwe &inp,
                       const std::vector<int> &trlweCipherIds,
                       const std::vector<int> &trgswCipherIds) const;
  bool internalProduct(int &cipherIdResult, int cipherIdA, int cipherIdB);
  bool cMux(Trlwe &out, const Trlwe &inp,
            const std::vector<int> &trlweCipherTrueIds,
            const std::vector<int> &trlweCipherFalseIds,
            const std::vector<int> &trgswCipherIds) const;
};

} // namespace thesis

#endif
