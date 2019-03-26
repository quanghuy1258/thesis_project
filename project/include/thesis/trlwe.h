#ifndef THESIS_TRLWE_H
#define THESIS_TRLWE_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"
#include "thesis/tlwe.h"

namespace thesis {

class Trlwe {
private:
  int _N;
  int _k;

  std::vector<PolynomialBinary> _s;
  std::vector<std::vector<PolynomialTorus>> _ciphertexts;
  std::vector<double> _stddevErrors;
  std::vector<double> _varianceErrors;
  std::vector<PolynomialBinary> _plaintexts;

  friend class Trgsw;

public:
  // Constructors
  Trlwe();
  Trlwe(const Trlwe &obj) = delete;

  // Destructor
  ~Trlwe();

  // Copy assignment operator
  Trlwe &operator=(const Trlwe &obj) = delete;

  // Get params
  int get_N() const;
  int get_k() const;

  // Set attributes
  void clear_s();
  void clear_ciphertexts();
  void clear_plaintexts();
  bool set_s(const std::vector<PolynomialBinary> &s);
  void generate_s();
  bool addCiphertext(const std::vector<PolynomialTorus> &cipher,
                     double stddevError, double varianceError);
  bool addPlaintext(const PolynomialBinary &plain);

  // Get attributes
  const std::vector<PolynomialBinary> &get_s() const;
  const std::vector<std::vector<PolynomialTorus>> &get_ciphertexts() const;
  const std::vector<double> &get_stddevErrors() const;
  const std::vector<double> &get_varianceErrors() const;
  const std::vector<PolynomialBinary> &get_plaintexts() const;

  // Utilities
  bool encryptAll();
  bool decryptAll();
  bool getAllErrorsForDebugging(
      std::vector<double> &errors,
      const std::vector<PolynomialBinary> &expectedPlaintexts) const;
  void setParamTo(Tlwe &obj) const;
  void tlweExtractAll(Tlwe &out) const;
  bool tlweExtract(Tlwe &out, const std::vector<int> &ps,
                   const std::vector<int> &cipherIDs) const;
};

} // namespace thesis

#endif
