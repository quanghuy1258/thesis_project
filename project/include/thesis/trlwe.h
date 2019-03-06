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
  double _alpha;

  std::vector<PolynomialBinary> _s;
  std::vector<std::vector<PolynomialTorus>> _ciphertexts;
  std::vector<PolynomialBinary> _plaintexts;

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
  double get_alpha() const;

  // Set attributes
  void clear_s();
  void clear_ciphertexts();
  void clear_plaintexts();
  bool set_s(const std::vector<PolynomialBinary> &s);
  void generate_s();
  bool addCiphertext(const std::vector<PolynomialTorus> &cipher);
  bool addPlaintext(const PolynomialBinary &plain);

  // Get attributes
  bool get_s(std::vector<PolynomialBinary> &s) const;
  void
  get_ciphertexts(std::vector<std::vector<PolynomialTorus>> &ciphertexts) const;
  void get_plaintexts(std::vector<PolynomialBinary> &plaintexts) const;

  // Utilities
  bool encryptAll();
  bool decryptAll();
  bool tlweExtractAll(Tlwe &out) const;
  bool tlweExtractOne(Tlwe &out, int p, int cipherID);
};

} // namespace thesis

#endif
