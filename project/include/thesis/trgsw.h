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
  double _alpha;

  std::vector<PolynomialBinary> _s;
  std::vector<std::vector<PolynomialTorus>> _ciphertexts;
  std::vector<PolynomialInteger> _plaintexts;

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
  int get_alpha() const;

  // Set attributes
  void clear_s();
  void clear_ciphertexts();
  void clear_plaintexts();
  bool set_s(const std::vector<PolynomialBinary> &s);
  void generate_s();
  bool addCiphertext(const std::vector<PolynomialTorus> &cipher);
  bool addPlaintext(const PolynomialInteger &plain);

  // Get attributes
  bool get_s(std::vector<PolynomialBinary> &s) const;
  void
  get_ciphertexts(std::vector<std::vector<PolynomialTorus>> &ciphertexts) const;
  void get_plaintexts(std::vector<PolynomialInteger> &plaintexts) const;

  // Utilities
  bool encryptAll();
  bool decryptAll();
};

} // namespace thesis

#endif
