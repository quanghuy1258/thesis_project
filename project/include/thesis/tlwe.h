#ifndef THESIS_TLWE_H
#define THESIS_TLWE_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class Tlwe {
private:
  int _n;

  std::vector<Integer> _s;
  std::vector<std::vector<Torus>> _ciphertexts;
  std::vector<double> _stddevErrors;
  std::vector<bool> _plaintexts;

public:
  // Constructors
  Tlwe();
  Tlwe(const Tlwe &obj) = delete;

  // Destructor
  ~Tlwe();

  // Copy assignment operator
  Tlwe &operator=(const Tlwe &obj) = delete;

  // Get params
  int get_n() const;

  // Set params
  bool set_n(int n, bool isForcedClear = false);

  // Set attributes
  void clear_s();
  void clear_ciphertexts();
  void clear_plaintexts();
  bool set_s(const std::vector<Integer> &s);
  void generate_s();
  bool addCiphertext(const std::vector<Torus> &cipher, double stddevError);
  void addPlaintext(bool bit);

  // Get attributes
  const std::vector<Integer> &get_s() const;
  const std::vector<std::vector<Torus>> &get_ciphertexts() const;
  const std::vector<double> &get_stddevErrors() const;
  const std::vector<bool> &get_plaintexts() const;

  // Utilities
  bool encryptAll();
  bool decryptAll();
};

}; // namespace thesis

#endif
