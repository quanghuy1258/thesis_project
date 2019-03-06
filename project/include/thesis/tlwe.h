#ifndef THESIS_TLWE_H
#define THESIS_TLWE_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class Tlwe {
private:
  int _n;
  double _stddevError;

  std::vector<Integer> _s;
  std::vector<std::vector<Torus>> _ciphertexts;
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
  double get_stddevError() const;

  // Set params
  bool set_n(int n, bool isForcedClear = false);

  // Set attributes
  void clear_s();
  void clear_ciphertexts();
  void clear_plaintexts();
  bool set_s(const std::vector<Integer> &s);
  void generate_s();
  bool addCiphertext(const std::vector<Torus> &cipher);
  void addPlaintext(const bool &bit);

  // Get attributes
  bool get_s(std::vector<Integer> &s) const;
  void get_ciphertexts(std::vector<std::vector<Torus>> &ciphertexts) const;
  void get_plaintexts(std::vector<bool> &plaintexts) const;

  // Utilities
  bool encryptAll();
  bool decryptAll();
};

}; // namespace thesis

#endif
