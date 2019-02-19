#ifndef THESIS_TLWE_H
#define THESIS_TLWE_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class Tlwe {
private:
  static INTEGER _n;
  static REAL _sdError;
  static REAL _sdMaximumError;

  std::vector<INTEGER> _s;
  std::vector<std::vector<INTEGER>> _ciphertexts;
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
  static INTEGER get_n();
  static REAL get_sdError();
  static REAL get_sdMaximumError();

  // Set attributes
  void clear_s();
  void clear_ciphertexts();
  void clear_plaintexts();
  bool set_s(const std::vector<INTEGER> &s);
  bool addCiphertext(const std::vector<INTEGER> &cipher);
  void addPlaintext(const bool &bit);

  // Get attributes
  bool get_s(std::vector<INTEGER> &s) const;
  void get_ciphertexts(std::vector<std::vector<INTEGER>> &ciphertexts) const;
  void get_plaintexts(std::vector<bool> &plaintexts) const;

  // Utilities
  void encryptAll();
  void decryptAll();
};

}; // namespace thesis

#endif
