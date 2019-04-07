#ifndef THESIS_TLWE_H
#define THESIS_TLWE_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class Tlwe {
private:
  int _n;

  std::vector<bool> _s;
  std::vector<std::vector<Torus>> _ciphertexts;
  std::vector<double> _stddevErrors;
  std::vector<double> _varianceErrors;
  std::vector<bool> _plaintexts;

  friend class Trlwe;
  friend class Trgsw;

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
  bool set_n(int n, bool isForcedToCheck = true);

  // Set attributes
  void clear_s();
  void clear_ciphertexts();
  void clear_plaintexts();
  bool set_s(const std::vector<bool> &s, bool isForcedToCheck = true);
  bool moveTo_s(std::vector<bool> &s, bool isForcedToCheck = true);
  bool generate_s(bool isForcedToCheck = true);
  bool addCiphertext(const std::vector<Torus> &cipher, double stddevError,
                     double varianceError, bool isForcedToCheck = true);
  bool moveCiphertext(std::vector<Torus> &cipher, double stddevError,
                      double varianceError, bool isForcedToCheck = true);
  void addPlaintext(bool bit);

  // Get attributes
  const std::vector<bool> &get_s() const;
  const std::vector<std::vector<Torus>> &get_ciphertexts() const;
  const std::vector<double> &get_stddevErrors() const;
  const std::vector<double> &get_varianceErrors() const;
  const std::vector<bool> &get_plaintexts() const;

  // Utilities
  bool encryptAll(bool isForcedToCheck = true);
  bool decryptAll(bool isForcedToCheck = true);
  bool getAllErrorsForDebugging(std::vector<double> &errors,
                                const std::vector<bool> &expectedPlaintexts,
                                bool isForcedToCheck = true) const;
  bool initPublicKeySwitching(const std::vector<bool> &key, int t,
                              bool isForcedToCheck = true);
};

}; // namespace thesis

#endif
