#ifndef CIPHER_H
#define CIPHER_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class Cipher {
public:
  TorusInteger *_data;
  int _size;
  bool _isOwnData;
  double _sdError;
  double _varError;

  Cipher() = delete;
  Cipher(const Cipher &) = delete;
  Cipher(int size, double sdError, double varError);
  Cipher(TorusInteger *data, int size, double sdError, double varError);

  Cipher &operator=(const Cipher &) = delete;

  Cipher(Cipher &&obj);
  virtual Cipher &operator=(Cipher &&obj);

  virtual ~Cipher();
};

} // namespace thesis

#endif
