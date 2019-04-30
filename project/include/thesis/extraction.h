#ifndef EXTRACTION_H
#define EXTRACTION_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class Extraction {
private:
  int _N;
  int _k;

public:
  Extraction() = delete;
  Extraction(const Extraction &) = delete;
  Extraction(int N, int k);

  Extraction &operator=(const Extraction &) = delete;

  ~Extraction();
};

} // namespace thesis

#endif
