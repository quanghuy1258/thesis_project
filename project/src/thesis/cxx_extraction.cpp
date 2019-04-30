#include "thesis/extraction.h"

namespace thesis {

Extraction::Extraction(int N, int k) {
  if (N < 2 || (N & (N - 1)) || k < 1)
    throw std::invalid_argument("N = 2^a with a > 0 ; k > 0");
  _N = N;
  _k = k;
}
Extraction::~Extraction() {}

} // namespace thesis
