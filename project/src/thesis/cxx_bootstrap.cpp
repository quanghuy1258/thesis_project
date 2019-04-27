#include "thesis/bootstrap.h"

namespace thesis {

Bootstrap::Bootstrap(int N, int k, int l, int Bgbit) {
  if (N < 2 || (N & (N - 1)) || k < 1 || l < 1 || Bgbit < 1)
    throw std::invalid_argument(
        "N = 2^a with a > 0 ; k > 0 ; l > 0 ; Bgbit > 0");
  _N = N;
  _k = k;
  _l = l;
  _Bgbit = Bgbit;
}
Bootstrap::~Bootstrap() {}
void Bootstrap::bootstrap() {}

} // namespace thesis
