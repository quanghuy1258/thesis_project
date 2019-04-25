#ifndef DECOMPOSITION_H
#define DECOMPOSITION_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class Decomposition {
private:
  int _N;
  int _k;
  int _l;
  int _Bgbit;
  std::vector<void *> _stream;

public:
  Decomposition() = delete;
  Decomposition(const Decomposition &) = delete;
  Decomposition(int N, int k, int l, int Bgbit, int num_stream);

  Decomposition &operator=(const Decomposition &) = delete;

  ~Decomposition();

  void product(TorusInteger *out, TorusInteger *inp,
               int streamID); // Only decomposition
  void bootstrap(TorusInteger *out, TorusInteger *inp, int oefficient,
                 int streamID); // Rotate --> True - False --> Decomposition
};

} // namespace thesis

#endif
