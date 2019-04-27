#include "thesis/decomposition.h"
#include "thesis/stream.h"

namespace thesis {

Decomposition::Decomposition(int N, int k, int l, int Bgbit, int num_stream) {
  if (N < 2 || (N & (N - 1)) || k < 1 || l < 1 || Bgbit < 1 || num_stream < 1)
    throw std::invalid_argument(
        "N = 2^a with a > 0 ; k > 0 ; l > 0 ; Bgbit > 0 ; num_stream > 0");
  _N = N;
  _k = k;
  _l = l;
  _Bgbit = Bgbit;
  _stream.resize(num_stream);
  for (int i = 0; i < num_stream; i++)
    _stream[i] = Stream::createS();
}
Decomposition::~Decomposition() {
  const int num_stream = _stream.size();
  for (int i = 0; i < num_stream; i++)
    Stream::destroyS(_stream[i]);
}
void Decomposition::product(TorusInteger *out, TorusInteger *inp,
                            int streamID) {
  const int num_stream = _stream.size();
  if (!out || !inp || streamID < 0 || streamID >= num_stream)
    return;
}
void Decomposition::bootstrap(TorusInteger *out, TorusInteger *inp,
                              int oefficient, int streamID) {
  const int num_stream = _stream.size();
  if (!out || !inp || streamID < 0 || streamID >= num_stream)
    return;
}

} // namespace thesis
