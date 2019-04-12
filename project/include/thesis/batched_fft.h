#ifndef THESIS_BATCHED_FFT_H
#define THESIS_BATCHED_FFT_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class BatchedFFT {
protected:
  int _N;
  int _batch;
  int _cache;
  std::vector<int> _multiplication_pair;
  std::vector<double> _inp;
  std::vector<std::complex<double>> _fft_inp;
  std::vector<double> _out;

  // Constructors
  BatchedFFT(int N, int batch, int cache);

  static BatchedFFT *newCustomInstance(int N, int batch, int cache);

public:
  // Constructors
  BatchedFFT() = delete;
  BatchedFFT(const BatchedFFT &) = delete;

  static std::unique_ptr<BatchedFFT>
  createInstance(int N, int batch = 1, int cache = 0,
                 bool isForcedToCheck = true);

  // Destructor
  virtual ~BatchedFFT();

  // Copy assignment operator
  virtual BatchedFFT &operator=(const BatchedFFT &obj) = delete;

  // Get params
  int get_N() const;
  int get_batch() const;
  int get_cache() const;

  // Utilities
  bool setTorusInput(const PolynomialTorus &inp, int pos,
                     void *eigenBarrierNotifier = nullptr,
                     bool isForcedToCheck = true);
  bool setIntegerInput(const PolynomialInteger &inp, int pos,
                       void *eigenBarrierNotifier = nullptr,
                       bool isForcedToCheck = true);
  bool setBinaryInput(const PolynomialBinary &inp, int pos,
                      void *eigenBarrierNotifier = nullptr,
                      bool isForcedToCheck = true);
  bool copyTo(int from, int to, void *eigenBarrierNotifier = nullptr,
              bool isForcedToCheck = true);
  bool setMultiplicationPair(int left, int right, int result,
                             bool isForcedToCheck = true);
  bool getOutput(PolynomialTorus &out, int pos,
                 void *eigenBarrierNotifier = nullptr,
                 bool isForcedToCheck = true) const;
  bool addOutput(PolynomialTorus &out, int pos,
                 void *eigenBarrierNotifier = nullptr,
                 bool isForcedToCheck = true) const;
  bool subOutput(PolynomialTorus &out, int pos,
                 void *eigenBarrierNotifier = nullptr,
                 bool isForcedToCheck = true) const;

  virtual bool doFFT() = 0;
  virtual bool doMultiplicationAndIFFT() = 0;
};

} // namespace thesis

#endif
