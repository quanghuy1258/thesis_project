#ifndef THESIS_BATCHED_FFT_H
#define THESIS_BATCHED_FFT_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class BatchedFFT {
protected:
  int _N;
  int _batch_inp;
  int _batch_out;

  // Constructors
  BatchedFFT(int N, int batch_inp, int batch_out);

  static BatchedFFT *_createInstance(int N, int batch_inp, int batch_out);

  virtual void _setTorusInp(const PolynomialTorus &inp, int pos) = 0;
  virtual void _setIntegerInp(const PolynomialInteger &inp, int pos) = 0;
  virtual void _setBinaryInp(const PolynomialBinary &inp, int pos) = 0;

  virtual void _setMulPair(int left, int right, int result) = 0;

  virtual void _addAllOut(PolynomialTorus &out) = 0;
  virtual void _subAllOut(PolynomialTorus &out) = 0;

  virtual void _waitAll() = 0;

public:
  // Constructors
  BatchedFFT() = delete;
  BatchedFFT(const BatchedFFT &) = delete;

  static std::unique_ptr<BatchedFFT>
  createInstance(int N, int batch_inp = 1, int batch_out = 1,
                 bool isForcedToCheck = true);

  // Destructor
  virtual ~BatchedFFT();

  // Copy assignment operator
  virtual BatchedFFT &operator=(const BatchedFFT &obj) = delete;

  // Get params
  int get_N() const;
  int get_batch_inp() const;
  int get_batch_out() const;

  // Utilities
  bool setTorusInp(const PolynomialTorus &inp, int pos,
                   bool isForcedToCheck = true);
  bool setIntegerInp(const PolynomialInteger &inp, int pos,
                     bool isForcedToCheck = true);
  bool setBinaryInp(const PolynomialBinary &inp, int pos,
                    bool isForcedToCheck = true);

  bool setMulPair(int left, int right, int result, bool isForcedToCheck = true);

  bool addAllOut(PolynomialTorus &out, bool isForcedToCheck = true);
  bool subAllOut(PolynomialTorus &out, bool isForcedToCheck = true);

  void waitAll();
};

} // namespace thesis

#endif
