#include "thesis/batched_fft.h"

namespace thesis {

BatchedFFT::BatchedFFT(int N, int row, int col) {
  if (N < 2 || (N & (N - 1)) || row < 1 || col < 1)
    throw std::invalid_argument("N = 2^k with k > 0 ; row > 0 ; col > 0");
  _N = N;
  _row = row;
  _col = col;
}
BatchedFFT::~BatchedFFT() {}
/*
// Constructors
BatchedFFT::BatchedFFT(int N, int batch_inp, int batch_out)
    : _N(N), _batch_inp(batch_inp), _batch_out(batch_out) {}

std::unique_ptr<BatchedFFT> BatchedFFT::createInstance(int N, int batch_inp,
                                                       int batch_out,
                                                       bool isForcedToCheck) {
  std::unique_ptr<BatchedFFT> ptr;
  if (!isForcedToCheck ||
      (N > 1 && (N & (N - 1)) == 0 && batch_inp > 0 && batch_out > 0))
    ptr.reset(_createInstance(N, batch_inp, batch_out));
  return std::move(ptr);
}

// Destructor
BatchedFFT::~BatchedFFT() {}

// Get params
int BatchedFFT::get_N() const { return _N; }
int BatchedFFT::get_batch_inp() const { return _batch_inp; }
int BatchedFFT::get_batch_out() const { return _batch_out; }

// Utilities
bool BatchedFFT::setTorusInp(const PolynomialTorus &inp, int pos,
                             bool isForcedToCheck) {
  const int inp_size = inp.size();
  if (isForcedToCheck && (inp_size != _N || pos < 0 || pos >= _batch_inp))
    return false;
  _setTorusInp(inp, pos);
  return true;
}
bool BatchedFFT::setIntegerInp(const PolynomialInteger &inp, int pos,
                               bool isForcedToCheck) {
  const int inp_size = inp.size();
  if (isForcedToCheck && (inp_size != _N || pos < 0 || pos >= _batch_inp))
    return false;
  _setIntegerInp(inp, pos);
  return true;
}
bool BatchedFFT::setBinaryInp(const PolynomialBinary &inp, int pos,
                              bool isForcedToCheck) {
  const int inp_size = inp.size();
  if (isForcedToCheck && (inp_size != _N || pos < 0 || pos >= _batch_inp))
    return false;
  _setBinaryInp(inp, pos);
  return true;
}

bool BatchedFFT::setMulPair(int left, int right, int result,
                            bool isForcedToCheck) {
  if (isForcedToCheck &&
      (left < 0 || left >= _batch_inp || right < 0 || right >= _batch_inp ||
       result < 0 || result >= _batch_out))
    return false;
  _setMulPair(left, right, result);
  return true;
}

bool BatchedFFT::addAllOut(PolynomialTorus &out, bool isForcedToCheck) {
  const int out_size = out.size();
  if (isForcedToCheck && out_size != _N)
    return false;
  _addAllOut(out);
  return true;
}
bool BatchedFFT::subAllOut(PolynomialTorus &out, bool isForcedToCheck) {
  const int out_size = out.size();
  if (isForcedToCheck && out_size != _N)
    return false;
  _subAllOut(out);
  return true;
}

void BatchedFFT::waitAll() { _waitAll(); }
*/
} // namespace thesis
