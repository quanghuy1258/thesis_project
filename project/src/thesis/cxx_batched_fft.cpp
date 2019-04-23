#include "thesis/batched_fft.h"
#include "thesis/memory_management.h"
#include "thesis/stream.h"

#ifndef USING_CUDA
#include <fftw3.h>
#endif

namespace thesis {

BatchedFFT::BatchedFFT(int N, int row, int col) {
#if defined(USING_32BIT)
  const int mode = 4;
#else
  const int mode = 8;
#endif
  if (N < 2 || (N & (N - 1)) || row < 1 || col < 1)
    throw std::invalid_argument("N = 2^k with k > 0 ; row > 0 ; col > 0");
  _N = N;
  _row = row;
  _col = col;
  {
    std::complex<double> *ptr =
        (std::complex<double> *)MemoryManagement::mallocMM(
            (N * mode + 1) * (2 * row + 1) * col *
            sizeof(std::complex<double>));
    _data_inp.resize((row + 1) * col);
    _stream_inp.resize((row + 1) * col);
    for (int i = 0; i < (row + 1) * col; i++) {
      _data_inp[i] = ptr + (N * mode + 1) * i;
      _stream_inp[i] = Stream::createS();
    }
    _data_mul.resize(row * col);
    _stream_mul.resize(row * col);
    for (int i = 0; i < row * col; i++) {
      _data_mul[i] = ptr + (N * mode + 1) * ((row + 1) * col + i);
      _stream_mul[i] = Stream::createS();
    }
    _stream_out.resize(row);
    for (int i = 0; i < row; i++)
      _stream_out[i] = Stream::createS();
  }
#ifdef USING_CUDA
  cudaCreatePlan();
#else
  {
    fftw_plan *ptr = new fftw_plan[(2 * row + 1) * col];
    _plan_inp.resize((row + 1) * col);
    for (int i = 0; i < (row + 1) * col; i++) {
      ptr[i] =
          fftw_plan_dft_r2c_1d(N * 2 * mode, (double *)_data_inp[i],
                               (fftw_complex *)_data_inp[i], FFTW_ESTIMATE);
      _plan_inp[i] = ptr + i;
    }
    _plan_mul.resize(row * col);
    for (int i = 0; i < row * col; i++) {
      ptr[(row + 1) * col + i] =
          fftw_plan_dft_c2r_1d(N * 2 * mode, (fftw_complex *)_data_mul[i],
                               (double *)_data_mul[i], FFTW_ESTIMATE);
      _plan_mul[i] = ptr + (row + 1) * col + i;
    }
  }
#endif
}
BatchedFFT::~BatchedFFT() {
  MemoryManagement::freeMM(_data_inp[0]);
  for (int i = 0; i < (_row + 1) * _col; i++)
    Stream::destroyS(_stream_inp[i]);
  for (int i = 0; i < _row * _col; i++)
    Stream::destroyS(_stream_mul[i]);
  for (int i = 0; i < _row; i++)
    Stream::destroyS(_stream_out[i]);
#ifdef USING_CUDA
  cudaDestroyPlan();
#else
  {
    fftw_plan *ptr = (fftw_plan *)_plan_inp[0];
    delete[] ptr;
  }
#endif
}

int BatchedFFT::get_N() { return _N; }
int BatchedFFT::get_row() { return _row; }
int BatchedFFT::get_col() { return _col; }

void BatchedFFT::setInp(TorusInteger *pol, int r, int c) {
#if defined(USING_32BIT)
  const int mode = 4;
#else
  const int mode = 8;
#endif
  if (r < 0 || r >= _row || c < 0 || c >= _col)
    return;
  Stream::synchronizeS(_stream_mul[r * _col + c]);
#ifdef USING_CUDA
  cudaSetInp(pol, r, c);
#else
  Stream::scheduleS(_stream_inp[r * _col + c], [this, pol, r, c, mode]() {
    double *double_ptr = (double *)_data_inp[r * _col + c];
    for (int i = 0; i < _N; i++) {
      TorusInteger num = pol[i];
      for (int j = 0; j < mode / 2; j++) {
        double_ptr[i * mode + j] = num & 0xFFFF;
        num >>= 16;
        double_ptr[i * mode + j] /= 2.0;
        double_ptr[(i + _N) * mode + j] = -double_ptr[i * mode + j];
      }
      for (int j = mode / 2; j < mode; j++) {
        double_ptr[i * mode + j] = 0;
        double_ptr[(i + _N) * mode + j] = -double_ptr[i * mode + j];
      }
    }
    fftw_execute(*(fftw_plan *)_plan_inp[r * _col + c]);
  });
#endif
}
void BatchedFFT::setInp(TorusInteger *pol, int c) {
#if defined(USING_32BIT)
  const int mode = 4;
#else
  const int mode = 8;
#endif
  if (c < 0 || c >= _col)
    return;
  for (int i = 0; i < _row; i++)
    Stream::synchronizeS(_stream_mul[i * _col + c]);
#ifdef USING_CUDA
  cudaSetInp(pol, c);
#else
  Stream::scheduleS(_stream_inp[_row * _col + c], [this, pol, c, mode]() {
    double *double_ptr = (double *)_data_inp[_row * _col + c];
    for (int i = 0; i < _N; i++) {
      TorusInteger num = pol[i];
      for (int j = 0; j < mode / 2; j++) {
        double_ptr[i * mode + j] = num & 0xFFFF;
        num >>= 16;
        double_ptr[i * mode + j] /= 2.0;
        double_ptr[(i + _N) * mode + j] = -double_ptr[i * mode + j];
      }
      for (int j = mode / 2; j < mode; j++) {
        double_ptr[i * mode + j] = 0;
        double_ptr[(i + _N) * mode + j] = -double_ptr[i * mode + j];
      }
    }
    fftw_execute(*(fftw_plan *)_plan_inp[_row * _col + c]);
  });
#endif
}
void BatchedFFT::setMul(int r, int c) {
#if defined(USING_32BIT)
  const int mode = 4;
#else
  const int mode = 8;
#endif
  if (r < 0 || r >= _row || c < 0 || c >= _col)
    return;
  Stream::synchronizeS(_stream_inp[r * _col + c]);
  Stream::synchronizeS(_stream_inp[_row * _col + c]);
  Stream::synchronizeS(_stream_out[r]);
#ifdef USING_CUDA
  cudaSetMul(pol, r, c);
#else
  Stream::scheduleS(_stream_mul[r * _col + c], [this, r, c, mode]() {
    std::complex<double> *left =
        (std::complex<double> *)_data_inp[r * _col + c];
    std::complex<double> *right =
        (std::complex<double> *)_data_inp[_row * _col + c];
    std::complex<double> *result =
        (std::complex<double> *)_data_mul[r * _col + c];
    for (int i = 0; i <= _N * mode; i++)
      result[i] = left[i] * right[i];
    fftw_execute(*(fftw_plan *)_plan_mul[r * _col + c]);
    TorusInteger *torus_ptr = (TorusInteger *)result;
    double *double_ptr = (double *)result;
    for (int i = 0; i < _N; i++) {
      TorusInteger num = 0;
      for (int j = mode - 1; j >= 0; j--) {
        num <<= 16;
        num += std::llround(double_ptr[i * mode + j] / (_N * mode));
      }
      torus_ptr[i] = num;
    }
  });
#endif
}
void BatchedFFT::addAllOut(TorusInteger *pol, int r) {
  if (r < 0 || r >= _row)
    return;
  for (int i = 0; i < _col; i++)
    Stream::synchronizeS(_stream_mul[r * _col + i]);
#ifdef USING_CUDA
  cudaAddAllOut(pol, r);
#else
  Stream::scheduleS(_stream_out[r], [this, pol, r]() {
    for (int i = 0; i < _col; i++) {
      TorusInteger *torus_ptr = (TorusInteger *)_data_mul[r * _col + i];
      for (int j = 0; j < _N; j++)
        pol[j] += torus_ptr[j];
    }
  });
#endif
}
void BatchedFFT::subAllOut(TorusInteger *pol, int r) {
  if (r < 0 || r >= _row)
    return;
  for (int i = 0; i < _col; i++)
    Stream::synchronizeS(_stream_mul[r * _col + i]);
#ifdef USING_CUDA
  cudaSubAllOut(pol, r);
#else
  Stream::scheduleS(_stream_out[r], [this, pol, r]() {
    for (int i = 0; i < _col; i++) {
      TorusInteger *torus_ptr = (TorusInteger *)_data_mul[r * _col + i];
      for (int j = 0; j < _N; j++)
        pol[j] -= torus_ptr[j];
    }
  });
#endif
}
void BatchedFFT::waitAllOut() {
  for (int i = 0; i < _row; i++)
    Stream::synchronizeS(_stream_out[i]);
}

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
