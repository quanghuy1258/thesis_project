#ifndef USING_CUDA

#include "thesis/batched_fft.h"
#include "thesis/threadpool.h"

#include <fftw3.h>

namespace thesis {
/*
class FFTW : public BatchedFFT {
private:
  std::vector<std::complex<double>> _data_inp;
  std::vector<fftw_plan> _plan_inp;

  std::vector<std::complex<double>> _data_mul;
  std::vector<fftw_plan> _plan_mul;

  std::vector<std::unique_ptr<Eigen::Barrier>> _notifier_inp;
  std::vector<std::unique_ptr<Eigen::Barrier>> _notifier_mul;
  std::unique_ptr<Eigen::Barrier> _notifier_out;

public:
  // Constructors
  FFTW() = delete;
  FFTW(const FFTW &) = delete;
  FFTW(int N, int batch_inp, int batch_out)
      : BatchedFFT(N, batch_inp, batch_out), _notifier_inp(batch_inp),
        _notifier_mul(batch_out) {
#if defined(USING_32BIT)
    const int mode = 4;
#else
    const int mode = 8;
#endif
    _data_inp.resize((N * mode + 1) * batch_inp, 0);
    _plan_inp.resize(batch_inp, 0);
    for (int i = 0; i < batch_inp; i++) {
      _plan_inp[i] = fftw_plan_dft_r2c_1d(
          N * 2 * mode,
          reinterpret_cast<double *>(&_data_inp[(_N * mode + 1) * i]),
          reinterpret_cast<fftw_complex *>(&_data_inp[(_N * mode + 1) * i]),
          FFTW_ESTIMATE);
    }
    _data_mul.resize((N * mode + 1) * batch_out, 0);
    _plan_mul.resize(batch_out, 0);
    for (int i = 0; i < batch_out; i++) {
      _plan_mul[i] = fftw_plan_dft_c2r_1d(
          N * 2 * mode,
          reinterpret_cast<fftw_complex *>(&_data_mul[(_N * mode + 1) * i]),
          reinterpret_cast<double *>(&_data_mul[(_N * mode + 1) * i]),
          FFTW_ESTIMATE);
    }
  }

  // Destructor
  ~FFTW();

  // Copy assignment operator
  using BatchedFFT::operator=;
  FFTW &operator=(const FFTW &obj) = delete;

  void _setTorusInp(const PolynomialTorus &inp, int pos);
  void _setIntegerInp(const PolynomialInteger &inp, int pos);
  void _setBinaryInp(const PolynomialBinary &inp, int pos);

  void _setMulPair(int left, int right, int result);

  void _addAllOut(PolynomialTorus &out);
  void _subAllOut(PolynomialTorus &out);

  void _waitAll();
};

FFTW::~FFTW() {
  _waitAll();
  for (int i = 0; i < _batch_inp; i++) {
    fftw_destroy_plan(_plan_inp[i]);
  }
  for (int i = 0; i < _batch_out; i++) {
    fftw_destroy_plan(_plan_mul[i]);
  }
}

void FFTW::_setTorusInp(const PolynomialTorus &inp, int pos) {
#if defined(USING_32BIT)
  const int mode = 4;
#else
  const int mode = 8;
#endif
  for (int i = 0; i < _batch_out; i++) {
    if (_notifier_mul[i])
      _notifier_mul[i]->Wait();
  }
  if (_notifier_inp[pos])
    _notifier_inp[pos]->Wait();
  _notifier_inp[pos].reset(new Eigen::Barrier(1));
  ThreadPool::get_threadPool().Schedule([this, &inp, pos, mode]() {
    double *double_ptr =
        reinterpret_cast<double *>(&_data_inp[(_N * mode + 1) * pos]);
    for (int i = 0; i < _N; i++) {
      Torus num = inp[i];
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
    fftw_execute(_plan_inp[pos]);
    _notifier_inp[pos]->Notify();
  });
}
void FFTW::_setIntegerInp(const PolynomialInteger &inp, int pos) {
  _setTorusInp(inp, pos);
}
void FFTW::_setBinaryInp(const PolynomialBinary &inp, int pos) {
#if defined(USING_32BIT)
  const int mode = 4;
#else
  const int mode = 8;
#endif
  for (int i = 0; i < _batch_out; i++) {
    if (_notifier_mul[i])
      _notifier_mul[i]->Wait();
  }
  if (_notifier_inp[pos])
    _notifier_inp[pos]->Wait();
  _notifier_inp[pos].reset(new Eigen::Barrier(1));
  ThreadPool::get_threadPool().Schedule([this, &inp, pos, mode]() {
    double *double_ptr =
        reinterpret_cast<double *>(&_data_inp[(_N * mode + 1) * pos]);
    for (int i = 0; i < _N; i++) {
      for (int j = 0; j < mode; j++) {
        double_ptr[i * mode + j] = 0;
        double_ptr[(i + _N) * mode + j] = -double_ptr[i * mode + j];
      }
      if (!inp[i])
        continue;
      double_ptr[i * mode] = 0.5;
      double_ptr[(i + _N) * mode] = -0.5;
    }
    fftw_execute(_plan_inp[pos]);
    _notifier_inp[pos]->Notify();
  });
}

void FFTW::_setMulPair(int left, int right, int result) {
#if defined(USING_32BIT)
  const int mode = 4;
#else
  const int mode = 8;
#endif
  if (_notifier_inp[left])
    _notifier_inp[left]->Wait();
  if (_notifier_inp[right])
    _notifier_inp[right]->Wait();
  if (_notifier_out)
    _notifier_out->Wait();
  if (_notifier_mul[result])
    _notifier_mul[result]->Wait();
  _notifier_mul[result].reset(new Eigen::Barrier(1));
  ThreadPool::get_threadPool().Schedule([this, left, right, result, mode]() {
    std::complex<double> *_left = &_data_inp[(_N * mode + 1) * left];
    std::complex<double> *_right = &_data_inp[(_N * mode + 1) * right];
    std::complex<double> *_result = &_data_mul[(_N * mode + 1) * result];
    for (int i = 0; i <= _N * mode; i++) {
      _result[i] = _left[i] * _right[i];
    }
    fftw_execute(_plan_mul[result]);
    Torus *torus_ptr = reinterpret_cast<Torus *>(_result);
    double *double_ptr = reinterpret_cast<double *>(_result);
    for (int i = 0; i < _N; i++) {
      Torus num = 0;
      for (int j = mode - 1; j >= 0; j--) {
        num <<= 16;
        num += std::llround(double_ptr[i * mode + j] / (_N * mode));
      }
      torus_ptr[i] = num;
    }
    _notifier_mul[result]->Notify();
  });
}

void FFTW::_addAllOut(PolynomialTorus &out) {
#if defined(USING_32BIT)
  const int mode = 4;
#else
  const int mode = 8;
#endif
  for (int i = 0; i < _batch_out; i++) {
    if (_notifier_mul[i])
      _notifier_mul[i]->Wait();
  }
  const int numberThreads = ThreadPool::get_numberThreads();
  if (_notifier_out)
    _notifier_out->Wait();
  _notifier_out.reset(new Eigen::Barrier(numberThreads));
  for (int i = 0; i < numberThreads; i++) {
    ThreadPool::get_threadPool().Schedule(
        [this, &out, i, numberThreads, mode]() {
          int s = (_N * i) / numberThreads, e = (_N * (i + 1)) / numberThreads;
          for (int j = 0; j < _batch_out; j++) {
            Torus *torus_ptr =
                reinterpret_cast<Torus *>(&_data_mul[(_N * mode + 1) * j]);
            for (int it = s; it < e; it++)
              out[it] += torus_ptr[it];
          }
          _notifier_out->Notify();
        });
  }
}
void FFTW::_subAllOut(PolynomialTorus &out) {
#if defined(USING_32BIT)
  const int mode = 4;
#else
  const int mode = 8;
#endif
  for (int i = 0; i < _batch_out; i++) {
    if (_notifier_mul[i])
      _notifier_mul[i]->Wait();
  }
  const int numberThreads = ThreadPool::get_numberThreads();
  if (_notifier_out)
    _notifier_out->Wait();
  _notifier_out.reset(new Eigen::Barrier(numberThreads));
  for (int i = 0; i < numberThreads; i++) {
    ThreadPool::get_threadPool().Schedule(
        [this, &out, i, numberThreads, mode]() {
          int s = (_N * i) / numberThreads, e = (_N * (i + 1)) / numberThreads;
          for (int j = 0; j < _batch_out; j++) {
            Torus *torus_ptr =
                reinterpret_cast<Torus *>(&_data_mul[(_N * mode + 1) * j]);
            for (int it = s; it < e; it++)
              out[it] -= torus_ptr[it];
          }
          _notifier_out->Notify();
        });
  }
}

void FFTW::_waitAll() {
  for (int i = 0; i < _batch_inp; i++) {
    if (_notifier_inp[i])
      _notifier_inp[i]->Wait();
  }
  for (int i = 0; i < _batch_out; i++) {
    if (_notifier_mul[i])
      _notifier_mul[i]->Wait();
  }
  if (_notifier_out)
    _notifier_out->Wait();
}

BatchedFFT *BatchedFFT::_createInstance(int N, int batch_inp, int batch_out) {
  return new FFTW(N, batch_inp, batch_out);
}
*/
} // namespace thesis

#endif
