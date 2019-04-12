#ifndef THESIS_BATCHED_FFT_H
#define THESIS_BATCHED_FFT_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"
#include "thesis/threadpool.h"

namespace thesis {

class BatchedFFT {
protected:
  int _N;
  int _batch;
  int _cache;
  std::vector<int> _multiplication_pair;
  std::vector<double> _inp;
  std::vector<std::complex<double>> _fft_inp;
  std::vector<std::complex<double>> _fft_out;
  std::vector<double> _out;

  // Constructors
  BatchedFFT(int N, int batch, int cache) {
#if defined(USING_32BIT)
    const int mode = 4;
#else
    const int mode = 8;
#endif
    _N = N;
    _batch = batch;
    _cache = cache;
    _multiplication_pair.resize(_batch * 2, 0);
    _inp.resize(_batch * _N * 2 * mode, 0);
    _fft_inp.resize((_batch + _cache) * (_N * mode + 1), 0);
    _fft_out.resize(_batch * (_N * mode + 1), 0);
    _out.resize(_batch * _N * 2 * mode, 0);
  }

  static BatchedFFT *newCustomInstance(int N, int batch, int cache);

public:
  // Constructors
  BatchedFFT() = delete;
  BatchedFFT(const BatchedFFT &) = delete;

  static std::unique_ptr<BatchedFFT>
  createInstance(int N, int batch = 1, int cache = 0,
                 bool isForcedToCheck = true) {
    std::unique_ptr<BatchedFFT> ptr;
    if (!isForcedToCheck ||
        (N > 1 && (N & (N - 1)) == 0 && batch > 0 && cache >= 0))
      ptr.reset(newCustomInstance(N, batch, cache));
    return std::move(ptr);
  }

  // Destructor
  virtual ~BatchedFFT() {}

  // Copy assignment operator
  virtual BatchedFFT &operator=(const BatchedFFT &obj) = delete;

  // Get params
  int get_N() const { return _N; }
  int get_batch() const { return _batch; }
  int get_cache() const { return _cache; }

  // Utilities
  bool setTorusInput(const PolynomialTorus &inp, int pos,
                     Eigen::Barrier *notifier = nullptr,
                     bool isForcedToCheck = true) {
    const int inp_size = inp.size();
    if (isForcedToCheck && (inp_size != _N || pos < 0 || pos >= _batch))
      return false;
    std::function<void()> fn = [this, &inp, pos, notifier]() {
#if defined(USING_32BIT)
      const int mode = 4;
#else
      const int mode = 8;
#endif
      double *ptr = _inp.data() + pos * _N * 2 * mode;
      std::memset(ptr, 0, _N * 2 * mode * sizeof(double));
      for (int i = 0; i < _N; i++) {
        Torus num = inp[i];
        for (int j = 0; j < mode / 2; j++) {
          ptr[i * mode + j] = num & 0xFFFF;
          num >>= 16;
          ptr[i * mode + j] /= 2;
          ptr[(i + _N) * mode + j] = -ptr[i * mode + j];
        }
      }
      if (notifier != nullptr)
        notifier->Notify();
    };
    if (notifier != nullptr) {
      ThreadPool::get_threadPool().Schedule(std::move(fn));
    } else {
      fn();
    }
    return true;
  }
  bool setIntegerInput(const PolynomialInteger &inp, int pos,
                       Eigen::Barrier *notifier = nullptr,
                       bool isForcedToCheck = true) {
    return setTorusInput(inp, pos, notifier, isForcedToCheck);
  }
  bool setBinaryInput(const PolynomialBinary &inp, int pos,
                      Eigen::Barrier *notifier = nullptr,
                      bool isForcedToCheck = true) {
    const int inp_size = inp.size();
    if (isForcedToCheck && (inp_size != _N || pos < 0 || pos >= _batch))
      return false;
    std::function<void()> fn = [this, &inp, pos, notifier]() {
#if defined(USING_32BIT)
      const int mode = 4;
#else
      const int mode = 8;
#endif
      double *ptr = _inp.data() + pos * _N * 2 * mode;
      std::memset(ptr, 0, _N * 2 * mode * sizeof(double));
      for (int i = 0; i < _N; i++) {
        if (!inp[i])
          continue;
        ptr[i * mode] = 0.5;
        ptr[(i + _N) * mode] = -0.5;
      }
      if (notifier != nullptr)
        notifier->Notify();
    };
    if (notifier != nullptr) {
      ThreadPool::get_threadPool().Schedule(std::move(fn));
    } else {
      fn();
    }
    return true;
  }
  bool copyTo(int from, int to, Eigen::Barrier *notifier = nullptr,
              bool isForcedToCheck = true) {
    if (isForcedToCheck && (from < 0 || from >= _batch + _cache || to < 0 ||
                            to >= _batch + _cache))
      return false;
    std::function<void()> fn = [this, from, to, notifier]() {
#if defined(USING_32BIT)
      const int mode = 4;
#else
      const int mode = 8;
#endif
      std::memcpy(_fft_inp.data() + to * (_N * mode + 1),
                  _fft_inp.data() + from * (_N * mode + 1),
                  (_N * mode + 1) * sizeof(std::complex<double>));
      if (notifier != nullptr)
        notifier->Notify();
    };
    if (notifier != nullptr) {
      ThreadPool::get_threadPool().Schedule(std::move(fn));
    } else {
      fn();
    }
    return true;
  }
  bool setMultiplicationPair(int left, int right, int result,
                             bool isForcedToCheck = true) {
    if (isForcedToCheck &&
        (left < 0 || left >= _batch + _cache || right < 0 ||
         right >= _batch + _cache || result < 0 || result >= _batch))
      return false;
    _multiplication_pair[result * 2] = left;
    _multiplication_pair[result * 2 + 1] = right;
    return true;
  }
  bool getOutput(PolynomialTorus &out, int pos,
                 Eigen::Barrier *notifier = nullptr,
                 bool isForcedToCheck = true) const {
    const int out_size = out.size();
    if (isForcedToCheck && (out_size != _N || pos < 0 || pos >= _batch))
      return false;
    std::function<void()> fn = [this, &out, pos, notifier]() {
#if defined(USING_32BIT)
      const int mode = 4;
#else
      const int mode = 8;
#endif
      const double *ptr = _out.data() + pos * _N * 2 * mode;
      for (int i = 0; i < _N; i++) {
        out[i] = 0;
        for (int j = mode / 2 - 1; j >= 0; j--) {
          out[i] <<= 16;
          out[i] += std::llround(ptr[i * mode + j] / (_N * mode));
        }
      }
      if (notifier != nullptr)
        notifier->Notify();
    };
    if (notifier != nullptr) {
      ThreadPool::get_threadPool().Schedule(std::move(fn));
    } else {
      fn();
    }
    return true;
  }
  bool addOutput(PolynomialTorus &out, int pos,
                 Eigen::Barrier *notifier = nullptr,
                 bool isForcedToCheck = true) const {
    const int out_size = out.size();
    if (isForcedToCheck && (out_size != _N || pos < 0 || pos >= _batch))
      return false;
    std::function<void()> fn = [this, &out, pos, notifier]() {
#if defined(USING_32BIT)
      const int mode = 4;
#else
      const int mode = 8;
#endif
      const double *ptr = _out.data() + pos * _N * 2 * mode;
      for (int i = 0; i < _N; i++) {
        Torus num = 0;
        for (int j = mode / 2 - 1; j >= 0; j--) {
          num <<= 16;
          num += std::llround(ptr[i * mode + j] / (_N * mode));
        }
        out[i] += num;
      }
      if (notifier != nullptr)
        notifier->Notify();
    };
    if (notifier != nullptr) {
      ThreadPool::get_threadPool().Schedule(std::move(fn));
    } else {
      fn();
    }
    return true;
  }
  bool subOutput(PolynomialTorus &out, int pos,
                 Eigen::Barrier *notifier = nullptr,
                 bool isForcedToCheck = true) const {
    const int out_size = out.size();
    if (isForcedToCheck && (out_size != _N || pos < 0 || pos >= _batch))
      return false;
    std::function<void()> fn = [this, &out, pos, notifier]() {
#if defined(USING_32BIT)
      const int mode = 4;
#else
      const int mode = 8;
#endif
      const double *ptr = _out.data() + pos * _N * 2 * mode;
      for (int i = 0; i < _N; i++) {
        Torus num = 0;
        for (int j = mode / 2 - 1; j >= 0; j--) {
          num <<= 16;
          num += std::llround(ptr[i * mode + j] / (_N * mode));
        }
        out[i] -= num;
      }
      if (notifier != nullptr)
        notifier->Notify();
    };
    if (notifier != nullptr) {
      ThreadPool::get_threadPool().Schedule(std::move(fn));
    } else {
      fn();
    }
    return true;
  }

  virtual void doFFT() = 0;
  virtual void doIFFT() = 0;
  virtual void doMultiplication() = 0;
}; // namespace thesis

} // namespace thesis

#endif
