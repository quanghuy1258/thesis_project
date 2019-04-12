#include <fftw3.h>

#include "thesis/batched_fft.h"
#include "thesis/threadpool.h"

#ifndef USING_CUDA

namespace thesis {

class FFTW : public BatchedFFT {
private:
  std::vector<std::complex<double>> _fft_out;
  std::vector<fftw_plan> _inp_plan;
  std::vector<fftw_plan> _out_plan;

public:
  // Constructors
  FFTW() = delete;
  FFTW(const FFTW &) = delete;
  FFTW(int N, int batch, int cache) : BatchedFFT(N, batch, cache) {
#if defined(USING_32BIT)
    const int mode = 4;
#else
    const int mode = 8;
#endif
    _fft_out.resize(_batch * (_N * mode + 1), 0);
    _inp_plan.resize(_batch, nullptr);
    _out_plan.resize(_batch, nullptr);
    for (int i = 0; i < _batch; i++) {
      _inp_plan[i] =
          fftw_plan_dft_r2c_1d(_N * 2 * mode, _inp.data() + i * _N * 2 * mode,
                               reinterpret_cast<fftw_complex *>(
                                   _fft_inp.data() + i * (_N * mode + 1)),
                               FFTW_ESTIMATE);
      _out_plan[i] =
          fftw_plan_dft_c2r_1d(_N * 2 * mode,
                               reinterpret_cast<fftw_complex *>(
                                   _fft_out.data() + i * (_N * mode + 1)),
                               _out.data() + i * _N * 2 * mode, FFTW_ESTIMATE);
    }
  }

  // Destructor
  ~FFTW() {
    for (int i = 0; i < _batch; i++) {
      fftw_destroy_plan(_inp_plan[i]);
      fftw_destroy_plan(_out_plan[i]);
    }
  }

  // Copy assignment operator
  using BatchedFFT::operator=;
  FFTW &operator=(const FFTW &obj) = delete;

  using BatchedFFT::doFFT;
  bool doFFT() {
    Eigen::Barrier barrier(_batch);
    for (int i = 0; i < _batch; i++) {
      ThreadPool::get_threadPool().Schedule([this, &barrier, i]() {
        fftw_execute(_inp_plan[i]);
        barrier.Notify();
      });
    }
    barrier.Wait();
    return true;
  }
  using BatchedFFT::doMultiplicationAndIFFT;
  bool doMultiplicationAndIFFT() {
    {
      const int numberThreads = ThreadPool::get_numberThreads();
      Eigen::Barrier barrier(numberThreads);
      for (int i = 0; i < numberThreads; i++) {
        ThreadPool::get_threadPool().Schedule(
            [this, &barrier, i, numberThreads]() {
#if defined(USING_32BIT)
              const int mode = 4;
#else
              const int mode = 8;
#endif
              int s = (_batch * _N * (mode / 2) * i) / numberThreads,
                  e = (_batch * _N * (mode / 2) * (i + 1)) / numberThreads;
              for (int it = s; it < e; it++) {
                int j = it / (_N * (mode / 2));
                int k = it % (_N * (mode / 2));
                int left = _multiplication_pair[j * 2];
                int right = _multiplication_pair[j * 2 + 1];
                _fft_out[j * (_N * mode + 1) + 2 * k + 1] =
                    _fft_inp[left * (_N * mode + 1) + 2 * k + 1] *
                    _fft_inp[right * (_N * mode + 1) + 2 * k + 1];
              }
              barrier.Notify();
            });
      }
      barrier.Wait();
    }
    {
      Eigen::Barrier barrier(_batch);
      for (int i = 0; i < _batch; i++) {
        ThreadPool::get_threadPool().Schedule([this, &barrier, i]() {
          fftw_execute(_out_plan[i]);
          barrier.Notify();
        });
      }
      barrier.Wait();
    }
    return true;
  }
};

BatchedFFT *BatchedFFT::newCustomInstance(int N, int batch, int cache) {
  return new FFTW(N, batch, cache);
}

} // namespace thesis

#endif
