#ifdef USING_CUDA

#include "thesis/batched_fft.h"

#include <cuComplex.h>
#include <cufft.h>

namespace thesis {
/*
__global__ void _expand(int N, cuDoubleComplex *data_inp) {
#if defined(USING_32BIT)
  const int mode = 4;
#else
  const int mode = 8;
#endif
  Torus *torus_ptr = (Torus *)data_inp;
  double *double_ptr = (double *)data_inp;
  for (int i = N - 1; i >= 0; i--) {
    Torus num = torus_ptr[i];
    for (int j = 0; j < mode / 2; j++) {
      double_ptr[i * mode + j] = num & 0xFFFF;
      num >>= 16;
      double_ptr[i * mode + j] /= 2.0;
      double_ptr[(i + N) * mode + j] = -double_ptr[i * mode + j];
    }
    for (int j = mode / 2; j < mode; j++) {
      double_ptr[i * mode + j] = 0;
      double_ptr[(i + N) * mode + j] = -double_ptr[i * mode + j];
    }
  }
}

__global__ void _multiply(int N, cuDoubleComplex *left, cuDoubleComplex *right,
                          cuDoubleComplex *result) {
  int _N = blockIdx.x * blockDim.x + threadIdx.x;
  if (_N < N)
    result[_N] = cuCmul(left[_N], right[_N]);
}

__global__ void _collapse(int N, cuDoubleComplex *data_mul) {
#if defined(USING_32BIT)
  const int mode = 4;
#else
  const int mode = 8;
#endif
  Torus *torus_ptr = (Torus *)data_mul;
  double *double_ptr = (double *)data_mul;
  for (int i = 0; i < N; i++) {
    Torus num = 0;
    for (int j = mode - 1; j >= 0; j--) {
      num <<= 16;
      num += llround(double_ptr[i * mode + j] / (N * mode));
    }
    torus_ptr[i] = num;
  }
}

__global__ void _add(int N, int batch, cuDoubleComplex *data_mul,
                     Torus *data_out) {
#if defined(USING_32BIT)
  const int mode = 4;
#else
  const int mode = 8;
#endif
  int _N = blockIdx.x * blockDim.x + threadIdx.x;
  if (_N < N) {
    for (int i = 0; i < batch; i++) {
      Torus *torus_ptr = (Torus *)&data_mul[(N * mode + 1) * i];
      data_out[_N] += torus_ptr[_N];
    }
  }
}

__global__ void _sub(int N, int batch, cuDoubleComplex *data_mul,
                     Torus *data_out) {
#if defined(USING_32BIT)
  const int mode = 4;
#else
  const int mode = 8;
#endif
  int _N = blockIdx.x * blockDim.x + threadIdx.x;
  if (_N < N) {
    for (int i = 0; i < batch; i++) {
      Torus *torus_ptr = (Torus *)&data_mul[i * (_N * mode + 1)];
      data_out[_N] -= torus_ptr[_N];
    }
  }
}

class CuFFT : public BatchedFFT {
private:
  std::vector<cudaStream_t> _stream_inp;
  cuDoubleComplex *_data_inp;
  std::vector<cufftHandle> _plan_inp;

  std::vector<cudaStream_t> _stream_mul;
  cuDoubleComplex *_data_mul;
  std::vector<cufftHandle> _plan_mul;

  cudaStream_t _stream_out;
  Torus *_data_out;

public:
  // Constructors
  CuFFT() = delete;
  CuFFT(const CuFFT &) = delete;
  CuFFT(int N, int batch_inp, int batch_out)
      : BatchedFFT(N, batch_inp, batch_out), _stream_inp(batch_inp, 0),
        _data_inp(0), _plan_inp(batch_inp, 0), _stream_mul(batch_out, 0),
        _data_mul(0), _plan_mul(batch_out, 0), _stream_out(0), _data_out(0) {}

  bool init();
  void clean();

  // Destructor
  ~CuFFT();

  // Copy assignment operator
  using BatchedFFT::operator=;
  CuFFT &operator=(const CuFFT &obj) = delete;

  void _setTorusInp(const PolynomialTorus &inp, int pos);
  void _setIntegerInp(const PolynomialInteger &inp, int pos);
  void _setBinaryInp(const PolynomialBinary &inp, int pos);

  void _setMulPair(int left, int right, int result);

  void _addAllOut(PolynomialTorus &out);
  void _subAllOut(PolynomialTorus &out);

  void _waitAll();
};

CuFFT::~CuFFT() {
  _waitAll();
  clean();
}

bool CuFFT::init() {
#if defined(USING_32BIT)
  const int mode = 4;
#else
  const int mode = 8;
#endif
  if (cudaMalloc(&_data_inp, sizeof(cuDoubleComplex) * (_N * mode + 1) *
                                 _batch_inp) != cudaSuccess) {
    _data_inp = 0;
    clean();
    return false;
  }
  for (int i = 0; i < _batch_inp; i++) {
    if (cudaStreamCreate(&_stream_inp[i]) != cudaSuccess) {
      _stream_inp[i] = 0;
      clean();
      return false;
    }
    if (cufftPlan1d(&_plan_inp[i], _N * 2 * mode, CUFFT_D2Z, 1) !=
        CUFFT_SUCCESS) {
      _plan_inp[i] = 0;
      clean();
      return false;
    }
    if (cufftSetStream(_plan_inp[i], _stream_inp[i]) != CUFFT_SUCCESS) {
      clean();
      return false;
    }
  }
  if (cudaMalloc(&_data_mul, sizeof(cuDoubleComplex) * (_N * mode + 1) *
                                 _batch_out) != cudaSuccess) {
    _data_mul = 0;
    clean();
    return false;
  }
  for (int i = 0; i < _batch_out; i++) {
    if (cudaStreamCreate(&_stream_mul[i]) != cudaSuccess) {
      _stream_mul[i] = 0;
      clean();
      return false;
    }
    if (cufftPlan1d(&_plan_mul[i], _N * 2 * mode, CUFFT_Z2D, 1) !=
        CUFFT_SUCCESS) {
      _plan_mul[i] = 0;
      clean();
      return false;
    }
    if (cufftSetStream(_plan_mul[i], _stream_mul[i]) != CUFFT_SUCCESS) {
      clean();
      return false;
    }
  }
  if (cudaMalloc(&_data_out, sizeof(Torus) * _N) != cudaSuccess) {
    _data_out = 0;
    clean();
    return false;
  }
  if (cudaStreamCreate(&_stream_out) != cudaSuccess) {
    _stream_out = 0;
    clean();
    return false;
  }
  return true;
}
void CuFFT::clean() {
  if (_stream_out) {
    cudaStreamDestroy(_stream_out);
    _stream_out = 0;
  }
  if (_data_out) {
    cudaFree(_data_out);
    _data_out = 0;
  }
  for (int i = 0; i < _batch_out; i++) {
    if (_plan_mul[i]) {
      cufftDestroy(_plan_mul[i]);
      _plan_mul[i] = 0;
    }
    if (_stream_mul[i]) {
      cudaStreamDestroy(_stream_mul[i]);
      _stream_mul[i] = 0;
    }
  }
  if (_data_mul) {
    cudaFree(_data_mul);
    _data_mul = 0;
  }
  for (int i = 0; i < _batch_inp; i++) {
    if (_plan_inp[i]) {
      cufftDestroy(_plan_inp[i]);
      _plan_inp[i] = 0;
    }
    if (_stream_inp[i]) {
      cudaStreamDestroy(_stream_inp[i]);
      _stream_inp[i] = 0;
    }
  }
  if (_data_inp) {
    cudaFree(_data_inp);
    _data_inp = 0;
  }
}

void CuFFT::_setTorusInp(const PolynomialTorus &inp, int pos) {
#if defined(USING_32BIT)
  const int mode = 4;
#else
  const int mode = 8;
#endif
  for (int i = 0; i < _batch_out; i++)
    cudaStreamSynchronize(_stream_mul[i]);
  cudaMemcpyAsync(&_data_inp[(_N * mode + 1) * pos], inp.data(),
                  sizeof(Torus) * _N, cudaMemcpyHostToDevice, _stream_inp[pos]);
  _expand<<<1, 1, 0, _stream_inp[pos]>>>(_N, &_data_inp[(_N * mode + 1) * pos]);
  cufftExecD2Z(_plan_inp[pos], (double *)&_data_inp[(_N * mode + 1) * pos],
               &_data_inp[(_N * mode + 1) * pos]);
}
void CuFFT::_setIntegerInp(const PolynomialInteger &inp, int pos) {
  _setTorusInp(inp, pos);
}
void CuFFT::_setBinaryInp(const PolynomialBinary &inp, int pos) {
#if defined(USING_32BIT)
  const int mode = 4;
#else
  const int mode = 8;
#endif
  std::vector<double> temp_inp(_N * 2 * mode);
  for (int i = 0; i < _N; i++) {
    for (int j = 0; j < mode; j++) {
      temp_inp[i * mode + j] = 0;
      temp_inp[(i + _N) * mode + j] = -temp_inp[i * mode + j];
    }
    if (!inp[i])
      continue;
    temp_inp[i * mode] = 0.5;
    temp_inp[(i + _N) * mode] = -0.5;
  }
  for (int i = 0; i < _batch_out; i++)
    cudaStreamSynchronize(_stream_mul[i]);
  cudaMemcpy(&_data_inp[(_N * mode + 1) * pos], temp_inp.data(),
             sizeof(double) * _N * 2 * mode, cudaMemcpyHostToDevice);
  cufftExecD2Z(_plan_inp[pos], (double *)&_data_inp[(_N * mode + 1) * pos],
               &_data_inp[(_N * mode + 1) * pos]);
}

void CuFFT::_setMulPair(int left, int right, int result) {
#if defined(USING_32BIT)
  const int mode = 4;
#else
  const int mode = 8;
#endif
  cudaStreamSynchronize(_stream_inp[left]);
  cudaStreamSynchronize(_stream_inp[right]);
  cudaStreamSynchronize(_stream_out);
  int threadsPerBlock = 512;
  // _N * mode + 512 = (_N * mode + 1) + (512 - 1)
  int numBlocks = (_N * mode + 512) / 512;
  _multiply<<<numBlocks, threadsPerBlock, 0, _stream_mul[result]>>>(
      _N * mode + 1, &_data_inp[(_N * mode + 1) * left],
      &_data_inp[(_N * mode + 1) * right],
      &_data_mul[(_N * mode + 1) * result]);
  cufftExecZ2D(_plan_mul[result], &_data_mul[(_N * mode + 1) * result],
               (double *)&_data_mul[(_N * mode + 1) * result]);
  _collapse<<<1, 1, 0, _stream_mul[result]>>>(
      _N, &_data_mul[(_N * mode + 1) * result]);
}

void CuFFT::_addAllOut(PolynomialTorus &out) {
  cudaMemcpyAsync(_data_out, out.data(), sizeof(Torus) * _N,
                  cudaMemcpyHostToDevice, _stream_out);
  for (int i = 0; i < _batch_out; i++)
    cudaStreamSynchronize(_stream_mul[i]);
  int threadsPerBlock = 512;
  // _N + 511 = _N + (512 - 1)
  int numBlocks = (_N + 511) / 512;
  _add<<<numBlocks, threadsPerBlock, 0, _stream_out>>>(_N, _batch_out,
                                                       _data_mul, _data_out);
  cudaMemcpyAsync(out.data(), _data_out, sizeof(Torus) * _N,
                  cudaMemcpyDeviceToHost, _stream_out);
}
void CuFFT::_subAllOut(PolynomialTorus &out) {
  cudaMemcpyAsync(_data_out, out.data(), sizeof(Torus) * _N,
                  cudaMemcpyHostToDevice, _stream_out);
  for (int i = 0; i < _batch_out; i++)
    cudaStreamSynchronize(_stream_mul[i]);
  int threadsPerBlock = 512;
  // _N + 511 = _N + (512 - 1)
  int numBlocks = (_N + 511) / 512;
  _sub<<<numBlocks, threadsPerBlock, 0, _stream_out>>>(_N, _batch_out,
                                                       _data_mul, _data_out);
  cudaMemcpyAsync(out.data(), _data_out, sizeof(Torus) * _N,
                  cudaMemcpyDeviceToHost, _stream_out);
}

void CuFFT::_waitAll() { cudaDeviceSynchronize(); }

BatchedFFT *BatchedFFT::_createInstance(int N, int batch_inp, int batch_out) {
  CuFFT *obj = new CuFFT(N, batch_inp, batch_out);
  if (obj->init())
    return obj;
  delete obj;
  return nullptr;
}
*/
} // namespace thesis

#endif
