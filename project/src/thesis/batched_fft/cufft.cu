#include "thesis/batched_fft.h"

#include <cuComplex.h>
#include <cufft.h>

#ifdef USING_CUDA

__global__ void multiply(int N, int batch, cuDoubleComplex *inp, int *id,
                         cuDoubleComplex *out) {
  int _N = blockIdx.x * blockDim.x + threadIdx.x;
  int _batch = blockIdx.y * blockDim.y + threadIdx.y;
  if (_batch < batch && _N < N) {
    int left = id[_batch * 2];
    int right = id[_batch * 2 + 1];
    out[_batch * N + _N] = cuCmul(inp[left * N + _N], inp[right * N + _N]);
  }
}

namespace thesis {

class CuFFT : public BatchedFFT {
private:
  int _isInitCode;
  int *_multiplication_pair_ptr;
  cufftDoubleComplex *_inp_data;
  cufftDoubleComplex *_out_data;
  cufftHandle _inp_plan;
  cufftHandle _out_plan;

public:
  // Constructors
  CuFFT() = delete;
  CuFFT(const CuFFT &) = delete;
  CuFFT(int N, int batch, int cache) : BatchedFFT(N, batch, cache) {
    _isInitCode = 0;
    _multiplication_pair_ptr = nullptr;
    _inp_data = nullptr;
    _out_data = nullptr;
    _inp_plan = 0;
    _out_plan = 0;
  }

  void clean() {
    if (_isInitCode & 16) {
      cufftDestroy(_out_plan);
      _out_plan = 0;
    }
    if (_isInitCode & 8) {
      cufftDestroy(_inp_plan);
      _inp_plan = 0;
    }
    if (_isInitCode & 4) {
      cudaFree(_out_data);
      _out_data = nullptr;
    }
    if (_isInitCode & 2) {
      cudaFree(_inp_data);
      _inp_data = nullptr;
    }
    if (_isInitCode & 1) {
      cudaFree(_multiplication_pair_ptr);
      _multiplication_pair_ptr = nullptr;
    }
    _isInitCode = 0;
  }
  bool init() {
#if defined(USING_32BIT)
    const int mode = 4;
#else
    const int mode = 8;
#endif
    if (cudaMalloc(&_multiplication_pair_ptr, sizeof(int) * _batch * 2) ==
        cudaSuccess)
      _isInitCode |= 1;
    if (cudaMalloc(&_inp_data, sizeof(cufftDoubleComplex) * (_N * mode + 1) *
                                   (_batch + _cache)) == cudaSuccess)
      _isInitCode |= 2;
    if (cudaMalloc(&_out_data, sizeof(cufftDoubleComplex) * (_N * mode + 1) *
                                   _batch) == cudaSuccess)
      _isInitCode |= 4;
    if (cufftPlan1d(&_inp_plan, _N * 2 * mode, CUFFT_D2Z, _batch) !=
        CUFFT_SUCCESS)
      _isInitCode |= 8;
    if (cufftPlan1d(&_out_plan, _N * 2 * mode, CUFFT_Z2D, _batch) !=
        CUFFT_SUCCESS)
      _isInitCode |= 16;
    if (_isInitCode == 31)
      return true;
    clean();
    return false;
  }

  // Destructor
  ~CuFFT() { clean(); }

  // Copy assignment operator
  using BatchedFFT::operator=;
  CuFFT &operator=(const CuFFT &obj) = delete;

  using BatchedFFT::doFFT;
  bool doFFT() {
#if defined(USING_32BIT)
    const int mode = 4;
#else
    const int mode = 8;
#endif
    if (_isInitCode == 0)
      return false;
    if (cudaMemcpy(_inp_data, _inp.data(),
                   sizeof(cufftDoubleReal) * _N * 2 * mode * _batch,
                   cudaMemcpyHostToDevice) != cudaSuccess)
      return false;
    if (cufftExecD2Z(_inp_plan, (cufftDoubleReal *)_inp_data, _inp_data) !=
        CUFFT_SUCCESS)
      return false;
    if (cudaDeviceSynchronize() != cudaSuccess)
      return false;
    if (cudaMemcpy(_fft_inp.data(), _inp_data,
                   sizeof(cufftDoubleComplex) * (_N * mode + 1) * _batch,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      return false;
    return true;
  }
  using BatchedFFT::doMultiplicationAndIFFT;
  bool doMultiplicationAndIFFT() {
#if defined(USING_32BIT)
    const int mode = 4;
#else
    const int mode = 8;
#endif
    if (_isInitCode == 0)
      return false;
    if (cudaMemcpy(_multiplication_pair_ptr, _multiplication_pair.data(),
                   sizeof(int) * _batch * 2,
                   cudaMemcpyHostToDevice) != cudaSuccess)
      return false;
    if (cudaMemcpy(_inp_data, _fft_inp.data(),
                   sizeof(cufftDoubleComplex) * (_N * mode + 1) *
                       (_batch + _cache),
                   cudaMemcpyHostToDevice) != cudaSuccess)
      return false;
    int threadsPerBlock = 512;
    // _N * mode + 512 = (_N * mode + 1) + (512 - 1)
    dim3 numBlocks((_N * mode + 512) / 512, _batch);
    multiply<<<numBlocks, threadsPerBlock>>>(
        _N * mode + 1, _batch, _inp_data, _multiplication_pair_ptr, _out_data);
    if (cufftExecZ2D(_out_plan, _out_data, (cufftDoubleReal *)_out_data) !=
        CUFFT_SUCCESS)
      return false;
    if (cudaDeviceSynchronize() != cudaSuccess)
      return false;
    if (cudaMemcpy(_out.data(), _out_data,
                   sizeof(cufftDoubleReal) * _N * 2 * mode * _batch,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      return false;
    return true;
  }
};

BatchedFFT *BatchedFFT::newCustomInstance(int N, int batch, int cache) {
  CuFFT *obj = new CuFFT(N, batch, cache);
  if (obj->init())
    return obj;
  delete obj;
  return nullptr;
}

} // namespace thesis

#endif
