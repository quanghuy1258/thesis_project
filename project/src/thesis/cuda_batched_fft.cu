#ifdef USING_CUDA

#include "thesis/batched_fft.h"

#include <cuComplex.h>
#include <cufft.h>

namespace thesis {

#if defined(USING_32BIT)
const int mode = 4;
#else
const int mode = 8;
#endif

__global__ void _expand(int N, TorusInteger *pol, void *data_inp) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (n < N && m < mode) {
    double *double_ptr = (double *)data_inp;
    if (m < mode / 2) {
      TorusInteger num = pol[n];
      for (int i = 0; i < m; i++)
        num >>= 16;
      double_ptr[n * mode + m] = num & 0xFFFF;
      double_ptr[n * mode + m] /= 2.0;
      double_ptr[(n + N) * mode + m] = -double_ptr[n * mode + m];
    } else {
      double_ptr[n * mode + m] = 0.0;
      double_ptr[(n + N) * mode + m] = -double_ptr[n * mode + m];
    }
  }
}

__global__ void _multiply(int length, void *left, void *right, void *result) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length) {
    cuDoubleComplex *_left = (cuDoubleComplex *)left;
    cuDoubleComplex *_right = (cuDoubleComplex *)right;
    cuDoubleComplex *_result = (cuDoubleComplex *)result;
    _result[i] = cuCmul(_left[i], _right[i]);
  }
}

__global__ void _collapse(int N, void *data_mul) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    TorusInteger *torus_ptr = (TorusInteger *)data_mul;
    double *double_ptr = (double *)data_mul;
    TorusInteger num = 0;
    for (int i = mode / 2 - 1; i >= 0; i--) {
      num <<= 16;
      num += llround(-double_ptr[(n + N) * mode + i] / (N * mode));
    }
    torus_ptr[n] = num;
  }
}

__global__ void _add(int N, int col, TorusInteger *pol, void *data_mul) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    cuDoubleComplex *_data_mul = (cuDoubleComplex *)data_mul;
    for (int i = 0; i < col; i++) {
      TorusInteger *torus_ptr =
          (TorusInteger *)(_data_mul + (N * mode + 1) * i);
      pol[n] += torus_ptr[n];
    }
  }
}

__global__ void _sub(int N, int col, TorusInteger *pol, void *data_mul) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    cuDoubleComplex *_data_mul = (cuDoubleComplex *)data_mul;
    for (int i = 0; i < col; i++) {
      TorusInteger *torus_ptr =
          (TorusInteger *)(_data_mul + (N * mode + 1) * i);
      pol[n] -= torus_ptr[n];
    }
  }
}

void BatchedFFT::cudaCreatePlan() {
  _plan_inp.resize((_row + 1) * _col);
  for (int i = 0; i < (_row + 1) * _col; i++) {
    cufftHandle *cufftHandle_ptr =
        (cufftHandle *)std::malloc(sizeof(cufftHandle));
    cudaStream_t *cudaStream_t_ptr = (cudaStream_t *)_stream_inp[i];
    if (cufftPlan1d(cufftHandle_ptr, _N * 2 * mode, CUFFT_D2Z, 1) !=
        CUFFT_SUCCESS) {
      std::free(cufftHandle_ptr);
      _plan_inp[i] = nullptr;
      throw std::runtime_error("Cannot create cufftPlan1d");
    } else {
      _plan_inp[i] = cufftHandle_ptr;
      cufftSetStream(*cufftHandle_ptr, *cudaStream_t_ptr);
    }
  }
  _plan_mul.resize(_row * _col);
  for (int i = 0; i < _row * _col; i++) {
    cufftHandle *cufftHandle_ptr =
        (cufftHandle *)std::malloc(sizeof(cufftHandle));
    cudaStream_t *cudaStream_t_ptr = (cudaStream_t *)_stream_mul[i];
    if (cufftPlan1d(cufftHandle_ptr, _N * 2 * mode, CUFFT_Z2D, 1) !=
        CUFFT_SUCCESS) {
      std::free(cufftHandle_ptr);
      _plan_mul[i] = nullptr;
      throw std::runtime_error("Cannot create cufftPlan1d");
    } else {
      _plan_mul[i] = cufftHandle_ptr;
      cufftSetStream(*cufftHandle_ptr, *cudaStream_t_ptr);
    }
  }
}
void BatchedFFT::cudaDestroyPlan() {
  for (int i = 0; i < (_row + 1) * _col; i++) {
    if (_plan_inp[i]) {
      cufftHandle *cufftHandle_ptr = (cufftHandle *)_plan_inp[i];
      cufftDestroy(*cufftHandle_ptr);
      std::free(cufftHandle_ptr);
    }
  }
  for (int i = 0; i < _row * _col; i++) {
    if (_plan_mul[i]) {
      cufftHandle *cufftHandle_ptr = (cufftHandle *)_plan_mul[i];
      cufftDestroy(*cufftHandle_ptr);
      std::free(cufftHandle_ptr);
    }
  }
}
void BatchedFFT::cudaSetInp(TorusInteger *pol, int r, int c) {
  int threadsPerBlock = 512;
  // _N * mode + 512 = (_N * mode + 1) + (512 - 1)
  dim3 numBlocks((_N * mode + 512) / 512, mode);
  cudaStream_t *cudaStream_t_ptr = (cudaStream_t *)_stream_inp[r * _col + c];
  _expand<<<numBlocks, threadsPerBlock, 0, *cudaStream_t_ptr>>>(
      _N, pol, _data_inp[r * _col + c]);
  cufftExecD2Z(*(cufftHandle *)_plan_inp[r * _col + c],
               (double *)_data_inp[r * _col + c],
               (cuDoubleComplex *)_data_inp[r * _col + c]);
}
void BatchedFFT::cudaSetInp(TorusInteger *pol, int c) {
  int threadsPerBlock = 512;
  // _N * mode + 512 = (_N * mode + 1) + (512 - 1)
  dim3 numBlocks((_N * mode + 512) / 512, mode);
  cudaStream_t *cudaStream_t_ptr = (cudaStream_t *)_stream_inp[_row * _col + c];
  _expand<<<numBlocks, threadsPerBlock, 0, *cudaStream_t_ptr>>>(
      _N, pol, _data_inp[_row * _col + c]);
  cufftExecD2Z(*(cufftHandle *)_plan_inp[_row * _col + c],
               (double *)_data_inp[_row * _col + c],
               (cuDoubleComplex *)_data_inp[_row * _col + c]);
}
void BatchedFFT::cudaSetMul(int r, int c) {
  int threadsPerBlock = 512;
  // _N * mode + 512 = (_N * mode + 1) + (512 - 1)
  int numBlocks = (_N * mode + 512) / 512;
  cudaStream_t *cudaStream_t_ptr = (cudaStream_t *)_stream_mul[r * _col + c];
  _multiply<<<numBlocks, threadsPerBlock, 0, *cudaStream_t_ptr>>>(
      _N * mode + 1, _data_inp[r * _col + c], _data_inp[_row * _col + c],
      _data_mul[r * _col + c]);
  cufftExecZ2D(*(cufftHandle *)_plan_mul[r * _col + c],
               (cuDoubleComplex *)_data_mul[r * _col + c],
               (double *)_data_mul[r * _col + c]);
  _collapse<<<numBlocks, threadsPerBlock, 0, *cudaStream_t_ptr>>>(
      _N, _data_mul[r * _col + c]);
}
void BatchedFFT::cudaAddAllOut(TorusInteger *pol, int r) {
  int threadsPerBlock = 512;
  // _N + 511 = _N + (512 - 1)
  int numBlocks = (_N + 511) / 512;
  cudaStream_t *cudaStream_t_ptr = (cudaStream_t *)_stream_out[r];
  _add<<<numBlocks, threadsPerBlock, 0, *cudaStream_t_ptr>>>(
      _N, _col, pol, _data_mul[r * _col]);
}
void BatchedFFT::cudaSubAllOut(TorusInteger *pol, int r) {
  int threadsPerBlock = 512;
  // _N + 511 = _N + (512 - 1)
  int numBlocks = (_N + 511) / 512;
  cudaStream_t *cudaStream_t_ptr = (cudaStream_t *)_stream_out[r];
  _sub<<<numBlocks, threadsPerBlock, 0, *cudaStream_t_ptr>>>(
      _N, _col, pol, _data_mul[r * _col]);
}

} // namespace thesis

#endif
