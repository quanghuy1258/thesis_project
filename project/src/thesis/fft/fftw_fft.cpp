#include <fftw3.h>

#include "thesis/fft.h"

#ifdef USING_FFTW_FFT

namespace thesis {

typedef struct {
  double *_normalForm;
  fftw_complex *_fftForm;
  fftw_plan _normal2fft;
  fftw_plan _fft2normal;
} FFT_Data;

void FFT::renewVectorFFT(void *&pointer, int n) {
  deleteVectorFFT(pointer);
  if (n < 2 || n & (n - 1))
    return;
  FFT_Data *obj = new FFT_Data;
#if defined(USING_32BIT)
  int newN = n << 3;
#else
  int newN = n << 4;
#endif
  obj->_normalForm = fftw_alloc_real(newN);
  obj->_fftForm = fftw_alloc_complex((newN >> 1) + 1);
  obj->_normal2fft = fftw_plan_dft_r2c_1d(newN, obj->_normalForm, obj->_fftForm,
                                          FFTW_ESTIMATE);
  obj->_fft2normal = fftw_plan_dft_c2r_1d(newN, obj->_fftForm, obj->_normalForm,
                                          FFTW_ESTIMATE);
  pointer = obj;
}
void FFT::deleteVectorFFT(void *&pointer) {
  if (pointer != nullptr) {
    FFT_Data *obj = (FFT_Data *)pointer;
    fftw_free(obj->_normalForm);
    fftw_free(obj->_fftForm);
    fftw_destroy_plan(obj->_normal2fft);
    fftw_destroy_plan(obj->_fft2normal);
    delete obj;
    pointer = nullptr;
  }
}

void FFT::torusPolynomialToFFT(void *out, const PolynomialTorus &inp) {
  FFT_Data *obj = (FFT_Data *)out;
#if defined(USING_32BIT)
  int newN = _N << 3;
#else
  int newN = _N << 4;
#endif
  for (int i = 0; i < newN; i++) {
    obj->_normalForm[i] = 0;
  }
  for (int i = 0; i < _N; i++) {
#if defined(USING_32BIT)
    uint32_t temp = inp[i];
    obj->_normalForm[(i << 2)] = (temp & 0xFFFF);
    obj->_normalForm[(i << 2) + 1] = ((temp >> 16) & 0xFFFF);
#else
    uint64_t temp = inp[i];
    obj->_normalForm[(i << 3)] = (temp & 0xFFFF);
    obj->_normalForm[(i << 3) + 1] = ((temp >> 16) & 0xFFFF);
    obj->_normalForm[(i << 3) + 2] = ((temp >> 32) & 0xFFFF);
    obj->_normalForm[(i << 3) + 3] = ((temp >> 48) & 0xFFFF);
#endif
  }
  fftw_execute(obj->_normal2fft);
}
void FFT::integerPolynomialToFFT(void *out, const PolynomialInteger &inp) {
  FFT_Data *obj = (FFT_Data *)out;
#if defined(USING_32BIT)
  int newN = _N << 3;
#else
  int newN = _N << 4;
#endif
  for (int i = 0; i < newN; i++) {
    obj->_normalForm[i] = 0;
  }
  for (int i = 0; i < _N; i++) {
#if defined(USING_32BIT)
    uint32_t temp = inp[i];
    obj->_normalForm[(i << 2)] = (temp & 0xFFFF);
    obj->_normalForm[(i << 2) + 1] = ((temp >> 16) & 0xFFFF);
#else
    uint64_t temp = inp[i];
    obj->_normalForm[(i << 3)] = (temp & 0xFFFF);
    obj->_normalForm[(i << 3) + 1] = ((temp >> 16) & 0xFFFF);
    obj->_normalForm[(i << 3) + 2] = ((temp >> 32) & 0xFFFF);
    obj->_normalForm[(i << 3) + 3] = ((temp >> 48) & 0xFFFF);
#endif
  }
  fftw_execute(obj->_normal2fft);
}
void FFT::binaryPolynomialToFFT(void *out, const PolynomialBinary &inp) {
  FFT_Data *obj = (FFT_Data *)out;
#if defined(USING_32BIT)
  int newN = _N << 3;
#else
  int newN = _N << 4;
#endif
  for (int i = 0; i < newN; i++) {
    obj->_normalForm[i] = 0;
  }
  for (int i = 0; i < _N; i++) {
#if defined(USING_32BIT)
    obj->_normalForm[(i << 2)] = (inp[i]) ? 1 : 0;
#else
    obj->_normalForm[(i << 3)] = (inp[i]) ? 1 : 0;
#endif
  }
  fftw_execute(obj->_normal2fft);
}
void FFT::torusPolynomialFromFFT(PolynomialTorus &out, void *inp) {
  FFT_Data *obj = (FFT_Data *)inp;
#if defined(USING_32BIT)
  int newN = _N << 3;
#else
  int newN = _N << 4;
#endif
  fftw_execute(obj->_fft2normal);
  for (int i = 0; i < _N; i++) {
#if defined(USING_32BIT)
    uint64_t temp[2];
    temp[0] = std::llround(obj->_normalForm[(i << 2)] / newN);
    temp[1] = std::llround(obj->_normalForm[(i << 2) + 1] / newN);
    out[i] = (temp[0] + (temp[1] << 16));
#else
    uint64_t temp[4];
    temp[0] = std::llround(obj->_normalForm[(i << 3)] / newN);
    temp[1] = std::llround(obj->_normalForm[(i << 3) + 1] / newN);
    temp[2] = std::llround(obj->_normalForm[(i << 3) + 2] / newN);
    temp[3] = std::llround(obj->_normalForm[(i << 3) + 3] / newN);
    out[i] = (temp[0] + (temp[1] << 16) + (temp[2] << 32) + (temp[3] << 48));
#endif
  }
}
void FFT::fftMultiplication(void *result, void *a, void *b) {
  FFT_Data *temp_a = (FFT_Data *)a;
  FFT_Data *temp_b = (FFT_Data *)b;
  FFT_Data *temp_result = (FFT_Data *)result;
#if defined(USING_32BIT)
  int newN = _N << 3;
#else
  int newN = _N << 4;
#endif
  for (int i = 0; i <= (newN >> 1); i++) {
    temp_result->_fftForm[i][0] =
        temp_a->_fftForm[i][0] * temp_b->_fftForm[i][0] -
        temp_a->_fftForm[i][1] * temp_b->_fftForm[i][1];
    temp_result->_fftForm[i][1] =
        temp_a->_fftForm[i][0] * temp_b->_fftForm[i][1] +
        temp_a->_fftForm[i][1] * temp_b->_fftForm[i][0];
  }
}

} // namespace thesis

#endif
