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
  obj->_normalForm = fftw_alloc_real(n << 3);
  obj->_fftForm = fftw_alloc_complex((n << 2) + 1);
  obj->_normal2fft = fftw_plan_dft_r2c_1d(n << 3, obj->_normalForm,
                                          obj->_fftForm, FFTW_ESTIMATE);
  obj->_fft2normal = fftw_plan_dft_c2r_1d(n << 3, obj->_fftForm,
                                          obj->_normalForm, FFTW_ESTIMATE);
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
  for (int i = 0; i < _N; i++) {
    uint32_t temp = (unsigned)inp[i];
    obj->_normalForm[(i << 2)] = (temp & 0xFFFF);
    obj->_normalForm[(i << 2) + 1] = (temp >> 16);
    obj->_normalForm[(i << 2) + 2] = obj->_normalForm[(i << 2) + 3] = 0;
    obj->_normalForm[(_N << 2) + (i << 2)] =
        obj->_normalForm[(_N << 2) + (i << 2) + 1] =
            obj->_normalForm[(_N << 2) + (i << 2) + 2] =
                obj->_normalForm[(_N << 2) + (i << 2) + 3] = 0;
  }
  fftw_execute(obj->_normal2fft);
}
void FFT::integerPolynomialToFFT(void *out, const PolynomialInteger &inp) {
  FFT_Data *obj = (FFT_Data *)out;
  for (int i = 0; i < _N; i++) {
    uint32_t temp = (unsigned)((int32_t)inp[i]);
    obj->_normalForm[(i << 2)] = (temp & 0xFFFF);
    obj->_normalForm[(i << 2) + 1] = (temp >> 16);
    obj->_normalForm[(i << 2) + 2] = obj->_normalForm[(i << 2) + 3] = 0;
    obj->_normalForm[(_N << 2) + (i << 2)] =
        obj->_normalForm[(_N << 2) + (i << 2) + 1] =
            obj->_normalForm[(_N << 2) + (i << 2) + 2] =
                obj->_normalForm[(_N << 2) + (i << 2) + 3] = 0;
  }
  fftw_execute(obj->_normal2fft);
}
void FFT::binaryPolynomialToFFT(void *out, const PolynomialBinary &inp) {
  FFT_Data *obj = (FFT_Data *)out;
  for (int i = 0; i < _N; i++) {
    obj->_normalForm[(i << 2)] = (inp[i]) ? 1 : 0;
    obj->_normalForm[(i << 2) + 1] = obj->_normalForm[(i << 2) + 2] =
        obj->_normalForm[(i << 2) + 3] =
            obj->_normalForm[(_N << 2) + (i << 2)] =
                obj->_normalForm[(_N << 2) + (i << 2) + 1] =
                    obj->_normalForm[(_N << 2) + (i << 2) + 2] =
                        obj->_normalForm[(_N << 2) + (i << 2) + 3] = 0;
  }
  fftw_execute(obj->_normal2fft);
}
void FFT::torusPolynomialFromFFT(PolynomialTorus &out, void *inp) {
  FFT_Data *obj = (FFT_Data *)inp;
  fftw_execute(obj->_fft2normal);
  for (int i = 0; i < _N; i++) {
    uint32_t temp_0 =
        (unsigned)std::llround(obj->_normalForm[(i << 2)] / (_N << 3));
    uint32_t temp_1 =
        (unsigned)std::llround(obj->_normalForm[(i << 2) + 1] / (_N << 3));
    out[i] = temp_0 + (temp_1 << 16);
  }
}
void FFT::fftMultiplication(void *result, void *a, void *b) {
  FFT_Data *temp_a = (FFT_Data *)a;
  FFT_Data *temp_b = (FFT_Data *)b;
  FFT_Data *temp_result = (FFT_Data *)result;
  for (int i = 0; i <= (_N << 2); i++) {
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
