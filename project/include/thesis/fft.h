#ifndef THESIS_FFT_H
#define THESIS_FFT_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

#define USING_FFTW_FFT

namespace thesis {

class FFT {
private:
  int _N;

  void *_a;
  void *_b;
  void *_result;

  void renewVectorFFT(void *&pointer, int n);
  void deleteVectorFFT(void *&pointer);

  void torusPolynomialToFFT(void *out, const PolynomialTorus &inp);
  void integerPolynomialToFFT(void *out, const PolynomialInteger &inp);
  void binaryPolynomialToFFT(void *out, const PolynomialBinary &inp);
  void torusPolynomialFromFFT(PolynomialTorus &out, void *inp);
  void fftMultiplication(void *result, void *a, void *b);

public:
  // Constructors
  FFT() {
    _N = 0;
    _a = nullptr;
    _b = nullptr;
    _result = nullptr;
  }
  FFT(const FFT &obj) = delete;

  // Destructor
  ~FFT() {
    _N = 0;
    deleteVectorFFT(_a);
    deleteVectorFFT(_b);
    deleteVectorFFT(_result);
  }

  // Copy assignment operator
  FFT &operator=(const FFT &obj) = delete;

  // Get params
  int get_N() const { return _N; }

  // Set attributes
  bool set_N(int n) {
    if (n < 2 || n & (n - 1)) // n = 2^k, k >= 1
      return false;

    if (_N != n) {
      _N = n;
      renewVectorFFT(_a, _N);
      renewVectorFFT(_b, _N);
      renewVectorFFT(_result, _N);
    }
    if (_a == nullptr || _b == nullptr || _result == nullptr) {
      deleteVectorFFT(_a);
      deleteVectorFFT(_b);
      deleteVectorFFT(_result);
      _N = 0;
      return false;
    }
    return true;
  }

  // Utilities
  bool torusPolynomialMultiplication(PolynomialTorus &result,
                                     const PolynomialInteger &a,
                                     const PolynomialTorus &b) {
    if (_N == 0 || (signed)a.size() != _N || (signed)b.size() != _N)
      return false;
    result.resize(_N);
    integerPolynomialToFFT(_a, a);
    torusPolynomialToFFT(_b, b);
    fftMultiplication(_result, _a, _b);
    torusPolynomialFromFFT(result, _result);
    return true;
  }
  bool torusPolynomialMultiplication(PolynomialTorus &result,
                                     const PolynomialBinary &a,
                                     const PolynomialTorus &b) {
    if (_N == 0 || (signed)a.size() != _N || (signed)b.size() != _N)
      return false;
    result.resize(_N);
    binaryPolynomialToFFT(_a, a);
    torusPolynomialToFFT(_b, b);
    fftMultiplication(_result, _a, _b);
    torusPolynomialFromFFT(result, _result);
    return true;
  }
};

} // namespace thesis

#endif
