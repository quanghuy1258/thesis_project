#ifndef THESIS_BATCHED_FFT_H
#define THESIS_BATCHED_FFT_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class BatchedFFT {
private:
  int _N;
  int _row;
  int _col;

  std::vector<void *> _data_inp;
  std::vector<void *> _data_mul;

  std::vector<void *> _plan_inp;
  std::vector<void *> _plan_mul;

  std::vector<void *> _stream_inp;
  std::vector<void *> _stream_mul;
  std::vector<void *> _stream_out;

#ifdef USING_CUDA
  void cudaCreatePlan();
  void cudaDestroyPlan();
  void cudaSetInp(TorusInteger *pol, int r, int c);
  void cudaSetInp(TorusInteger *pol, int c);
  void cudaSetMul(int r, int c);
  void cudaAddAllOut(TorusInteger *pol, int r);
  void cudaSubAllOut(TorusInteger *pol, int r);
#endif

public:
  BatchedFFT() = delete;
  BatchedFFT(const BatchedFFT &) = delete;
  BatchedFFT(int N, int row, int col);

  BatchedFFT &operator=(const BatchedFFT &) = delete;

  ~BatchedFFT();

  int get_N();
  int get_row();
  int get_col();

  void setInp(TorusInteger *pol, int r, int c);
  void setInp(TorusInteger *pol, int c);
  void setMul(int r, int c);
  void addAllOut(TorusInteger *pol, int r);
  void subAllOut(TorusInteger *pol, int r);
  void waitOut(int r);
  void waitAllOut();
};

} // namespace thesis

#endif
