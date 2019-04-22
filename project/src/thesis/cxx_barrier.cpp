#include "thesis/barrier.h"

#include <unsupported/Eigen/CXX11/ThreadPool>

namespace thesis {

Barrier::Barrier() { _data = nullptr; }
Barrier::~Barrier() {
  if (_data) {
    wait();
    Eigen::Barrier *ptr = (Eigen::Barrier *)_data;
    delete ptr;
  }
}

void Barrier::reset(size_t count) {
  if (_data) {
    wait();
    Eigen::Barrier *ptr = (Eigen::Barrier *)_data;
    delete ptr;
  }
  if (count)
    _data = new Eigen::Barrier(count);
  else
    _data = nullptr;
}
void Barrier::notify() {
  if (_data) {
    Eigen::Barrier *ptr = (Eigen::Barrier *)_data;
    ptr->Notify();
  }
}
void Barrier::wait() {
  if (_data) {
    Eigen::Barrier *ptr = (Eigen::Barrier *)_data;
    ptr->Wait();
  }
}

} // namespace thesis

