#include "thesis/stream.h"
#include "thesis/thread_management.h"

namespace thesis {

#ifndef USING_CUDA
class PrivateStream {
private:
  std::mutex _mtx;
  std::condition_variable _cv;
  bool _ready;

  std::queue<std::function<void()>> _jobs;

  void doNextJob() {
    std::function<void()> fn;
    {
      std::lock_guard<std::mutex> guard(_mtx);
      fn = std::move(_jobs.front());
      _jobs.pop();
    }
    fn();
    {
      std::lock_guard<std::mutex> guard(_mtx);
      if (_jobs.empty()) {
        _ready = true;
        _cv.notify_all();
      } else
        ThreadManagement::schedule([this]() { doNextJob(); });
    }
  }

public:
  PrivateStream() { _ready = true; }
  PrivateStream(const PrivateStream &) = delete;
  PrivateStream &operator=(const PrivateStream &) = delete;

  ~PrivateStream() { synchronizePS(); }

  void synchronizePS() {
    std::unique_lock<std::mutex> lck(_mtx);
    while (!_ready)
      _cv.wait(lck);
  }
  void schedulePS(std::function<void()> fn) {
    std::lock_guard<std::mutex> guard(_mtx);
    _jobs.push(std::move(fn));
    if (_ready) {
      ThreadManagement::schedule([this]() { doNextJob(); });
      _ready = false;
    }
  }
};

void Stream::scheduleS(void *streamPtr, std::function<void()> fn) {
  if (streamPtr) {
    PrivateStream *ptr = (PrivateStream *)streamPtr;
    ptr->schedulePS(std::move(fn));
  } else
    fn();
}

#endif

void *Stream::createS() {
#ifdef USING_CUDA
  return cudaCreateS();
#else
  return new PrivateStream();
#endif
}
void Stream::synchronizeS(void *streamPtr) {
  if (!streamPtr)
    return;
#ifdef USING_CUDA
  cudaSynchronizeS(streamPtr);
#else
  PrivateStream *ptr = (PrivateStream *)streamPtr;
  ptr->synchronizePS();
#endif
}
void Stream::destroyS(void *streamPtr) {
  if (!streamPtr)
    return;
#ifdef USING_CUDA
  cudaDestroyS(streamPtr);
#else
  PrivateStream *ptr = (PrivateStream *)streamPtr;
  delete ptr;
#endif
}

} // namespace thesis
