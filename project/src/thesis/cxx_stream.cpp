#include "thesis/barrier.h"
#include "thesis/stream.h"
#include "thesis/thread_management.h"

namespace thesis {

#ifndef USING_CUDA
class PrivateStream {
private:
  std::mutex _mtx;
  std::condition_variable _cv;
  bool _ready;

  int _num;
  std::queue<std::function<void(int, int)>> _job_que;
  std::queue<int> _parallel_que;

  std::mutex _mtx_task;
  int _numDoingTask;

  void doNextJob() {
    std::function<void(int, int)> fn;
    int parallel;
    {
      std::lock_guard<std::mutex> guard(_mtx);
      fn = std::move(_job_que.front());
      parallel = _parallel_que.front();
      --_num;
      _job_que.pop();
      _parallel_que.pop();
    }
    if (parallel > 1) {
      _numDoingTask = parallel;
      for (int i = 0; i < parallel; i++)
        ThreadManagement::schedule([this, fn, i, parallel]() {
          fn(i, parallel);
          doMultitask();
        });
    } else {
      fn(0, 1);
      checkNextJob();
    }
  }
  void doMultitask() {
    bool isLastTask;
    {
      std::lock_guard<std::mutex> guard(_mtx_task);
      isLastTask = !(--_numDoingTask);
    }
    if (isLastTask)
      checkNextJob();
  }
  void checkNextJob() {
    std::lock_guard<std::mutex> guard(_mtx);
    if (_num)
      ThreadManagement::schedule([this]() { doNextJob(); });
    else {
      _ready = true;
      _cv.notify_all();
    }
  }

public:
  PrivateStream() {
    _ready = true;
    _num = 0;
    _numDoingTask = 0;
  }
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
    ++_num;
    _job_que.push([fn](int, int) { fn(); });
    _parallel_que.push(1);
    if (_ready) {
      ThreadManagement::schedule([this]() { doNextJob(); });
      _ready = false;
    }
  }
  void schedulePS(std::function<void(int, int)> fn, int parallel) {
    if (parallel < 1)
      return;
    std::lock_guard<std::mutex> guard(_mtx);
    ++_num;
    _job_que.push(std::move(fn));
    _parallel_que.push(parallel);
    if (_ready) {
      ThreadManagement::schedule([this]() { doNextJob(); });
      _ready = false;
    }
  }
};

void Stream::scheduleS(std::function<void()> fn, void *streamPtr) {
  if (streamPtr) {
    PrivateStream *ptr = (PrivateStream *)streamPtr;
    ptr->schedulePS(std::move(fn));
  } else
    fn();
}
void Stream::scheduleS(std::function<void(int, int)> fn, int parallel,
                       void *streamPtr) {
  if (parallel < 1)
    return;
  if (streamPtr) {
    PrivateStream *ptr = (PrivateStream *)streamPtr;
    ptr->schedulePS(std::move(fn), parallel);
  } else if (parallel > 1) {
    Barrier bar;
    bar.reset(parallel);
    for (int i = 0; i < parallel; i++)
      ThreadManagement::schedule([&fn, i, parallel, &bar]() {
        fn(i, parallel);
        bar.notify();
      });
    bar.wait();
  } else
    fn(0, 1);
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
