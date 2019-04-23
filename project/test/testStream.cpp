#include "gtest/gtest.h"

#include "thesis/declarations.h"
#include "thesis/load_lib.h"
#include "thesis/stream.h"

#ifndef USING_CUDA
TEST(Thesis, StreamCPU) {
  std::srand(std::time(nullptr));
  std::vector<int> timeSleep(100);
  std::vector<int> tests(100, 0);
  for (int i = 0; i < 100; i++)
    timeSleep[i] = (std::rand() & 15) + 1;

  void *streamPtr = thesis::Stream::createS();
  int x = 0;
  for (int i = 0; i < 100; i++) {
    thesis::Stream::scheduleS(streamPtr, [&tests, &timeSleep, &x, i]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(timeSleep[i]));
      tests[i] = (x++);
    });
    if (!(rand() & 7)) {
      std::cout << "Synchronizing ... " << std::endl;
      thesis::Stream::synchronizeS(streamPtr);
    }
  }
  thesis::Stream::destroyS(streamPtr);

  for (int i = 0; i < 100; i++)
    ASSERT_TRUE(tests[i] == i);
}
#endif
