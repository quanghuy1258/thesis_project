# googletest
ADD_SUBDIRECTORY (googletest EXCLUDE_FROM_ALL)

# benchmark
SET (BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Message: Disable BENCHMARK_GTEST_TESTS")
ADD_SUBDIRECTORY (benchmark EXCLUDE_FROM_ALL)

# eigen
ADD_SUBDIRECTORY (eigen-git-mirror EXCLUDE_FROM_ALL)

# fftw
INCLUDE (fftw/external_fftw.cmake)
