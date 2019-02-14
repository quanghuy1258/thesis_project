
#
# googletest
#
add_subdirectory(googletest EXCLUDE_FROM_ALL)

#
# benchmark
#
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Message: Disable BENCHMARK_GTEST_TESTS")
add_subdirectory(benchmark EXCLUDE_FROM_ALL)
