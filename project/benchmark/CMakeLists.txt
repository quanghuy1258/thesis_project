#
# Add Executable
#
ADD_EXECUTABLE (mainBenchmark main.cpp example.cpp)
TARGET_LINK_LIBRARIES (mainBenchmark benchmark)
