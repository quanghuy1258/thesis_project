#
# Add Executable
#
ADD_EXECUTABLE (mainBenchmark main.cpp example.cpp openblas.cpp)
TARGET_LINK_LIBRARIES (mainBenchmark benchmark openblas)
