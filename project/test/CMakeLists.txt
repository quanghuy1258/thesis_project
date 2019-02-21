#
# Add Executable
#
ADD_EXECUTABLE (mainTest main.cpp example.cpp threadpool_eigen.cpp)
TARGET_LINK_LIBRARIES (mainTest gtest eigen)
