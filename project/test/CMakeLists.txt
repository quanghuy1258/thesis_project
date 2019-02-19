#
# Add Executable
#
ADD_EXECUTABLE (mainTest main.cpp example.cpp openblas.cpp)
TARGET_LINK_LIBRARIES (mainTest gtest openblas)
