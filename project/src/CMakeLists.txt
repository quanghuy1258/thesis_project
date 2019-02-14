#
# Add Executable
#
AUX_SOURCE_DIRECTORY ("${CMAKE_CURRENT_SOURCE_DIR}/thesis" THESIS)
add_executable(main main.cpp ${THESIS})

