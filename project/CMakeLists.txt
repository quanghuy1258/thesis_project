#
# Guard against in-source builds
#
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	MESSAGE (FATAL_ERROR "In-source builds not allowed. Please change to build directory and run CMake from there. You may need to remove CMakeCache.txt and CMakeFiles directory.")
endif()

#
# CMake functions
#
FUNCTION (print_all_variables)
	GET_CMAKE_PROPERTY (_variableNames VARIABLES)
	LIST (SORT _variableNames)
	foreach(_variableName ${_variableNames})
		MESSAGE (STATUS "${_variableName}=${${_variableName}}")
	endforeach()
ENDFUNCTION (print_all_variables)
FUNCTION (print_all_include_directories)
	GET_PROPERTY (dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
	foreach(dir ${dirs})
		MESSAGE (STATUS "include_directory='${dir}'")
	endforeach()
ENDFUNCTION (print_all_include_directories)

#
# Project Properties
#
CMAKE_MINIMUM_REQUIRED (VERSION 3.5)
PROJECT (thesis_project C CXX)
if(NOT CMAKE_BUILD_TYPE)
	MESSAGE (STATUS "No build type selected, default to Release")
	SET (CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release" FORCE)
endif()
SET (CMAKE_EXPORT_COMPILE_COMMANDS ON)
SET (CMAKE_CXX_STANDARD 14)
SET (CMAKE_CXX_STANDARD_REQUIRED ON)
SET (CMAKE_CXX_FLAGS "-O3 -Wall")

#
# Check requirements
#
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.1")
		MESSAGE (FATAL_ERROR "Insufficient gcc or g++ version")
	endif()
else()
	MESSAGE (FATAL_ERROR "Require C++ compiler (gcc and g++)")
endif()

#
# Debugging Options
#
if(NOT CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]")
	SET (CMAKE_VERBOSE_MAKEFILE 0) # Use 1 for debugging, 0 for release
else()
	SET (CMAKE_VERBOSE_MAKEFILE 1) # Use 1 for debugging, 0 for release
endif()

#
# Project Output Paths
#
SET (MAINFOLDER ${PROJECT_SOURCE_DIR})
SET (EXECUTABLE_OUTPUT_PATH "${MAINFOLDER}/bin")
SET (LIBRARY_OUTPUT_PATH "${MAINFOLDER}/lib")

#
# Project Search Paths
#
LIST (APPEND CMAKE_PREFIX_PATH "${MAINFOLDER}")
ADD_SUBDIRECTORY(thirdparty)
INCLUDE_DIRECTORIES("${MAINFOLDER}/include")

#
# Add packages
#
FIND_PACKAGE(Threads)

#
# Add Build Targets
#
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(benchmark)

#
# For debugging
#
if(NOT CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]")
	RETURN()
endif()
MESSAGE ("===== DEBUG MODE =====")
MESSAGE ("===== PRINT ALL VARIABLES =====")
print_all_variables()
print_all_include_directories()
