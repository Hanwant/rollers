cmake_minimum_required(VERSION 3.10)

project(rollers)


include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
# include_directories($"include")
set(PYTHON_INCLUDE_PATH "/home/hemu/miniconda3/envs/madigan/include/python3.7m")
set(PYTHON_LIBRARIES "/home/hemu/miniconda3/envs/madigan/lib/libpython3.7m.so")
# set(PYTHON_INCLUDE_DIR "/home/hemu/miniconda3/envs/madigan/include/python3.7m")
# set(PYTHON_LIBRARY "/home/hemu/miniconda3/envs/madigan/lib/libpython3.7m.so")



set(CMAKE_CXX_STANDARD 17)
set(CUSTOM_FLAGS "-O3 -march=native -ffast-math -fopenmp -mavx2 -mfma -g")# -DNDEBUG")#-g -fopt-info-vec")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUSTOM_FLAGS}")
# set(CMAKE_CXX_STANDARD_REQUIRED True)
# include_directories(/home/hemu/Downloads/FindPythonAnaconda-master)
# include(FindPythonAnaconda.cmake)
find_package(pybind11 REQUIRED)
# add_subdirectory(pybind11)
include_directories(${pybind11_INCLUDE_DIR})
find_package(cctz REQUIRED)
# find_package(HDF5 COMPONENTS C CXX HL REQUIRED)
find_package(Boost 1.73 COMPONENTS filesystem system iostreams program_options REQUIRED)
# link_directories( ${HDF5_LIBRARY_DIRS} )
# include_directories( ${HDF5_INCLUDE_DIR} )
# set (LINK_LIBS ${LINK_LIBS} ${HDF5_C_${LIB_TYPE}_LIBRARY})

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${Boost_LIBRARY_DIRS})
include_directories(${PYTHON_INCLUDE_PATH})
# link_directories(${PYTHON_INCLUDE_PATH})
link_libraries(${Boost_LIBRARIES})
link_libraries(${HDF5_LIBRARIES})
link_libraries(cctz::cctz)
link_libraries(${PYTHON_LIBRARIES})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# add_executable (main elements.cpp)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS src/*.cpp include/*.h)
pybind11_add_module(_rollers MODULE _rollers.cpp "${sources}")
# pybind11_add_module(rollers_testing MODULE rollers_testing.cpp)
# target_link_libraries(rollers PRIVATE ${MY_LIBRARIES})
# target_link_libraries (rollers PRIVATE ${HDF5_CXX_LIBRARIES})
# target_link_libraries(rollers PRIVATE ${Boost_LIBRARIES})
# target_link_libraries(rollers PRIVATE ${HDF5_LIBRARIES})
# add_library(rollers src/rollerX.cpp, src/rollerY.cpp)
# add_library(rollers "${sources}")
# add_executable(main elements.cpp movingWindow.cpp crud.cpp "${sources}")
# target_include_directories(main PUBLIC "~/Downloads/gnuplot-iostream")
# target_include_directories(main PUBLIC ${Boost_INCLUDE_DIRS})
# target_include_directories(main PUBLIC ${HDF5_INCLUDE_DIRS})
# target_link_libraries(main ${Boost_LIBRARIES})
# target_link_libraries(main ${HDF5_LIBRARIES})

message("HDF INCLUDE DIRS and LIBS ${HDF5_INCLUDE_DIRS} ${HDF5_LIBRARIES}")
message(STATUS "Boost version: ${Boost_VERSION}")
message(STATUS "Python Version: ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
message("Current source dir: ${CMAKE_CURRENT_SOURCE_DIR}")

#If boost version is newer than CMake version then this is needed
if(NOT TARGET Boost::filesystem)
  add_library(Boost::filesystem IMPORTED INTERFACE)
  set_property(TARGET Boost::filesystem PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIR})
  set_property(TARGET Boost::filesystem PROPERTY
    INTERFACE_LINK_LIBRARIES ${Boost_LIBRARIES})
  message("WARNING - Boost version is newer than CMake version")
endif()

enable_testing()
add_test(NAME rollers_test
  COMMAND pytest tests/ -s)
