PROJECT(neuralNet)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11 -pthread")

SET(EXECUTABLE_OUTPUT_PATH ../bin/)

# list of the source files
set(neuralnet_files ../src/neuralnet.cpp)
set(testfunctions ../src/testfunctions.cpp)
set(neuralnet_p_files ../src/neuralnet_p.cpp)
set(neuralnet_p_printimages_files ../src/neuralnet_p_printimages.cpp)
# define the executable
if(WIN32)
  add_executable(neuralNet WIN32 ${neuralnet_files})
  add_executable(testfunctions WIN32 ${testfunctions})
  add_executable(neuralNet_p WIN32 ${neuralnet_p_files})
  add_executable(neuralNet_p_printimages WIN32 ${neuralnet_p_printimages_files})
else()
  add_executable(neuralNet ${neuralnet_files})
  add_executable(testfunctions ${testfunctions})
  add_executable(neuralNet_p ${neuralnet_p_files})
  add_executable(neuralNet_p_printimages ${neuralnet_p_printimages_files})
endif()
# in Windows + Visual Studio we need this to make it a console application
if(WIN32)
  if(MSVC)
    set_target_properties(neuralNet PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
    set_target_properties(testfunctions PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
    set_target_properties(neuralNet_p PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
    set_target_properties(neuralNet_p_printimages PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
  endif(MSVC)
endif(WIN32)
# install application
