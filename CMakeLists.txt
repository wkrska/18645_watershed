# cmake_minimum_required(VERSION 2.8)
# project( benchmark )
# find_package( OpenCV REQUIRED )
# include_directories( ${OpenCV_INCLUDE_DIRS} )
# add_executable( benchmark benchmark.cpp )
# target_link_libraries( benchmark ${OpenCV_LIBS} )


cmake_minimum_required(VERSION 2.8)
project( test )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( test test.cpp )
target_link_libraries( test ${OpenCV_LIBS} )