# cmake_minimum_required(VERSION 2.8)
# project( benchmark )
# find_package( OpenCV REQUIRED )
# include_directories( ${OpenCV_INCLUDE_DIRS} )
# add_executable( benchmark benchmark.cpp )
# target_link_libraries( benchmark ${OpenCV_LIBS} )


cmake_minimum_required(VERSION 2.8)
project( opencv_benchmark )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( opencv_benchmark opencv_benchmark.cpp )
target_link_libraries( opencv_benchmark ${OpenCV_LIBS} )