cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(PointCloudsPython)
find_package(PCL 1.3 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_library(PointCloudsPython SHARED PointCloudsPython.cpp)
target_link_libraries(PointCloudsPython ${PCL_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES})
