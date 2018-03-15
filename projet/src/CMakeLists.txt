project (features)

cmake_minimum_required(VERSION 2.8.11)

file(GLOB HDRS	"*.h")
file(GLOB SRCS	"*.cpp")
include_directories($ENV{Eigen3_DIR})
add_executable(features ${SRCS} ${HDRS})