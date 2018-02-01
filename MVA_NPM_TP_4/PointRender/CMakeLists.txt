project (render)

cmake_minimum_required(VERSION 2.8.11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})


if( NOT WIN32 )
        SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -w")
endif()


find_package(OpenGL)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

include_directories( ${GLUT_INCLUDE_DIR} ${GLEW_INCLUDE_DIR})
message("{GLUT_INCLUDE_DIR}=${GLUT_INCLUDE_DIR}")
message("{GLUT_LIBRARY}=${GLUT_LIBRARY}")
message("{GLEW_INCLUDE_DIR}=${GLEW_INCLUDE_DIR}")
message("{GLEW_LIBRARY}=${GLEW_LIBRARY}")
file(GLOB HDRS
	"*.h"
)

file(GLOB SRCS
	"*.cpp"
)

add_executable(PointRender ${SRCS} ${HDRS})

target_link_libraries(PointRender ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARY})