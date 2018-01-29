project (render)

cmake_minimum_required(VERSION 2.8.11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})


if( NOT WIN32 )
        SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -w")
endif()


find_package(OpenGL)
find_package(GLUT REQUIRED)

include_directories( ${GLUT_INCLUDE_DIR} )
message("{GLUT_INCLUDE_DIR}=${GLUT_INCLUDE_DIR}")
file(GLOB HDRS
	"*.h"
)

file(GLOB SRCS
	"*.cpp"
)

add_executable(PointRender ${SRCS} ${MOCS})

target_link_libraries(PointRender ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )