#pragma once
#ifdef __APPLE__
    #include <OpenGL/gl.h>
    #include <OpenGL/glut.h>
#else
    #ifdef _WIN32
        #define NOMINMAX
        #include <windows.h>
        #undef NOMINMAX
    #endif

    #include <GL/glew.h>
    #include <GL/gl.h>
    #include <GL/glut.h>
    //#include <GL/freeglut.h>
#endif