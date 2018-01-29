// -------------------------------------------
// PointRender: a basic point cloud rendering
// app in C++ and OpenGL.
// Copyright (C) 2015 Tamy Boubekeur
// All rights reserved.
// -------------------------------------------

// -------------------------------------------
// Disclaimer: this code is meant to be simple
// to read and edit and does not aim at 
// providing a high performance rendering. 
// Use vertex buffers and shaders (OpenGL 3.3) 
// for faster rendering.
// -------------------------------------------

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "include_opengl.h"

#include "Exception.h"
#include "Vec3.h"
#include "PointCloud.h"
#include "Camera.h"

using namespace std;

// -------------------------------------------
// App global variables.
// -------------------------------------------

static GLint window;
static unsigned int SCREENWIDTH = 800;
static unsigned int SCREENHEIGHT = 600;
static Camera camera;
static bool mouseRotatePressed = false;
static bool mouseMovePressed = false;
static bool mouseZoomPressed = false;
static int lastX=0, lastY=0, lastZoom=0;
static unsigned int FPS = 0;
static bool fullScreen = false;

// Point cloud 
static PointCloud pointCloud;
static float pointSize = 1.f; // screen splat size

// Light and material environment
static Vec3f ambientLightColor;
static float ambientLightCoeff;
static Vec3f lightPos[3];
static Vec3f lightColor[3];

static Vec3f matDiffuseColor;
static float matDiffuseCoeff;

static Vec3f matSpecularColor;
static float matSpecularShininess;
static float matSpecularCoeff;

static float alpha;

// -------------------------------------------
// App Code.
// -------------------------------------------

void printUsage () {
    cerr << endl 
         << "PointRender: a basic point cloud rendering app." << endl
         << "Author : Tamy Boubekeur (http://www.telecom-paristech.fr/~boubek)" << endl << endl
         << "Usage : ./PointRender [<file.pn>]" << endl
         << "PN format : binary file made of a list of 6 floats chunk:" << endl
         << "   x0,y0,z0,nx0,ny0,nz0, x1,y1,z1,nx1,ny1,nz1,..." << endl
         << "with {x,y,z} the position and {nx,ny,nz} the normal vector of each point sample" << endl
         << "Keyboard commands" << endl 
         << "------------------" << endl
         << " ?: Print help" << endl 
         << " f: Toggle full screen mode" << endl
         << " +/-: Increase/Decrease screen point size" << endl
         << " <drag>+<left button>: rotate model" << endl 
         << " <drag>+<right button>: move model" << endl
         << " <drag>+<middle button>: zoom" << endl
         << " q, <esc>: Quit" << endl << endl; 
}

void usage () {
    printUsage ();
    exit (EXIT_FAILURE);
}

void init (const string & modelFilename) {
    camera.resize (SCREENWIDTH, SCREENHEIGHT);
    pointCloud.loadPN (modelFilename);
    ambientLightColor = Vec3f (0.0f, 0.1f, 0.4f);
    ambientLightCoeff = 0.3f;

    matDiffuseColor = Vec3f(.0f, .8f, .0f);
    matDiffuseCoeff = .3f;

    matSpecularColor = Vec3f(.8f, .8f, .8f);
    matSpecularCoeff = .7f;
    matSpecularShininess = 8.0f;


    lightPos[0] = Vec3f (2.f, -2.f, -2.f);
    lightColor[0] = Vec3f (0.8f, 0.5f, 0.f);
    lightPos[1] = Vec3f (-2.f, -2.f, 0.f);
    lightColor[1] = Vec3f (0.2f, 0.7f, 0.2f);
    lightPos[2] = Vec3f (-2.f, 2.f, 2.f);
    lightColor[2] = Vec3f (0.1f, 0.5f, 0.8f);
    alpha = 0.3;
    glCullFace (GL_BACK);
    glEnable (GL_CULL_FACE);
    glDepthFunc (GL_LESS);
    glEnable (GL_DEPTH_TEST);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glEnable (GL_POINT_SMOOTH);
    glClearColor (0.0f, 0.0f, 0.0f, 1.0f);
}

void render () {
    glLoadIdentity ();
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    camera.apply ();
    glPointSize (pointSize);
    glBegin (GL_POINTS);
    Vec3f eye;
    camera.getPos (eye[0], eye[1], eye[2]);
    for (unsigned int i = 0; i < pointCloud.size (); i++)  {
        const PointCloud::Point & point = pointCloud (i);
        const Vec3f & p = point.position ();
        const Vec3f & n = point.normal ();
        //Vec3f rgb (1.f, 1.f, 1.f); // Color response of the point sample
        // ------ A REMPLIR -----------
        float y = p[1];
        y = std::max(std::min(y, 1.f), -1.f);
        y = (y + 1.f) / 2.f;
        Vec3f color_object = Vec3f(y, 0.f, 1.f-y);

        Vec3f ambient = ambientLightColor*ambientLightCoeff;

        Vec3f diffuse(0.f, 0.f, 0.f);
        Vec3f specular(0.f, 0.f, 0.f);
        Vec3f N = normalize(n);
        for (int i = 0; i < 3; i++) {
            Vec3f L = normalize(lightPos[i] - p);
            diffuse += dot(L, N) * matDiffuseColor * lightColor[i];
            Vec3f V = normalize(eye - p);
            Vec3f H = normalize(V + L);
            specular += std::pow(dot(N, H), matSpecularShininess) *  matSpecularColor * lightColor[i];
        }

        Vec3f rgb = ambient*color_object + diffuse + specular;

        
        // ----------------------------
        glColor3f (rgb[0], rgb[1], rgb[2]);
        glVertex3f (p[0], p[1], p[2]);	
    }
    glEnd ();
    glutSwapBuffers ();
}

void idle () {
    static float lastTime = glutGet ((GLenum)GLUT_ELAPSED_TIME);
    static unsigned int counter = 0;
    counter++;
    float currentTime = glutGet ((GLenum)GLUT_ELAPSED_TIME);
    if (currentTime - lastTime >= 1000.0f) {
        FPS = counter;
        counter = 0;
        static char winTitle [64];
        sprintf (winTitle, "PointRender - Num. Of Points: %d - FPS: %d", pointCloud.size (), FPS);
        glutSetWindowTitle (winTitle);
        lastTime = currentTime;
    }
    glutPostRedisplay ();
}

void key (unsigned char keyPressed, int x, int y) {
    switch (keyPressed) {
    case 'f':
        if (fullScreen == true) {
            glutReshapeWindow (SCREENWIDTH, SCREENHEIGHT);
            fullScreen = false;
        } else {
            glutFullScreen ();
            fullScreen = true;
        }      
        break;
    case '+':
        pointSize += 1.0f;
        break;
    case '-':
        if (pointSize > 1.0f)
            pointSize -= 1.0f;
        break;
    case '*':
        if (alpha  < 1.0)
            alpha += 0.1f;
        break;
    case '/':
        if (alpha > 0.0f)
            alpha -= 0.1f;
        break;
    case 'q':
    case 27:
        exit (0);
        break;
    default:
        printUsage ();
        break;
    }
    idle ();
}

void mouse (int button, int state, int x, int y) {
    if (state == GLUT_UP) {
        mouseMovePressed = false;
        mouseRotatePressed = false;
        mouseZoomPressed = false;
    } else {
        if (button == GLUT_LEFT_BUTTON) {
            camera.beginRotate (x, y);
            mouseMovePressed = false;
            mouseRotatePressed = true;
            mouseZoomPressed = false;
        } else if (button == GLUT_RIGHT_BUTTON) {
            lastX = x;
            lastY = y;
            mouseMovePressed = true;
            mouseRotatePressed = false;
            mouseZoomPressed = false;
        } else if (button == GLUT_MIDDLE_BUTTON) {
            if (mouseZoomPressed == false) {
                lastZoom = y;
                mouseMovePressed = false;
                mouseRotatePressed = false;
                mouseZoomPressed = true;
            }
        }
    }
    idle ();
}

void motion (int x, int y) {
    if (mouseRotatePressed == true) {
        camera.rotate (x, y);
    }
    else if (mouseMovePressed == true) {
        camera.move ((x-lastX)/static_cast<float>(SCREENWIDTH), (lastY-y)/static_cast<float>(SCREENHEIGHT), 0.0);
        lastX = x;
        lastY = y;
    }
    else if (mouseZoomPressed == true) {
        camera.zoom (float (y-lastZoom)/SCREENHEIGHT);
        lastZoom = y;
    }
}

void reshape(int w, int h) {
    camera.resize (w, h);
}

int main (int argc, char ** argv) {
    if (argc > 2) {
        printUsage ();
        exit (EXIT_FAILURE);
    }
    glutInit (&argc, argv);
    glutInitDisplayMode (GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize (SCREENWIDTH, SCREENHEIGHT);
    window = glutCreateWindow ("PointRender");
    //try {
        init (argc == 2 ? string (argv[1]) : string ("data/face.pn"));
    //} catch (const Exception & e) {
    //    cerr << "Error at initialization: " << e.message () << endl;
    //    exit (1);
    //};
    glutIdleFunc (idle);
    glutDisplayFunc (render);
    glutKeyboardFunc (key);
    glutReshapeFunc (reshape);
    glutMotionFunc (motion);
    glutMouseFunc (mouse);
    key ('?', 0, 0);   
    glutMainLoop ();
    return EXIT_SUCCESS;
}

