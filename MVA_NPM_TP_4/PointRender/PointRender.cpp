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
#include "matrixmath.h"
#include "kd_tree_3d.h"
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
static int lastX = 0, lastY = 0, lastZoom = 0;
static unsigned int FPS = 0;
static bool fullScreen = false;

// Point cloud 
static PointCloud pointCloud;
static float pointSize = 1.f; // screen splat size
static std::vector<float> point_data;

//VBO, VAO, UBO
GLuint vao, vbo, ubo;

// Light and material environment
struct Lighting {
    Vec3f ambientLightColor;
    float ambientLightCoeff;
    Vec3f lightPos[3];
    Vec3f lightColor[3];

    Vec3f matDiffuseColor;
    float matDiffuseCoeff;

    Vec3f matSpecularColor;
    float matSpecularShininess;
    float matSpecularCoeff;
};

static Lighting lighting;
static float alpha;

//shader
GLuint vertexShader, fragmentShader;
GLuint vertexShaderObject, fragmentShaderObject;
GLenum gl_program;

const GLchar* vShaderSrc[] = {
R"(
#version 330

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal_in;
layout(location = 2) in float knn_distance;
uniform vec3 eye;
uniform mat4 projection_modelview;

out vec3 normal;
out vec3 p;

void main()
{
    gl_Position = projection_modelview * vec4(pos,1.0);
    gl_PointSize = 500.0 * knn_distance / length(pos-eye);
    normal = normal_in;
    p = pos;
}
)"
};

const GLchar* fShaderSrc[] = {
R"(
#version 330
in vec3 normal;
in vec3 p;
out vec4 fragColor;

layout(std140) uniform LightingBlock {
    vec3 ambientLightColor;
    float ambientLightCoeff;
    vec3 lightPos[3];
    vec3 lightColor[3];

    vec3 matDiffuseColor;
    float matDiffuseCoeff;

    vec3 matSpecularColor;
    float matSpecularShininess;
    float matSpecularCoeff;
} lighting;

uniform vec3 eye;

void main()
{
    vec3 ambient = lighting.ambientLightColor*lighting.ambientLightCoeff;

    vec3 diffuse = vec3(0., 0., 0.);
    vec3 specular = vec3(0., 0., 0.);
    vec3 N = normalize(normal);
    for (int i = 0; i < 3; i++) {
        vec3 L = normalize(lighting.lightPos[i] - p);
        float lambertian = max(dot(L, N), 0.0);
        diffuse += lambertian * lighting.lightColor[i];

        if (lambertian > 0.) {
            vec3 V = normalize(eye - p);
            vec3 H = normalize(V + L);
            vec3 R = 2.0 * N * dot(N, L) - L;


            //// blinn-phong
            specular += pow(max(dot(N, H), 0.0), lighting.matSpecularShininess) * lighting.lightColor[i];
            // phong
            //specular += pow(max(dot(R, V), 0.0), matSpecularShininess / 4.0) * lightColor[i];
        }
    }

    vec3 rgb =
        ambient * lighting.matDiffuseCoeff * lighting.matDiffuseColor +
        diffuse * lighting.matDiffuseCoeff * lighting.matDiffuseColor +
        specular * lighting.matSpecularCoeff * lighting.matSpecularColor;

    fragColor = vec4(rgb, 1.0);
}
)"
};

enum class DrawingMode { Manuel, OpenGL_Fixed_Pipeline, OpenGL_Modern, NUM_MODE };
DrawingMode drawingMode;
// -------------------------------------------
// App Code.
// -------------------------------------------

void printUsage() {
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
        << " q, <esc>: Quit" << endl
        << " w/s : Increase/Decrease ambientLightCoeff " << endl
        << " d/a : Increase/Decrease matDiffuseCoeff " << endl
        << " x/z : Increase/Decrease matSpecularCoeff " << endl
        << " r/e : Increase/Decrease matSpecularShininess " << endl
        << " g : Switch between modes : Manuel, OpenGL_Fixed_Pipeline, OpenGL_Modern" << endl
        << endl;
}

void usage() {
    printUsage();
    exit(EXIT_FAILURE);
}

void compile_shader() {
    GLint compiled, linked;
    vertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
    fragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

    //vertex shader
    glShaderSource(vertexShaderObject, 1, vShaderSrc, NULL);
    glCompileShader(vertexShaderObject);
    glGetShaderiv(vertexShaderObject, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        GLint length;
        GLchar* log;
        glGetShaderiv(vertexShaderObject, GL_INFO_LOG_LENGTH,
            &length);
        log = (GLchar*)malloc(length);
        glGetShaderInfoLog(vertexShaderObject, length, &length, log);
        fprintf(stderr, "compile log = '%s'\n", log);
        free(log);
    }
    //frag shader
    glShaderSource(fragmentShaderObject, 1, fShaderSrc, NULL);
    glCompileShader(fragmentShaderObject);
    glGetShaderiv(fragmentShaderObject, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        GLint length;
        GLchar* log;
        glGetShaderiv(fragmentShaderObject, GL_INFO_LOG_LENGTH,
            &length);
        log = (GLchar*)malloc(length);
        glGetShaderInfoLog(fragmentShaderObject, length, &length, log);
        fprintf(stderr, "compile log = '%s'\n", log);
        free(log);
    }
    // gl program
    gl_program = glCreateProgram();
    glAttachShader(gl_program, vertexShaderObject);
    glAttachShader(gl_program, fragmentShaderObject);
    glLinkProgram(gl_program);
    glGetProgramiv(gl_program, GL_LINK_STATUS, &linked);
    if (!linked) {
        GLint length;
        GLchar* log;
        glGetProgramiv(gl_program, GL_INFO_LOG_LENGTH, &length);
        log = (GLchar*)malloc(length);
        glGetProgramInfoLog(gl_program, length, &length, log);
        fprintf(stderr, "link log = '%s'\n", log);
        free(log);
    }
}

void update_uniforms() {
    static const GLchar* uniformNames[9] = {
        "LightingBlock.ambientLightColor",       //0
        "LightingBlock.ambientLightCoeff",       //1
        "LightingBlock.lightPos",                //2
        "LightingBlock.lightColor",              //3
        "LightingBlock.matDiffuseColor",         //4
        "LightingBlock.matDiffuseCoeff",         //5
        "LightingBlock.matSpecularColor",        //6
        "LightingBlock.matSpecularShininess",    //7
        "LightingBlock.matSpecularCoeff",        //8
    };

    GLuint uniformIndices[9];

    glGetUniformIndices(gl_program, 9, uniformNames, uniformIndices);

    GLint uniformOffsets[9];
    GLint arrayStrides[9];

    glGetActiveUniformsiv(gl_program, 9, uniformIndices, GL_UNIFORM_OFFSET, uniformOffsets);
    glGetActiveUniformsiv(gl_program, 9, uniformIndices, GL_UNIFORM_ARRAY_STRIDE, arrayStrides);
    const int size = 4096;
    unsigned char* buffer = (unsigned char *)malloc(size);
    *(Vec3f *)(buffer + uniformOffsets[0]) = lighting.ambientLightColor;
    *(float*)(buffer + uniformOffsets[1]) = lighting.ambientLightCoeff;

    for (int i = 0; i < 3; i++) {
        *(Vec3f*)(buffer + uniformOffsets[2] + arrayStrides[2] * i) = lighting.lightPos[i];
        *(Vec3f*)(buffer + uniformOffsets[3] + arrayStrides[3] * i) = lighting.lightColor[i];
    }


    *(Vec3f *)(buffer + uniformOffsets[4]) = lighting.matDiffuseColor;
    *(float*)(buffer + uniformOffsets[5]) = lighting.matDiffuseCoeff;
    *(Vec3f *)(buffer + uniformOffsets[6]) = lighting.matSpecularColor;
    *(float*)(buffer + uniformOffsets[7]) = lighting.matSpecularShininess;
    *(float*)(buffer + uniformOffsets[8]) = lighting.matSpecularCoeff;

    glGenBuffers(1, &ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, size, buffer, GL_DYNAMIC_DRAW);

    GLuint blockIndex = glGetUniformBlockIndex(gl_program, "LightingBlock");
    GLuint uniformBlockBinding = 0;
    glUniformBlockBinding(gl_program, blockIndex, uniformBlockBinding);
    glBindBufferBase(GL_UNIFORM_BUFFER, uniformBlockBinding, ubo);

    free(buffer);

}

void init(const string & modelFilename) {
    camera.resize(SCREENWIDTH, SCREENHEIGHT);
    pointCloud.loadPN(modelFilename);
    lighting.ambientLightColor = Vec3f(0.6f, 0.6f, 0.6f);
    lighting.ambientLightCoeff = 0.3f;

    lighting.matDiffuseColor = Vec3f(.8f, .8f, .8f);
    lighting.matDiffuseCoeff = .5f;

    lighting.matSpecularColor = Vec3f(.8f, .8f, .8f);
    lighting.matSpecularCoeff = .7f;
    lighting.matSpecularShininess = 32.0f;

    drawingMode = DrawingMode::Manuel;


    lighting.lightPos[0] = Vec3f(2.f, -2.f, -2.f);
    lighting.lightColor[0] = Vec3f(0.8f, 0.0f, 0.f);
    lighting.lightPos[1] = Vec3f(-2.f, -2.f, 0.f);
    lighting.lightColor[1] = Vec3f(0.f, 0.7f, 0.f);
    lighting.lightPos[2] = Vec3f(-2.f, 2.f, 2.f);
    lighting.lightColor[2] = Vec3f(0.f, 0.5f, 1.f);
    alpha = 0.3;
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_POINT_SMOOTH);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    std::vector<float> average_dists(pointCloud.size());
    const int k = 15;
    {//calculate knn distance
        std::vector<Vec3f> points;
        for (unsigned int i = 0; i < pointCloud.size(); i++) 
            points.push_back(pointCloud(i).position());
        KdTree3D tree;
        tree.set_points(points);
        for (unsigned int i = 0; i < pointCloud.size(); i++){
            auto p = pointCloud(i).position();
            std::vector<size_t> indices = tree.query(p, k);
            float sum_d = 0;
            for (auto id : indices) {
                auto p1 = pointCloud(id).position();
                sum_d += (p1 - p).length();
            }
            average_dists[i] = sum_d / indices.size();
        }
    }

    point_data.clear();
    for (unsigned int i = 0; i < pointCloud.size(); i++) {
        const PointCloud::Point & point = pointCloud(i);
        const Vec3f & p = point.position();
        const Vec3f & n = point.normal();
        point_data.push_back(p[0]);
        point_data.push_back(p[1]);
        point_data.push_back(p[2]);
        point_data.push_back(n[0]);
        point_data.push_back(n[1]);
        point_data.push_back(n[2]);
        point_data.push_back(average_dists[i]);
    }



    {//VAO, VBO
        glCreateBuffers(1, &vbo);
        glNamedBufferStorage(vbo, sizeof(float)*point_data.size(), &point_data[0], 0);
        glCreateVertexArrays(1, &vao);

        GLuint bindingindex = 0;
        GLintptr offset = 0;
        GLsizei stride = 7 * sizeof(float);
        glVertexArrayVertexBuffer(vao, bindingindex, vbo, offset, stride);

        {//bind vertex position data
            GLuint attrib = 0;
            GLuint relativeoffset = 0;
            GLuint components = 3;
            glVertexArrayAttribBinding(vao, attrib, bindingindex);
            glVertexArrayAttribFormat(vao, attrib, components, GL_FLOAT, GL_FALSE, relativeoffset);
            glEnableVertexArrayAttrib(vao, attrib);
        }
        {//bind vertex normal data
            GLuint attrib = 1;
            GLuint relativeoffset = 3 * sizeof(float);
            GLuint components = 3;
            glVertexArrayAttribBinding(vao, attrib, bindingindex);
            glVertexArrayAttribFormat(vao, attrib, components, GL_FLOAT, GL_FALSE, relativeoffset);
            glEnableVertexArrayAttrib(vao, attrib);
        }
        {//bind averaged distance
            GLuint attrib = 2;
            GLuint relativeoffset = 6 * sizeof(float);
            GLuint components = 1;
            glVertexArrayAttribBinding(vao, attrib, bindingindex);
            glVertexArrayAttribFormat(vao, attrib, components, GL_FLOAT, GL_FALSE, relativeoffset);
            glEnableVertexArrayAttrib(vao, attrib);
        }
    }

    //shader
    compile_shader();

    //uniform
    update_uniforms();
}

void render() {
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    camera.apply();
    Vec3f eye;
    camera.getPos(eye[0], eye[1], eye[2]);

    switch (drawingMode) {
    case DrawingMode::Manuel: {
        glPointSize(pointSize);
        glBegin(GL_POINTS);

        for (unsigned int i = 0; i < pointCloud.size(); i++) {
            const PointCloud::Point & point = pointCloud(i);
            const Vec3f & p = point.position();
            const Vec3f & n = point.normal();
            //Vec3f rgb (1.f, 1.f, 1.f); // Color response of the point sample
            // ------ A REMPLIR -----------
            //float y = p[1];
            //y = std::max(std::min(y, 1.f), -1.f);
            //y = (y + 1.f) / 2.f;
            //Vec3f color_object = Vec3f(y, 0.f, 1.f - y);

            Vec3f ambient = lighting.ambientLightColor*lighting.ambientLightCoeff;

            Vec3f diffuse(0.f, 0.f, 0.f);
            Vec3f specular(0.f, 0.f, 0.f);
            Vec3f N = normalize(n);
            for (int i = 0; i < 3; i++) {
                Vec3f L = normalize(lighting.lightPos[i] - p);
                float lambertian = std::max(dot(L, N), 0.0f);
                diffuse += lambertian * lighting.lightColor[i];

                if (lambertian > 0.f) {
                    Vec3f V = normalize(eye - p);
                    Vec3f H = normalize(V + L);
                    Vec3f R = 2.0f * N * dot(N, L) - L;


                    //// blinn-phong
                    specular += std::pow(std::max(dot(N, H), 0.0f), lighting.matSpecularShininess) * lighting.lightColor[i];
                    // phong
                    //specular += std::pow(std::max(dot(R, V), 0.0f), matSpecularShininess / 4.0f) * lightColor[i];
                }
            }

            Vec3f rgb =
                ambient * lighting.matDiffuseCoeff * lighting.matDiffuseColor +
                diffuse * lighting.matDiffuseCoeff * lighting.matDiffuseColor +
                specular * lighting.matSpecularCoeff * lighting.matSpecularColor;


            // ----------------------------
            glColor3f(rgb[0], rgb[1], rgb[2]);
            glVertex3f(p[0], p[1], p[2]);
        }
        glEnd();
        break;
    }
    case DrawingMode::OpenGL_Fixed_Pipeline: {
        glEnable(GL_LIGHTING);
        glDisable(GL_COLOR_MATERIAL);
        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE); // realistic specular light

        Vec3f lightAmbient = lighting.ambientLightColor*lighting.ambientLightCoeff;
        GLfloat ambient[] = { lightAmbient[0], lightAmbient[1], lightAmbient[2], 1.0 };
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient);

        GLenum lights[] = { GL_LIGHT0, GL_LIGHT1, GL_LIGHT2 };
        for (int i = 0; i < 3; i++) {
            GLfloat color[] = { lighting.lightColor[i][0], lighting.lightColor[i][1], lighting.lightColor[i][2], 1.0 };
            glLightfv(lights[i], GL_DIFFUSE, color);
            glLightfv(lights[i], GL_SPECULAR, color);
            GLfloat pos[] = { lighting.lightPos[i][0], lighting.lightPos[i][1], lighting.lightPos[i][2], 1.0 };
            glLightfv(lights[i], GL_POSITION, pos);
            glEnable(lights[i]);
        }

        Vec3f matDiffuse = lighting.matDiffuseColor*lighting.matDiffuseCoeff;
        GLfloat matDiffuseRGBA[] = { matDiffuse[0], matDiffuse[1], matDiffuse[2], 1.0 };
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, matDiffuseRGBA);
        Vec3f matSpecular = lighting.matSpecularColor*lighting.matSpecularCoeff;
        GLfloat matSpecularRGBA[] = { matSpecular[0], matSpecular[1], matSpecular[2], 1.0 };
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, matSpecularRGBA);
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, &lighting.matSpecularShininess);

        glPointSize(pointSize);
        glBegin(GL_POINTS);
        for (unsigned int i = 0; i < pointCloud.size(); i++) {
            const PointCloud::Point & point = pointCloud(i);
            const Vec3f & p = point.position();
            const Vec3f & n = point.normal();
            glNormal3f(n[0], n[1], n[2]);
            glVertex3f(p[0], p[1], p[2]);
        }
        glEnd();

        glDisable(GL_LIGHTING);
        break;
    }
    case DrawingMode::OpenGL_Modern: {
        {//projection model view matrix
            GLfloat trans[4][4];
            math::translation_matrix(trans, -camera.x, -camera.y, -camera.z - camera._zoom);
            GLfloat rot[4][4];
            build_rotmatrix(rot, camera.curquat);
            math::transpose_in_place(rot);
            GLfloat modelview[4][4];
            math::multiply_matrix(modelview, trans, rot);

            GLfloat proj[4][4];
            math::matrix_gl_perspectiveGL(proj, camera.fovAngle, camera.aspectRatio, camera.nearPlane, camera.farPlane);


            GLfloat projection_modelview[4][4];
            math::multiply_matrix(projection_modelview, proj, modelview);
            GLint pmvMatrix = glGetUniformLocation(gl_program, "projection_modelview");
            glUniformMatrix4fv(pmvMatrix, 1, GL_TRUE, &projection_modelview[0][0]);
        }


        GLint eye_loc = glGetUniformLocation(gl_program, "eye");
        glUniform3fv(eye_loc, 1, &eye[0]);



        //draw
        glPointSize(pointSize);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, pointCloud.size());
        glBindVertexArray(0);

        break;
    }
    }
    glutSwapBuffers();
}

void idle() {
    static float lastTime = glutGet((GLenum)GLUT_ELAPSED_TIME);
    static unsigned int counter = 0;
    counter++;
    float currentTime = glutGet((GLenum)GLUT_ELAPSED_TIME);
    if (currentTime - lastTime >= 1000.0f) {
        FPS = counter;
        counter = 0;
        static char winTitle[64];
        sprintf(winTitle, "PointRender - Num. Of Points: %d - FPS: %d", pointCloud.size(), FPS);
        glutSetWindowTitle(winTitle);
        lastTime = currentTime;
    }
    glutPostRedisplay();
}

void key(unsigned char keyPressed, int x, int y) {
    switch (keyPressed) {
    case 'f':
        if (fullScreen) {
            glutReshapeWindow(SCREENWIDTH, SCREENHEIGHT);
            fullScreen = false;
        }
        else {
            glutFullScreen();
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
        if (alpha < 1.0)
            alpha += 0.1f;
        break;
    case '/':
        if (alpha > 0.0f)
            alpha -= 0.1f;
        break;
    case 'w':
        lighting.ambientLightCoeff += 0.1;
        std::cout << "ambientLightCoeff = " << lighting.ambientLightCoeff << std::endl;
        break;
    case 's':
        lighting.ambientLightCoeff -= 0.1;
        std::cout << "ambientLightCoeff = " << lighting.ambientLightCoeff << std::endl;
        break;
    case 'd':
        lighting.matDiffuseCoeff += 0.1;
        std::cout << "matDiffuseCoeff = " << lighting.matDiffuseCoeff << std::endl;
        break;
    case 'a':
        lighting.matDiffuseCoeff -= 0.1;
        std::cout << "matDiffuseCoeff = " << lighting.matDiffuseCoeff << std::endl;
        break;
    case 'x':
        lighting.matSpecularCoeff += 0.1;
        std::cout << "matSpecularCoeff = " << lighting.matSpecularCoeff << std::endl;
        break;
    case 'z':
        lighting.matSpecularCoeff -= 0.1;
        std::cout << "matSpecularCoeff = " << lighting.matSpecularCoeff << std::endl;
        break;
    case 'r':
        lighting.matSpecularShininess *= 1.2;
        std::cout << "matSpecularShininess = " << lighting.matSpecularShininess << std::endl;
        break;
    case 'e':
        lighting.matSpecularShininess /= 1.2;
        std::cout << "matSpecularShininess = " << lighting.matSpecularShininess << std::endl;
        break;
    case 'g':
        drawingMode = DrawingMode(((int)drawingMode + 1) % (int)DrawingMode::NUM_MODE);
        switch (drawingMode) {
        case DrawingMode::Manuel:
            std::cout << "Manuel lighting" << std::endl;
            break;
        case DrawingMode::OpenGL_Fixed_Pipeline:
            std::cout << "OpenGL" << std::endl;
            break;
        case DrawingMode::OpenGL_Modern:
            std::cout << "OpenGL_Shader" << std::endl;
            break;
        }
        if (drawingMode == DrawingMode::OpenGL_Modern) {
            glUseProgram(gl_program);
            glEnable(GL_PROGRAM_POINT_SIZE);
        }
        else {
            glUseProgram(0);
            glDisable(GL_PROGRAM_POINT_SIZE);
        }
        break;
    case 'q':
    case 27:
        exit(0);
        break;
    default:
        printUsage();
        break;
    }
    idle();
}

void mouse(int button, int state, int x, int y) {
    if (state == GLUT_UP) {
        mouseMovePressed = false;
        mouseRotatePressed = false;
        mouseZoomPressed = false;
    }
    else {
        if (button == GLUT_LEFT_BUTTON) {
            camera.beginRotate(x, y);
            mouseMovePressed = false;
            mouseRotatePressed = true;
            mouseZoomPressed = false;
        }
        else if (button == GLUT_RIGHT_BUTTON) {
            lastX = x;
            lastY = y;
            mouseMovePressed = true;
            mouseRotatePressed = false;
            mouseZoomPressed = false;
        }
        else if (button == GLUT_MIDDLE_BUTTON) {
            if (mouseZoomPressed == false) {
                lastZoom = y;
                mouseMovePressed = false;
                mouseRotatePressed = false;
                mouseZoomPressed = true;
            }
        }
    }
    idle();
}

void motion(int x, int y) {
    if (mouseRotatePressed == true) {
        camera.rotate(x, y);
    }
    else if (mouseMovePressed == true) {
        camera.move((x - lastX) / static_cast<float>(SCREENWIDTH), (lastY - y) / static_cast<float>(SCREENHEIGHT), 0.0);
        lastX = x;
        lastY = y;
    }
    else if (mouseZoomPressed == true) {
        camera.zoom(float(y - lastZoom) / SCREENHEIGHT);
        lastZoom = y;
    }
}

void reshape(int w, int h) {
    camera.resize(w, h);
}

int main(int argc, char ** argv) {
    if (argc > 2) {
        printUsage();
        exit(EXIT_FAILURE);
    }


    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(SCREENWIDTH, SCREENHEIGHT);
    window = glutCreateWindow("PointRender");


    GLenum err = glewInit();
    if (GLEW_OK != err) {
        /* Problem: glewInit failed, something is seriously wrong. */
        std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
    }
    std::cout << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;

    char* s = (char*)glGetString(GL_VERSION);
    std::cout << "OpenGL version = " << s << std::endl;

    try {
        init(argc == 2 ? string(argv[1]) : string("data/face.pn"));
    }
    catch (const Exception & e) {
        cerr << "Error at initialization: " << e.message() << endl;
        exit(1);
    };
    glutIdleFunc(idle);
    glutDisplayFunc(render);
    glutKeyboardFunc(key);
    glutReshapeFunc(reshape);
    glutMotionFunc(motion);
    glutMouseFunc(mouse);
    key('?', 0, 0);
    glutMainLoop();
    return EXIT_SUCCESS;
}

