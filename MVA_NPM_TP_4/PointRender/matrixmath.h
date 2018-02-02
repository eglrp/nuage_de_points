#pragma once
#include <algorithm>
#include <cmath>
namespace math {
    /* c=a*b */
    void multiply_matrix(float c[4][4], float a[4][4], float b[4][4]) {
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                float sumElements = 0.0f;
                for (int k = 0; k < 4; ++k)
                {
                    sumElements += a[i][k] * b[k][j];
                }
                c[i][j] = sumElements;
            }
        }
    }

    void translation_matrix(float a[4][4], float x, float y, float z) {
        float* s = &a[0][0];
        std::fill(s, s + 16, 0.0f);
        for (int i = 0; i < 4; i++)
            a[i][i] = 1.0f;
        a[0][3] = x;
        a[1][3] = y;
        a[2][3] = z;
    }

    /*a=a.T*/
    void transpose_in_place(float a[4][4]) {
        for (int i = 0; i < 3; ++i)
            for (int j = i + 1; j < 4; ++j)
                std::swap(a[i][j], a[j][i]);
    }

    void matrix_glFrustum(float a[4][4],
        float left,
        float right,
        float bottom,
        float top,
        float zNear,
        float zFar
    ) {
        std::fill(&a[0][0], &a[0][0] + 16, 0.0f);
        a[0][0] = 2 * zNear / (right - left);
        a[1][1] = 2 * zNear / (top - bottom);
        float A = (right + left) / (right - left);
        float B = (top + bottom) / (top - bottom);
        float C = -(zFar + zNear) / (zFar - zNear);
        float D = -2.f*zFar*zNear / (zFar - zNear);
        a[0][2] = A;
        a[1][2] = B;
        a[2][2] = C;
        a[3][2] = -1.f;
        a[2][3] = D;
    }

    void matrix_gl_perspectiveGL(float a[4][4], float fovY, float aspect, float zNear, float zFar)
    {
        // http://nehe.gamedev.net/article/replacement_for_gluperspective/21002/
        const float pi = 3.1415926535898;
        float fW, fH;
        fH = std::tan(fovY / 360 * pi) * zNear;
        fW = fH * aspect;
        matrix_glFrustum(a, -fW, fW, -fH, fH, zNear, zFar);
    }
}