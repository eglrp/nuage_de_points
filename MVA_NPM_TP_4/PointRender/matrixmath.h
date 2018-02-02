#pragma once
#include <algorithm>
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
}