#include "Matrix4.h"
#include "Vector3.h"
#include <cmath>

namespace MyEngine {

float Matrix4::determinant() const {
    float m0 = m[0][0], m1 = m[0][1], m2 = m[0][2], m3 = m[0][3];
    float m4 = m[1][0], m5 = m[1][1], m6 = m[1][2], m7 = m[1][3];
    float m8 = m[2][0], m9 = m[2][1], m10 = m[2][2], m11 = m[2][3];
    float m12 = m[3][0], m13 = m[3][1], m14 = m[3][2], m15 = m[3][3];

    return m0 * (m5 * (m10 * m15 - m11 * m14) - m6 * (m9 * m15 - m11 * m13) + m7 * (m9 * m14 - m10 * m13))
         - m1 * (m4 * (m10 * m15 - m11 * m14) - m6 * (m8 * m15 - m11 * m12) + m7 * (m8 * m14 - m10 * m12))
         + m2 * (m4 * (m9 * m15 - m11 * m13) - m5 * (m8 * m15 - m11 * m12) + m7 * (m8 * m13 - m9 * m12))
         - m3 * (m4 * (m9 * m14 - m10 * m13) - m5 * (m8 * m14 - m10 * m12) + m6 * (m8 * m13 - m9 * m12));
}

Matrix4 Matrix4::inverse() const {
    float m0 = m[0][0], m1 = m[0][1], m2 = m[0][2], m3 = m[0][3];
    float m4 = m[1][0], m5 = m[1][1], m6 = m[1][2], m7 = m[1][3];
    float m8 = m[2][0], m9 = m[2][1], m10 = m[2][2], m11 = m[2][3];
    float m12 = m[3][0], m13 = m[3][1], m14 = m[3][2], m15 = m[3][3];

    float b00 = m0 * m5 - m1 * m4;
    float b01 = m0 * m6 - m2 * m4;
    float b02 = m0 * m7 - m3 * m4;
    float b03 = m1 * m6 - m2 * m5;
    float b04 = m1 * m7 - m3 * m5;
    float b05 = m2 * m7 - m3 * m6;
    float b06 = m8 * m13 - m9 * m12;
    float b07 = m8 * m14 - m10 * m12;
    float b08 = m8 * m15 - m11 * m12;
    float b09 = m9 * m14 - m10 * m13;
    float b10 = m9 * m15 - m11 * m13;
    float b11 = m10 * m15 - m11 * m14;

    float det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;

    if (det == 0) return Matrix4::identity();

    float inv_det = 1.0f / det;

    Matrix4 r;
    r.m[0][0] = (m5 * b11 - m6 * b10 + m7 * b09) * inv_det;
    r.m[0][1] = (m2 * b10 - m1 * b11 - m3 * b09) * inv_det;
    r.m[0][2] = (m13 * b05 - m14 * b04 + m15 * b03) * inv_det;
    r.m[0][3] = (m10 * b04 - m9 * b05 - m11 * b03) * inv_det;
    r.m[1][0] = (m6 * b08 - m4 * b11 - m7 * b07) * inv_det;
    r.m[1][1] = (m0 * b11 - m2 * b08 + m3 * b07) * inv_det;
    r.m[1][2] = (m14 * b02 - m12 * b05 - m15 * b01) * inv_det;
    r.m[1][3] = (m9 * b05 - m8 * b02 - m11 * b01) * inv_det;
    r.m[2][0] = (m4 * b10 - m5 * b08 + m7 * b06) * inv_det;
    r.m[2][1] = (m1 * b08 - m0 * b10 - m3 * b06) * inv_det;
    r.m[2][2] = (m12 * b04 - m13 * b02 + m15 * b00) * inv_det;
    r.m[2][3] = (m8 * b02 - m9 * b00 - m11 * b00) * inv_det;
    r.m[3][0] = (m5 * b07 - m4 * b09 - m6 * b06) * inv_det;
    r.m[3][1] = (m0 * b09 - m1 * b07 + m2 * b06) * inv_det;
    r.m[3][2] = (m13 * b01 - m12 * b03 - m14 * b00) * inv_det;
    r.m[3][3] = (m8 * b03 - m9 * b01 + m10 * b00) * inv_det;

    return r;
}

Matrix4 Matrix4::transposed() const {
    Matrix4 r;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            r.m[i][j] = m[j][i];
        }
    }
    return r;
}

Vector3 Matrix4::get_scale() const {
    Vector3 x(m[0][0], m[1][0], m[2][0]);
    Vector3 y(m[0][1], m[1][1], m[2][1]);
    Vector3 z(m[0][2], m[1][2], m[2][2]);
    return Vector3(x.length(), y.length(), z.length());
}

} // namespace MyEngine
