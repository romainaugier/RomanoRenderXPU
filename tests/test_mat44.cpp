#include "romanorender/mat44.h"

#include "stdromano/logger.h"

#include <cmath>
#include <iostream>
#include <string>

// Simple test framework
#define TEST_EPSILON 1e-5f
#define ASSERT_TRUE(condition)                                                                     \
    if(!(condition))                                                                               \
    {                                                                                              \
        stdromano::log_error("TEST FAILED: {} - {}", __LINE__, #condition);                        \
        return false;                                                                              \
    }

#define ASSERT_FLOAT_EQ(a, b)                                                                      \
    if(maths::absf((a) - (b)) > TEST_EPSILON)                                                      \
    {                                                                                              \
        stdromano::log_error("TEST FAILED: {} - Expected {} to equal {}", __LINE__, a, b);         \
        return false;                                                                              \
    }

#define ASSERT_VEC3_EQ(v1, v2)                                                                     \
    ASSERT_FLOAT_EQ((v1).x, (v2).x);                                                               \
    ASSERT_FLOAT_EQ((v1).y, (v2).y);                                                               \
    ASSERT_FLOAT_EQ((v1).z, (v2).z);

#define RUN_TEST(test_func)                                                                        \
    {                                                                                              \
        stdromano::log_info("Running {}...", #test_func);                                          \
        if(test_func())                                                                            \
        {                                                                                          \
            stdromano::log_info("PASSED");                                                         \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            stdromano::log_info("FAILED");                                                         \
            failed_tests++;                                                                        \
        }                                                                                          \
        total_tests++;                                                                             \
    }

using namespace romanorender;

bool test_default_constructor()
{
    Mat44F mat;

    ASSERT_FLOAT_EQ(mat[0], 1.0f);
    ASSERT_FLOAT_EQ(mat[1], 0.0f);
    ASSERT_FLOAT_EQ(mat[2], 0.0f);
    ASSERT_FLOAT_EQ(mat[3], 0.0f);

    ASSERT_FLOAT_EQ(mat[4], 0.0f);
    ASSERT_FLOAT_EQ(mat[5], 1.0f);
    ASSERT_FLOAT_EQ(mat[6], 0.0f);
    ASSERT_FLOAT_EQ(mat[7], 0.0f);

    ASSERT_FLOAT_EQ(mat[8], 0.0f);
    ASSERT_FLOAT_EQ(mat[9], 0.0f);
    ASSERT_FLOAT_EQ(mat[10], 1.0f);
    ASSERT_FLOAT_EQ(mat[11], 0.0f);

    ASSERT_FLOAT_EQ(mat[12], 0.0f);
    ASSERT_FLOAT_EQ(mat[13], 0.0f);
    ASSERT_FLOAT_EQ(mat[14], 0.0f);
    ASSERT_FLOAT_EQ(mat[15], 1.0f);

    return true;
}

bool test_parameterized_constructor()
{
    Mat44F mat(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f);

    for(int i = 0; i < 16; i++)
    {
        ASSERT_FLOAT_EQ(mat[i], static_cast<float>(i + 1));
    }

    return true;
}

bool test_operators()
{
    Mat44F mat(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f);

    for(int i = 0; i < 16; i++)
    {
        ASSERT_FLOAT_EQ(mat[i], static_cast<float>(i + 1));
    }

    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            ASSERT_FLOAT_EQ(mat(i, j), static_cast<float>(i * 4 + j + 1));
        }
    }

    return true;
}

bool test_transpose()
{
    Mat44F mat(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f);

    Mat44F transposed = mat.transposed();

    ASSERT_FLOAT_EQ(transposed(0, 0), mat(0, 0));
    ASSERT_FLOAT_EQ(transposed(0, 1), mat(1, 0));
    ASSERT_FLOAT_EQ(transposed(0, 2), mat(2, 0));
    ASSERT_FLOAT_EQ(transposed(0, 3), mat(3, 0));

    ASSERT_FLOAT_EQ(transposed(1, 0), mat(0, 1));
    ASSERT_FLOAT_EQ(transposed(1, 1), mat(1, 1));
    ASSERT_FLOAT_EQ(transposed(1, 2), mat(2, 1));
    ASSERT_FLOAT_EQ(transposed(1, 3), mat(3, 1));

    ASSERT_FLOAT_EQ(transposed(2, 0), mat(0, 2));
    ASSERT_FLOAT_EQ(transposed(2, 1), mat(1, 2));
    ASSERT_FLOAT_EQ(transposed(2, 2), mat(2, 2));
    ASSERT_FLOAT_EQ(transposed(2, 3), mat(3, 2));

    ASSERT_FLOAT_EQ(transposed(3, 0), mat(0, 3));
    ASSERT_FLOAT_EQ(transposed(3, 1), mat(1, 3));
    ASSERT_FLOAT_EQ(transposed(3, 2), mat(2, 3));
    ASSERT_FLOAT_EQ(transposed(3, 3), mat(3, 3));

    Mat44F mat2 = mat;
    mat2.transpose();

    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            ASSERT_FLOAT_EQ(mat2(i, j), transposed(i, j));
        }
    }

    return true;
}

bool test_matrix_multiplication()
{
    Mat44F A(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f);

    Mat44F B(17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f);

    Mat44F C = mat44f_mul(A, B);

    ASSERT_FLOAT_EQ(C(0, 0), 250.0f);
    ASSERT_FLOAT_EQ(C(0, 1), 260.0f);
    ASSERT_FLOAT_EQ(C(0, 2), 270.0f);
    ASSERT_FLOAT_EQ(C(0, 3), 280.0f);

    ASSERT_FLOAT_EQ(C(1, 0), 618.0f);
    ASSERT_FLOAT_EQ(C(1, 1), 644.0f);
    ASSERT_FLOAT_EQ(C(1, 2), 670.0f);
    ASSERT_FLOAT_EQ(C(1, 3), 696.0f);

    ASSERT_FLOAT_EQ(C(2, 0), 986.0f);
    ASSERT_FLOAT_EQ(C(2, 1), 1028.0f);
    ASSERT_FLOAT_EQ(C(2, 2), 1070.0f);
    ASSERT_FLOAT_EQ(C(2, 3), 1112.0f);

    ASSERT_FLOAT_EQ(C(3, 0), 1354.0f);
    ASSERT_FLOAT_EQ(C(3, 1), 1412.0f);
    ASSERT_FLOAT_EQ(C(3, 2), 1470.0f);
    ASSERT_FLOAT_EQ(C(3, 3), 1528.0f);

    return true;
}

bool test_transform_point_and_dir()
{
    Mat44F trans;
    trans.set_translation(Vec3F(10.0f, 20.0f, 30.0f));

    Vec3F point(1.0f, 2.0f, 3.0f);
    Vec3F dir(1.0f, 0.0f, 0.0f);

    Vec3F transformed_point = mat44f_mul_point(trans, point);
    ASSERT_VEC3_EQ(transformed_point, Vec3F(11.0f, 22.0f, 33.0f));

    Vec3F transformed_dir = mat44f_mul_dir(trans, dir);
    ASSERT_VEC3_EQ(transformed_dir, dir);

    Mat44F rot = Mat44F::from_axis_angle(Vec3F(0.0f, 0.0f, 1.0f), maths::constants::pi / 2);

    Vec3F rotated_dir = mat44f_mul_dir(rot, Vec3F(1.0f, 0.0f, 0.0f));
    ASSERT_VEC3_EQ(rotated_dir, Vec3F(0.0f, 1.0f, 0.0f));

    return true;
}

bool test_from_trs()
{
    Vec3F t(10.0f, 20.0f, 30.0f);
    Vec3F r(0.0f, maths::constants::pi / 2, 0.0f);
    Vec3F s(2.0f, 2.0f, 2.0f);

    Mat44F mat = Mat44F::from_trs(t, r, s);

    Vec3F point(1.0f, 1.0f, 1.0f);
    Vec3F transformed = mat44f_mul_point(mat, point);

    ASSERT_VEC3_EQ(transformed, Vec3F(12.0f, 22.0f, 28.0f));

    Vec3F t_out, r_out, s_out;
    mat.decompose_trs(&t_out, &r_out, &s_out);

    ASSERT_VEC3_EQ(t_out, t);
    ASSERT_FLOAT_EQ(r_out.y, r.y);
    ASSERT_VEC3_EQ(s_out, s);

    return true;
}

bool test_from_lookat()
{
    Vec3F position(0.0f, 0.0f, 10.0f);
    Vec3F lookat(0.0f, 0.0f, 0.0f);

    Mat44F view = Mat44F::from_lookat(position, lookat);

    Vec3F transformed = mat44f_mul_point(view, lookat);

    ASSERT_FLOAT_EQ(transformed.z, -10.0f);

    return true;
}

bool test_from_xyzt()
{
    Vec3F x(1.0f, 0.0f, 0.0f);
    Vec3F y(0.0f, 1.0f, 0.0f);
    Vec3F z(0.0f, 0.0f, 1.0f);
    Vec3F t(10.0f, 20.0f, 30.0f);

    Mat44F mat = Mat44F::from_xyzt(x, y, z, t);

    Vec3F x_out, y_out, z_out;
    mat.decompose_xyz(&x_out, &y_out, &z_out);

    ASSERT_VEC3_EQ(x_out, x);
    ASSERT_VEC3_EQ(y_out, y);
    ASSERT_VEC3_EQ(z_out, z);

    ASSERT_VEC3_EQ(mat.get_translation(), t);

    return true;
}

bool test_translation_functions()
{
    Mat44F mat;
    Vec3F t(10.0f, 20.0f, 30.0f);

    mat.set_translation(t);
    ASSERT_VEC3_EQ(mat.get_translation(), t);

    mat.zero_translation();
    ASSERT_VEC3_EQ(mat.get_translation(), Vec3F(0.0f, 0.0f, 0.0f));

    return true;
}

int main()
{
    int total_tests = 0;
    int failed_tests = 0;

    RUN_TEST(test_default_constructor);
    RUN_TEST(test_parameterized_constructor);
    RUN_TEST(test_operators);
    RUN_TEST(test_transpose);
    RUN_TEST(test_matrix_multiplication);
    RUN_TEST(test_transform_point_and_dir);
    RUN_TEST(test_from_trs);
    RUN_TEST(test_from_lookat);
    RUN_TEST(test_from_xyzt);
    RUN_TEST(test_translation_functions);

    stdromano::log_info("--------------------");
    stdromano::log_info("Test summary: {}/{} tests passed", (total_tests - failed_tests), total_tests);

    return 0;
}