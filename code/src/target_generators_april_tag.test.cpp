
#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "april_tag_family_36h11.hpp"

using namespace reprojection_calibration::feature_extraction;

// TODO(Jack): Consider typedef for unsigned long long type used everywhere
Eigen::MatrixXi GenerateCodeMatrix(int const bit_count, unsigned long long const tag_code) {
    int const sqrt_bit_count{static_cast<int>(std::sqrt(bit_count))};

    Eigen::MatrixXi code_matrix(sqrt_bit_count, sqrt_bit_count);
    for (int i{0}; i < sqrt_bit_count; ++i) {
        for (int j{0}; j < sqrt_bit_count; ++j) {
            unsigned long long bit_sign{(tag_code & (static_cast<unsigned long long>(1) << (sqrt_bit_count * i + j)))};
            code_matrix(i, j) = not bit_sign;
        }
    }
    std::cout << code_matrix << std::endl;

    return Eigen::MatrixXi();
}

TEST(TargetGeneratorsAprilTag, XXX) {
    GenerateCodeMatrix(36, april_tag::t36h11[0]);
    EXPECT_TRUE(false);
}