
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "april_tag_family_36h11.hpp"

using namespace reprojection_calibration::feature_extraction;

// DO NOT COPY AND PASTE!!!!
Eigen::ArrayX2i GenerateGridIndices(int const rows, int const cols) {
    Eigen::ArrayXi const row_indices = Eigen::ArrayXi::LinSpaced(rows * cols, 0, rows - 1);
    Eigen::ArrayXi const col_indices = Eigen::ArrayXi::LinSpaced(cols, 0, cols).colwise().replicate(rows);

    Eigen::ArrayX2i grid_indices(rows * cols, 2);
    grid_indices.col(0) = row_indices;
    grid_indices.col(1) = col_indices;

    return grid_indices;
}

Eigen::MatrixXi Rotate90(Eigen::MatrixXi const& matrix, bool const clockwise = false) {
    Eigen::MatrixXi const matrix_star{matrix.transpose()};

    return clockwise ? matrix_star.rowwise().reverse().eval() : matrix_star.colwise().reverse().eval();
}

// TODO(Jack): Consider typedef for unsigned long long type used everywhere
Eigen::MatrixXi CalculateCodeMatrix(int const bit_count, unsigned long long const tag_code) {
    int const sqrt_bit_count{static_cast<int>(std::sqrt(bit_count))};

    Eigen::MatrixXi code_matrix(sqrt_bit_count, sqrt_bit_count);
    for (int i{0}; i < sqrt_bit_count; ++i) {
        for (int j{0}; j < sqrt_bit_count; ++j) {
            unsigned long long bit_sign{(tag_code & (static_cast<unsigned long long>(1) << (sqrt_bit_count * i + j)))};
            code_matrix(i, j) = not bit_sign;
        }
    }

    return Rotate90(Rotate90(code_matrix));
}

cv::Mat GenerateAprilTag(Eigen::MatrixXi const& code_matrix, int const border_thickness_bits,
                         int const bit_size_pixel) {
    // NOTE(Jack): This logic assumes the code_matrix is square and therefore the generated april tag is square
    int const border_thickness_pixels{2 * border_thickness_bits * bit_size_pixel};
    int const data_region_size_bits{static_cast<int>(code_matrix.rows())};  // One side of the square

    int const height{border_thickness_pixels + (data_region_size_bits * bit_size_pixel)};
    int const width{height};
    cv::Mat april_tag{cv::Mat::zeros(height, width, CV_8UC1)};

    // TODO(Jack): This logic is practically exactly the same as in the checkerboard generation... is there a practical
    // way to DRY ourselves here?
    Eigen::ArrayX2i const grid{GenerateGridIndices(data_region_size_bits, data_region_size_bits)};
    for (Eigen::Index i{0}; i < grid.rows(); ++i) {
        Eigen::Array2i const indices{grid.row(i)};

        if (not code_matrix(indices(0), indices(1))) {
            cv::Point const top_left_corner{(bit_size_pixel * indices(1)) + (border_thickness_pixels / 2),
                                            (bit_size_pixel * indices(0)) + (border_thickness_pixels / 2)};
            cv::Point const bottom_right_corner{top_left_corner.x + bit_size_pixel, top_left_corner.y + bit_size_pixel};
            cv::rectangle(april_tag, top_left_corner, bottom_right_corner, (255), -1);
        }
    }

    return april_tag;
}

TEST(TargetGeneratorsAprilTag, TestGenerateAprilTag) {
    Eigen::MatrixXi const code_matrix{CalculateCodeMatrix(36, april_tag::t36h11[0])};
    int const border_thickness_bits{2};  // Kalibr uses two, but most april grid applications only use one border bit.
    int const bit_size_pixel{10};
    cv::Mat const april_tag{GenerateAprilTag(code_matrix, border_thickness_bits, bit_size_pixel)};

    EXPECT_EQ(april_tag.rows, 100);
    EXPECT_EQ(april_tag.cols, 100);
}

TEST(TargetGeneratorsAprilTag, TestCalculateCodeMatrix) {
    Eigen::MatrixXi const code_matrix{CalculateCodeMatrix(36, april_tag::t36h11[0])};

    // Check two properties of the matrix and hope if anything in the implementation breaks these catch it -_-
    EXPECT_EQ(code_matrix.sum(), 20);                                                               // Heuristic
    EXPECT_TRUE(code_matrix.row(5).isApprox(Eigen::Vector<int, 6>{1, 1, 1, 0, 1, 1}.transpose()));  // Heuristic
}