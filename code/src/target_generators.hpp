#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace reprojection::feature_extraction {

cv::Mat GenerateCheckerboard(cv::Size const& pattern_size, int square_size_pixels);

cv::Mat GenerateCircleGrid(cv::Size const& pattern_size, int circle_radius_pixels, int circle_spacing_pixels,
                           bool asymmetric);

struct Aprilgrid3Generation {
    static cv::Mat GenerateBoard(int num_bits, uint64_t const tag_family[], int bit_size_pixels,
                                 cv::Size const& pattern_size);

    static cv::Mat GenerateTag(int num_bits, uint64_t tag_code, int bit_size_pixels);

    static cv::Mat GenerateTag(int bit_size_pixels, Eigen::MatrixXi const& code_matrix);

    static Eigen::MatrixXi GenerateCodeMatrix(int num_bits, uint64_t tag_code);

    static Eigen::MatrixXi Rotate90(Eigen::MatrixXi const& matrix, bool clockwise = false);
};

}  // namespace reprojection::feature_extraction