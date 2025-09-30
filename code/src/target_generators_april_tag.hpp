#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace reprojection_calibration::feature_extraction {

struct AprilBoard3Generation {
    static cv::Mat GenerateBoard(int const num_bits, uint64_t const tag_family[], int const bit_size_pixels,
                                 cv::Size const& pattern_size);

    static cv::Mat GenerateTag(int const num_bits, uint64_t const tag_code, int const bit_size_pixels);

    static cv::Mat GenerateTag(int const bit_size_pixels, Eigen::MatrixXi const& code_matrix);

    static Eigen::MatrixXi GenerateCodeMatrix(int const num_bits, uint64_t const tag_code);

    static Eigen::MatrixXi Rotate90(Eigen::MatrixXi const& matrix, bool const clockwise = false);
};

}  // namespace reprojection_calibration::feature_extraction