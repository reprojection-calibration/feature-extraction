#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace reprojection_calibration::feature_extraction {

cv::Mat GenerateAprilBoard(int const num_bits, uint64_t const tag_family[], int const bit_size_pixels,
                           cv::Size const& pattern_size);

cv::Mat GenerateAprilTag(int const num_bits, uint64_t const tag_code, int const bit_size_pixels);

cv::Mat GenerateAprilTag(int const bit_size_pixels, Eigen::MatrixXi const& code_matrix);

Eigen::MatrixXi CalculateCodeMatrix(int const num_bits, uint64_t const tag_code);

Eigen::MatrixXi Rotate90(Eigen::MatrixXi const& matrix, bool const clockwise = false);

}  // namespace reprojection_calibration::feature_extraction