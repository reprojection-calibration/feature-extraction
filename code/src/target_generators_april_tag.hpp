
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "grid_utilities.hpp"

namespace reprojection_calibration::feature_extraction {

// What if we want metric information, we work in pixel space now but we might want metric sizes eventually! This should
// be considered for all target types potentially!
// ERROR(Jack): We need to define a type that contains the tag family and its code, plus the number of bits. That
// information is currently hardcoded!
cv::Mat GenerateAprilBoard(cv::Size const& pattern_size,
                           int const bit_size_pixels, uint64_t const tag_family[]);

cv::Mat GenerateAprilTag(Eigen::MatrixXi const& code_matrix, int const bit_size_pixel);

// TODO(Jack): Consider typedef for unsigned long long type used everywhere
Eigen::MatrixXi CalculateCodeMatrix(int const bit_count, unsigned long long const tag_code);

Eigen::MatrixXi Rotate90(Eigen::MatrixXi const& matrix, bool const clockwise = false);

}  // namespace reprojection_calibration::feature_extraction