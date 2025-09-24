#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace reprojection_calibration::feature_extraction {

cv::Mat GenerateCheckerboard(cv::Size const& pattern_size, int const square_size_pixels);

cv::Mat GenerateCircleGrid(cv::Size const& pattern_size, int const circle_radius_pixels,
                           int const circle_spacing_pixels, bool const asymmetric);

}  // namespace reprojection_calibration::feature_extraction