#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace reprojection_calibration::feature_extraction {

cv::Mat GenerateCheckerboard(cv::Size const& pattern_size, int const unit_dimension_pixels);

// unit_spacing: Given as a fraction of unit_dimension_pixels!!!
cv::Mat GenerateCircleGrid(int rows, int cols, int const unit_dimension_pixels, int const unit_spacing_pixels,
                           bool const asymmetric);

Eigen::ArrayX2i GenerateGridIndices(int const rows, int const cols);

}  // namespace reprojection_calibration::feature_extraction