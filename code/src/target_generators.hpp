#pragma once

#include <opencv2/opencv.hpp>

namespace reprojection_calibration::feature_extraction {

cv::Mat GenerateCheckerboard(int const rows, int const cols, int const unit_dimension_pixels);

// unit_spacing: Given as a fraction of unit_dimension_pixels!!!
cv::Mat GenerateCircleGrid(int const rows, int const cols, int const unit_dimension_pixels,
                           int const unit_spacing_pixels, bool const asymmetric);

}  // namespace reprojection_calibration::feature_extraction