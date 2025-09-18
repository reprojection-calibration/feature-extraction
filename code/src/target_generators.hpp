#pragma once

#include <opencv2/opencv.hpp>

namespace reprojection_calibration::feature_extraction {

cv::Mat GenerateCheckerboard(int const rows, int const cols, int const unit_dimension_pixels);

}  // namespace reprojection_calibration::feature_extraction