#pragma once

#include <opencv2/opencv.hpp>

namespace reprojection_calibration::feature_extraction {

cv::Mat GenerateCheckboard(int const rows, int const cols, int const unit_pixel_dimension);

}  // namespace reprojection_calibration::feature_extraction