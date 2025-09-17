#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <tuple>

namespace reprojection_calibration::feature_extraction {

Eigen::MatrixX2d ExtractCheckerboardFeatures(cv::Mat const& image, cv::Point const& checkerboard_dimension);

}  // namespace reprojection_calibration::feature_extraction