#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <tuple>

namespace reprojection_calibration::feature_extraction {

std::tuple<Eigen::MatrixX2d, Eigen::MatrixX3d> ExtractCheckerboardFeatures(cv::Mat const& image,
                                                                           cv::Point const& checkerboard_dimension);

}  // namespace reprojection_calibration::feature_extraction