#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <tuple>

namespace reprojection_calibration::feature_extraction {

enum class TargetType { AprilTag, Chessboard, CircleGrid };

std::tuple<Eigen::MatrixX2d, Eigen::MatrixX3d> ExtractFeatures(cv::Mat const& image, TargetType const& target_type);

}  // namespace reprojection_calibration::feature_extraction