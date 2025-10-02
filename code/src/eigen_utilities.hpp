#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

namespace reprojection_calibration::feature_extraction {

Eigen::ArrayX2i GenerateGridIndices(int const rows, int const cols);

Eigen::MatrixX2d ToEigen(std::vector<cv::Point2f> const& points);

Eigen::ArrayXi ToEigen(std::vector<int> const& vector);

Eigen::ArrayXi MaskIndices(Eigen::ArrayXi const& array);

}  // namespace reprojection_calibration::feature_extraction