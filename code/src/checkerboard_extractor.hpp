#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace reprojection_calibration::feature_extraction {

// TODO(Jack): Rename
Eigen::MatrixX2d CheckerboardExtractorExtractPixelFeatures(cv::Mat const& image, cv::Size const pattern_size_);

// TODO(Jack): Rename
Eigen::MatrixX3d CheckerboardExtractorExtractPointFeatures(cv::Size const pattern_size_,
                                                           double const unit_dimension_meters_);

}  // namespace reprojection_calibration::feature_extraction