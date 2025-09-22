#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace reprojection_calibration::feature_extraction {

// TODO(Jack): Rename
Eigen::MatrixX2d CheckerboardExtractorExtractPixelFeatures(cv::Mat const& image, cv::Size const pattern_size);

// TODO(Jack): Rename
Eigen::MatrixX3d CheckerboardExtractorExtractPointFeatures(cv::Size const pattern_size,
                                                           double const unit_dimension_meters_);

Eigen::MatrixX2d CirclegridExtractorExtractPixelFeatures(cv::Mat const& image, cv::Size const pattern_size);

}  // namespace reprojection_calibration::feature_extraction