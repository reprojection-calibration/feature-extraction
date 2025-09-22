#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <optional>

namespace reprojection_calibration::feature_extraction {

// TODO(Jack): Rename
std::optional<Eigen::MatrixX2d> CheckerboardExtractorExtractPixelFeatures(cv::Mat const& image,
                                                                          cv::Size const pattern_size);

// TODO(Jack): Rename
Eigen::MatrixX3d CheckerboardExtractorExtractPointFeatures(cv::Size const pattern_size,
                                                           double const unit_dimension_meters_);

// TODO(Jack): Rename
std::optional<Eigen::MatrixX2d> CirclegridExtractorExtractPixelFeatures(cv::Mat const& image,
                                                                        cv::Size const pattern_size,
                                                                        bool const asymmetric);

}  // namespace reprojection_calibration::feature_extraction