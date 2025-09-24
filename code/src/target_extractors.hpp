#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <optional>

namespace reprojection_calibration::feature_extraction {

// TODO(Jack): Rename
std::optional<Eigen::MatrixX2d> CheckerboardExtractorExtractPixelFeatures(cv::Mat const& image,
                                                                          cv::Size const pattern_size);

// TODO(Jack): Rename
std::optional<Eigen::MatrixX2d> CirclegridExtractorExtractPixelFeatures(cv::Mat const& image,
                                                                        cv::Size const pattern_size,
                                                                        bool const asymmetric);

// TODO(Jack): Put in helper file if better organized there
Eigen::MatrixX2d ToEigen(std::vector<cv::Point2f> const& points);

}  // namespace reprojection_calibration::feature_extraction