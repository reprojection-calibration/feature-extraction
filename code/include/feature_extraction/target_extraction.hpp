#pragma once

#include <Eigen/Dense>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>

namespace reprojection_calibration::feature_extraction {

struct FeatureFrame {
    Eigen::MatrixX2d pixels;
    Eigen::MatrixX3d points;
    // TODO(Jack): Is it better to return a 2d index or 1d index?
    Eigen::ArrayX2i indices;
};

class TargetExtractor {
   public:
    TargetExtractor(cv::Size const& pattern_size, const double unit_dimension)
        : pattern_size_{pattern_size}, unit_dimension_{unit_dimension} {}

    virtual ~TargetExtractor() = default;

    virtual std::optional<FeatureFrame> Extract(cv::Mat const& image) const = 0;

   protected:
    cv::Size pattern_size_;
    double unit_dimension_;
    // NOTE(Jack): For detectors which can only detect "whole" boards the Extract() method will simply return these in
    // their entirety. For targets which can have partial detections (ex. AprilGrid3) their Extract() method will mask
    // out the indices and points which were visible and only return those.
    Eigen::ArrayX2i point_indices_;
    Eigen::MatrixX3d points_;
};

enum class TargetType { Checkerboard, CircleGrid, AprilGrid3 };

// NOTE(Jack): One day the argument to CreateTargetExtractor() will be the path to a configuration file. Until then we
// will probably hard code some things which in the future will come from that file. Maybe there would be a more
// eloquent way to do this for testing purposes with some test fixture utilities but I am not sure.
std::unique_ptr<TargetExtractor> CreateTargetExtractor(const TargetType type);

}  // namespace reprojection_calibration::feature_extraction