#pragma once

#include <Eigen/Dense>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>

// TODO(Jack): I think a better name for this file would reflect the fact that we are extracting targets here?

namespace reprojection_calibration::feature_extraction {

class TargetExtractor {
   public:
    TargetExtractor(cv::Size const& pattern_size) : pattern_size_{pattern_size} {}

    virtual ~TargetExtractor() {}

    // NOTE(Jack): In the future this will return a more complex data type that fully describes the
    virtual std::optional<Eigen::MatrixX2d> Extract(cv::Mat const& image) const = 0;

   protected:
    cv::Size pattern_size_;
};

enum class TargetType { Checkerboard, CircleGrid, AprilGrid3 };

// NOTE(Jack): One day the argument to CreateTargetExtractor() will be the path to a configuration file. Until then we
// will probably hard code some things which in the future will come from that file. Maybe there would be a more
// eloquent way to do this for testing purposes with some test fixture utilities but I am not sure.
std::unique_ptr<TargetExtractor> CreateTargetExtractor(const TargetType type);

}  // namespace reprojection_calibration::feature_extraction