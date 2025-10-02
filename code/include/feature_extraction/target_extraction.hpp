#pragma once

#include <Eigen/Dense>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>

namespace reprojection_calibration::feature_extraction {

struct FeatureFrame {
    Eigen::MatrixX2d pixels;
    // TODO(Jack): Is it better to return a 2d index or 1d index?
    Eigen::ArrayX2i indices;
};

class TargetExtractor {
   public:
    TargetExtractor(cv::Size const& pattern_size) : pattern_size_{pattern_size} {}

    virtual ~TargetExtractor() = default;

    // NOTE(Jack): In the future this will return a more complex data type that fully describes the target detection
    virtual std::optional<FeatureFrame> Extract(cv::Mat const& image) const = 0;

   protected:
    cv::Size pattern_size_;
};

enum class TargetType { Checkerboard, CircleGrid, AprilGrid3 };

// NOTE(Jack): One day the argument to CreateTargetExtractor() will be the path to a configuration file. Until then we
// will probably hard code some things which in the future will come from that file. Maybe there would be a more
// eloquent way to do this for testing purposes with some test fixture utilities but I am not sure.
std::unique_ptr<TargetExtractor> CreateTargetExtractor(const TargetType type);

}  // namespace reprojection_calibration::feature_extraction