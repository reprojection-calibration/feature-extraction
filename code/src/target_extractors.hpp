#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <optional>

#include "april_tag_cpp_wrapper.hpp"
#include "feature_extraction/target_extraction.hpp"

namespace reprojection_calibration::feature_extraction {

class CheckerboardExtractor : public TargetExtractor {
   public:
    CheckerboardExtractor(cv::Size const& pattern_size);

    std::optional<Eigen::MatrixX2d> Extract(cv::Mat const& image) const override;
};

class CircleGridExtractor : public TargetExtractor {
   public:
    CircleGridExtractor(cv::Size const& pattern_size, bool const asymmetric);

    std::optional<Eigen::MatrixX2d> Extract(cv::Mat const& image) const override;

   private:
    bool asymmetric_;
};

class AprilGrid3Extractor : public TargetExtractor {
   public:
    AprilGrid3Extractor(cv::Size const& pattern_size);

    std::optional<Eigen::MatrixX2d> Extract(cv::Mat const& image) const override;

   private:
    AprilTagFamily tag_family_;
    AprilTagDetector tag_detector_;
};

// TODO(Jack): Put in helper file if better organized there
Eigen::MatrixX2d ToEigen(std::vector<cv::Point2f> const& points);

}  // namespace reprojection_calibration::feature_extraction