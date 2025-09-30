#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <optional>

#include "feature_extraction/target_extraction.hpp"

namespace reprojection_calibration::feature_extraction {

class CheckerboardExtractor : public TargetExtractor {
   public:
    CheckerboardExtractor(cv::Size const& patern_size);

    std::optional<Eigen::MatrixX2d> Extract(cv::Mat const& image) const override;
};

class CircleGridExtractor : public TargetExtractor {
   public:
    CircleGridExtractor(cv::Size const& patern_size, bool const asymmetric);

    std::optional<Eigen::MatrixX2d> Extract(cv::Mat const& image) const override;

   private:
    bool asymmetric_;
};

// TODO(Jack): Put in helper file if better organized there
Eigen::MatrixX2d ToEigen(std::vector<cv::Point2f> const& points);

}  // namespace reprojection_calibration::feature_extraction