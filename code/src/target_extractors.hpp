#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <optional>

#include "april_tag_cpp_wrapper.hpp"
#include "feature_extraction/target_extraction.hpp"

namespace reprojection_calibration::feature_extraction {

class CheckerboardExtractor : public TargetExtractor {
   public:
    explicit CheckerboardExtractor(cv::Size const& pattern_size);

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
    explicit AprilGrid3Extractor(cv::Size const& pattern_size);

    std::optional<Eigen::MatrixX2d> Extract(cv::Mat const& image) const override;

    static Eigen::Matrix<double, 4, 2> EstimateExtractionCorners(Eigen::Matrix3d const& H, int const sqrt_num_bits);

    static Eigen::Matrix<double, 4, 2> RefineCorners(cv::Mat const& image,
                                                    Eigen::Matrix<double, 4, 2> const& extraction_corners);

   private:
    AprilTagFamily tag_family_;
    AprilTagDetector tag_detector_;
};

// TODO(Jack): Put in helper file if better organized there
Eigen::MatrixX2d ToEigen(std::vector<cv::Point2f> const& points);

}  // namespace reprojection_calibration::feature_extraction