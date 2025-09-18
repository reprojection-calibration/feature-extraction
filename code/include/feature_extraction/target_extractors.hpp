#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace reprojection_calibration::feature_extraction {

class TargetExtractor {
   public:
    virtual std::tuple<Eigen::MatrixX2d, Eigen::MatrixX3d> ExtractTarget(cv::Mat const& image) const = 0;

    TargetExtractor() = default;

    virtual ~TargetExtractor() = default;

   private:
    virtual Eigen::MatrixX2d ExtractPixelFeatures(cv::Mat const& image) const = 0;

    virtual Eigen::MatrixX3d ExtractPointFeatures() const = 0;
};

class CheckerboardExtractor : TargetExtractor {
   public:
    CheckerboardExtractor(cv::Size const& pattern_size, double const unit_dimension_meters);

    std::tuple<Eigen::MatrixX2d, Eigen::MatrixX3d> ExtractTarget(cv::Mat const& image) const final;

   private:
    Eigen::MatrixX2d ExtractPixelFeatures(cv::Mat const& image) const final;

    Eigen::MatrixX3d ExtractPointFeatures() const final;

    cv::Size pattern_size_;
    double unit_dimension_meters_;
};

}  // namespace reprojection_calibration::feature_extraction