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

    std::optional<FeatureFrame> Extract(cv::Mat const& image) const override;

   private:
    // Calculated once during construction because for checkerboards and circle grids we either capture the entire board
    // or none of it. Therefore we calculate it here once and return it with every successful detection. This is
    // different from the april grid which can have partial detections and requires more complex logic.
    Eigen::ArrayX2i point_indices_;
};

class CircleGridExtractor : public TargetExtractor {
   public:
    CircleGridExtractor(cv::Size const& pattern_size, bool const asymmetric);

    std::optional<FeatureFrame> Extract(cv::Mat const& image) const override;

   private:
    bool asymmetric_;
    // WARN(Jack): For the checkerboard I think the ID conceptually matches the extracted pixel coordinate. I.e. the top
    // left extracted point gets ID (0,0) and the next point in that row gets (0, 1). For the circle grid it does not
    // match so simply. The numbers of rows and columns and their corresponding relative position in the image match,
    // but maybe not the order. This might be a nothingburger, as long as we are consistent, or it might cause us
    // headaches later.
    Eigen::ArrayX2i point_indices_;
};

class AprilGrid3Extractor : public TargetExtractor {
   public:
    explicit AprilGrid3Extractor(cv::Size const& pattern_size);

    std::optional<FeatureFrame> Extract(cv::Mat const& image) const override;

    static Eigen::ArrayX2i CornerIndices(cv::Size const& pattern_size,
                                         std::vector<AprilTagDetection> const& detections);

   private:
    // TODO(Jack): Consider making these two extraction functions public and testing them!
    static Eigen::Matrix<double, 4, 2> EstimateExtractionCorners(Eigen::Matrix3d const& H, int const sqrt_num_bits);

    static Eigen::Matrix<double, 4, 2> RefineCorners(cv::Mat const& image,
                                                     Eigen::Matrix<double, 4, 2> const& extraction_corners);

    AprilTagFamily tag_family_;
    AprilTagDetector tag_detector_;
};

}  // namespace reprojection_calibration::feature_extraction