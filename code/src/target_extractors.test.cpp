#include "target_extractors.hpp"

#include <gtest/gtest.h>

#include "eigen_utilities.hpp"  // REMOVE
#include "target_generators.hpp"
#include "test_fixture_april_tag.hpp"

using namespace reprojection_calibration::feature_extraction;

TEST(TargetExtractors, TestCheckerboardExtractor) {
    cv::Size const pattern_size{4, 3};  // (width, height) == (cols, rows)
    int const square_size_pixels{50};
    cv::Mat const image{GenerateCheckerboard(pattern_size, square_size_pixels)};

    auto const extractor{CheckerboardExtractor{pattern_size}};

    std::optional<FeatureFrame> const target{extractor.Extract(image)};
    ASSERT_TRUE(target.has_value());

    Eigen::MatrixX2d const& pixels{target->pixels};
    EXPECT_EQ(pixels.rows(), pattern_size.height * pattern_size.width);
    EXPECT_TRUE(pixels.row(0).isApprox(Eigen::Vector2d{100, 100}.transpose(), 1e-6));   // First pixel - heuristic
    EXPECT_TRUE(pixels.row(11).isApprox(Eigen::Vector2d{250, 200}.transpose(), 1e-6));  // Last pixel - heuristic

    Eigen::ArrayX2i const& indices{target->indices};
    EXPECT_EQ(indices.rows(), pattern_size.width * pattern_size.height);
    EXPECT_TRUE(indices.row(0).isApprox(Eigen::Vector2i{0, 0}.transpose()));   // First index - heuristic
    EXPECT_TRUE(indices.row(11).isApprox(Eigen::Vector2i{2, 3}.transpose()));  // Last index - heuristic
}

TEST(TargetExtractors, TestCircleGridExtractor) {
    cv::Size const pattern_size{4, 3};
    int const circle_radius_pixels{25};
    int const circle_spacing_pixels{20};  // Between circle edges
    bool const asymmetric{false};
    cv::Mat const image{GenerateCircleGrid(pattern_size, circle_radius_pixels, circle_spacing_pixels, asymmetric)};

    auto const extractor{CircleGridExtractor{pattern_size, asymmetric}};

    std::optional<FeatureFrame> const target{extractor.Extract(image)};
    ASSERT_TRUE(target.has_value());

    Eigen::MatrixX2d const& pixels{target->pixels};
    EXPECT_EQ(pixels.rows(), pattern_size.width * pattern_size.height);
    EXPECT_TRUE(pixels.row(0).isApprox(Eigen::Vector2d{265, 195}.transpose(), 1e-6));
    EXPECT_TRUE(pixels.row(11).isApprox(Eigen::Vector2d{55, 55}.transpose(), 1e-6));

    Eigen::ArrayX2i const& indices{target->indices};
    EXPECT_EQ(indices.rows(), pattern_size.height * pattern_size.width);
    EXPECT_TRUE(indices.row(0).isApprox(Eigen::Vector2i{0, 0}.transpose()));
    EXPECT_TRUE(indices.row(11).isApprox(Eigen::Vector2i{2, 3}.transpose()));
}

TEST(TargetExtractors, TestCircleGridExtractorAsymmetric) {
    // Refactor to use cv::Size
    // WARN(Jack): Must be even (rows)! See comment below.
    // WARN(Jack): Must be an odd number (cols) to prevent 180 degree rotation symmetry!
    // https://answers.opencv.org/question/96561/calibration-with-findcirclesgrid-trouble-with-pattern-widthheight/
    cv::Size const pattern_size{7, 6};
    int const circle_radius_pixels{25};
    int const circle_spacing_pixels{20};
    bool const asymmetric{true};
    cv::Mat image{GenerateCircleGrid(pattern_size, circle_radius_pixels, circle_spacing_pixels, asymmetric)};

    auto const extractor{CircleGridExtractor{pattern_size, asymmetric}};

    std::optional<FeatureFrame> const target{extractor.Extract(image)};
    ASSERT_TRUE(target.has_value());

    Eigen::MatrixX2d const& pixels{target->pixels};
    EXPECT_EQ(pixels.rows(),
              (pattern_size.width * pattern_size.height) / 2);  // NOTE(Jack): Divide by two due to asymmetry!
    EXPECT_TRUE(pixels.row(0).isApprox(Eigen::Vector2d{475, 55}.transpose(), 1e-6));
    EXPECT_TRUE(pixels.row(20).isApprox(Eigen::Vector2d{55, 335}.transpose(), 1e-6));

    Eigen::ArrayX2i const& indices{target->indices};
    EXPECT_EQ(indices.rows(), (pattern_size.width * pattern_size.height) / 2);
    EXPECT_TRUE(indices.row(0).isApprox(Eigen::Vector2i{0, 0}.transpose()));
    EXPECT_TRUE(indices.row(20).isApprox(Eigen::Vector2i{6, 4}.transpose()));
}

TEST_F(AprilTagTestFixture, TestAprilGrid3Extractor) {
    cv::Mat const april_tag{AprilBoard3Generation::GenerateTag(bit_size_pixel_, code_matrix_0_)};

    cv::Size const pattern_size{4, 3};
    auto const extractor{AprilGrid3Extractor{pattern_size}};

    std::optional<FeatureFrame> const target{extractor.Extract(april_tag)};
    ASSERT_TRUE(target.has_value());

    Eigen::MatrixX2d const& pixels{target->pixels};
    EXPECT_EQ(pixels.rows(), 4);  // One tag
    Eigen::Matrix<double, 4, 2> const gt_pixels{{19.819417953491211, 119.27910614013672},
                                                {119.13014984130859, 119.13014984130859},
                                                {119.27910614013672, 19.819416046142578},
                                                {19.685731887817383, 19.685731887817383}};
    EXPECT_TRUE(pixels.isApprox(gt_pixels, 1e-6));
}

Eigen::ArrayX2i AprilGrid3Extractor::CornerIndices(cv::Size const& pattern_size,
                                                   std::vector<AprilTagDetection> const& detections) {
    // NOTE(Jack): Multiplied by two because every tag has four corners/points/pixels (two in each direction)
    Eigen::ArrayX2i const grid{GenerateGridIndices(2 * pattern_size.height, 2 * pattern_size.width)};

    // TODO(Jack): The logic in this method about indicing and masking is very similar to the code in the eigen utility
    // MaskIndices, keep your eyes peeled for optimization or code/idea reuse
    std::vector<int> mask_vec;
    mask_vec.reserve(grid.rows());
    for (auto const& detection : detections) {
        // WARN(Jack): THIS WILL ASSUME WE ARE ALWAYS STARTING from a tag ID of zero
        for (int i{0}; i < 4; ++i) {
            int const corner_id{(4 * detection.id) + i};
            mask_vec.push_back(corner_id);
        }
    }

    // COPY AND PASTED FROM EIGEN UTILTIES MASK FUNCTION
    Eigen::ArrayXi mask(std::size(mask_vec));
    mask = Eigen::Map<Eigen::ArrayXi>(mask_vec.data(), std::size(mask_vec));

    return grid(mask, Eigen::all);
}

TEST_F(AprilTagTestFixture, TestAprilGrid3ExtractorCornerIndices) {
    cv::Size const pattern_size{3, 2};

    std::vector<AprilTagDetection> detections;
    for (int i{0}; i < pattern_size.width * pattern_size.height; ++i) {
        AprilTagDetection detection_i;
        detection_i.id = i;

        detections.push_back(detection_i);
    }

    Eigen::ArrayX2i const corner_indices{AprilGrid3Extractor::CornerIndices(pattern_size, detections)};

    EXPECT_EQ(corner_indices.rows(), 4 * (pattern_size.width * pattern_size.height));

    std::cout << corner_indices << std::endl;
}