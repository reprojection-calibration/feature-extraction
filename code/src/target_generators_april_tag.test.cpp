#include "target_generators_april_tag.hpp"

#include <gtest/gtest.h>

#include <functional>  // For AprilTagFamily destructor - remove otherwise

#include "generated_apriltag_code/tagCustom36h11.h"

using namespace reprojection_calibration::feature_extraction;

// This is my attempt to RAII-ify the generated C code from the apriltag repository - without this function we need to
// manually remember to call the *_destroy() function after we are done using a tag family. Here we instead force the
// user to create a class that has both the tag family and its destruction function, which will be called when the class
// destructor is called.
//
// This answer is still not perfect because it counts on the fact that the user passes matching arguments to the
// constructor. If for example the user did the following:
//
//      AprilTagFamily(tagCustom36h11_create(), tag25h9_destroy)
//
// There will be problems because the passed destroy function does not match the created tag family. Given the
// conditions we have, and the generated C code we have to deal with, I think I have done my best, but maybe there is an
// even better way to enforce the RAII like behavior I want! The constness of the object and how it is used still needs
// to be enforced, but I am not sure how amenable C pointer magic is to this, I dot not think it is.
// TODO(Jack): Add a test to make sure the destructor logic is actually executing - already manually checked but still
// :)
struct AprilTagFamily {
   public:
    AprilTagFamily(apriltag_family_t* _tag_family, std::function<void(apriltag_family_t*)> _tag_family_destroy)
        : tag_family{_tag_family}, tag_family_destroy{std::move(_tag_family_destroy)} {}

    ~AprilTagFamily() { tag_family_destroy(tag_family); }

    apriltag_family_t* tag_family;

   private:
    std::function<void(apriltag_family_t*)> tag_family_destroy;
};

TEST(TargetGeneratorsAprilTag, TestGenerateAprilBoard) {
    cv::Size const pattern_size{4, 3};
    int const bit_size_pixel{10};
    AprilTagFamily const april_family_handler{tagCustom36h11_create(), tagCustom36h11_destroy};
    cv::Mat const april_board{GenerateAprilBoard(pattern_size, bit_size_pixel, april_family_handler.tag_family->codes)};

    EXPECT_EQ(april_board.rows, 420);
    EXPECT_EQ(april_board.cols, 560);
}

TEST(TargetGeneratorsAprilTag, TestGenerateAprilTag) {
    AprilTagFamily const april_family_handler{tagCustom36h11_create(), tagCustom36h11_destroy};
    Eigen::MatrixXi const code_matrix{CalculateCodeMatrix(36, april_family_handler.tag_family->codes[0])};
    int const bit_size_pixel{10};
    cv::Mat const april_tag{GenerateAprilTag(code_matrix, bit_size_pixel)};

    EXPECT_EQ(april_tag.rows, 140);
    EXPECT_EQ(april_tag.cols, 140);
}

TEST(TargetGeneratorsAprilTag, TestCalculateCodeMatrix) {
    AprilTagFamily const april_family_handler{tagCustom36h11_create(), tagCustom36h11_destroy};
    Eigen::MatrixXi const code_matrix{CalculateCodeMatrix(36, april_family_handler.tag_family->codes[0])};

    // Check two properties of the matrix and hope if anything in the implementation breaks these catch it -_-
    EXPECT_EQ(code_matrix.sum(), 17);                                                               // Heuristic
    EXPECT_TRUE(code_matrix.row(5).isApprox(Eigen::Vector<int, 6>{1, 1, 1, 0, 0, 1}.transpose()));  // Heuristic
}