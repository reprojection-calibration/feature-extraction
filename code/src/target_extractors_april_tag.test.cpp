
#include <gtest/gtest.h>

#include "april_tag_handling.hpp"

extern "C" {
#include "apriltag/apriltag.h"
#include "generated_apriltag_code/tagCustom36h11.h"
}

// {  }

namespace reprojection_calibration::feature_extraction {}  // namespace reprojection_calibration::feature_extraction

using namespace reprojection_calibration::feature_extraction;

TEST(TargetExtractorsAprilTag, XXX) {
    AprilTagFamily const tag_family_handler{tagCustom36h11_create(), tagCustom36h11_destroy};
    AprilTagDetectorSettings const settings{2.0, 0.0, 1, false, false};
    AprilTagDetector const tag_detector{tag_family_handler, settings};

    EXPECT_EQ(1, 2);
}
