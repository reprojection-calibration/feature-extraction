#pragma once

#include <functional>
#include <opencv2/opencv.hpp>

#include "apriltag/apriltag.h"

namespace reprojection_calibration::feature_extraction {

// WARN(Jack): The const correctness and memory safety of all apriltag related code is not clear at this point!
// (26.09.2025)

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
// WARN(Jack): This class is a footgun! Read description above.
struct AprilTagFamily {
   public:
    AprilTagFamily(apriltag_family_t* _tag_family, std::function<void(apriltag_family_t*)> _tag_family_destroy)
        : tag_family{_tag_family}, tag_family_destroy{std::move(_tag_family_destroy)} {}

    ~AprilTagFamily() { tag_family_destroy(tag_family); }

    apriltag_family_t* tag_family;

   private:
    std::function<void(apriltag_family_t*)> tag_family_destroy;
};

struct AprilTagDetectorSettings {
    double decimate;
    double blur;
    int threads;
    bool debug;
    bool refine_edges;
};

struct AprilTagDetections {
    AprilTagDetections(zarray_t* _detections) : detections{_detections} {}

    ~AprilTagDetections() { apriltag_detections_destroy(detections); }

    // WARN(Jack): This will not check out of bounds! It has assertions in the apriltag library but those will not exist
    // in a release build.
    apriltag_detection_t operator[](int i) const {
        apriltag_detection_t* det;
        zarray_get(detections, i, &det);

        return *det;
    }

    zarray_t* detections;
};

struct AprilTagDetector {
    AprilTagDetector(AprilTagFamily const& tag_family_handler, AprilTagDetectorSettings const& settings) {
        tag_detector = apriltag_detector_create();
        apriltag_detector_add_family(tag_detector, tag_family_handler.tag_family);

        tag_detector->quad_decimate = settings.decimate;
        tag_detector->quad_sigma = settings.blur;
        tag_detector->nthreads = settings.threads;
        tag_detector->debug = settings.debug;
        tag_detector->refine_edges = settings.refine_edges;
    }

    // WARN(Jack): Must be grayscale image
    AprilTagDetections Detect(cv::Mat const& gray) const {
        image_u8_t formatted_gray{gray.cols, gray.rows, gray.cols, gray.data};
        zarray_t* detections = apriltag_detector_detect(tag_detector, &formatted_gray);

        return AprilTagDetections{detections};
    }

    ~AprilTagDetector() { apriltag_detector_destroy(tag_detector); }

    apriltag_detector_t* tag_detector;
};

}  // namespace reprojection_calibration::feature_extraction