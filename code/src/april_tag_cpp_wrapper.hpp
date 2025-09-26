#pragma once

extern "C" {
#include <apriltag/apriltag.h>
}

#include <functional>
#include <opencv2/opencv.hpp>

// This is my attempt to RAII-ify the  C code from the apriltag repository. The main thing I try to fight here is
// manually having to deallocate memory. For the detector and tag detections themselves that is relatively easy because
// they have generic creation and destruction functions. For the tag family it is trickier because the generated code is
// actually specific to each one (see comment in AprilTagFamily).
//
// WARN(Jack): The const correctness and memory safety of all apriltag related code is not clear at this point
// (26.09.2025)! I am 99% sure that there are some big footguns in here, and we will find some "presents" later.

namespace reprojection_calibration::feature_extraction {

struct AprilTagFamily {
    // WARN(Jack): If the user passes mismatched _tag_family and _tag_family_destroy this class will not do what they
    // actually want it to! If for example the user did the following:
    //
    //      AprilTagFamily(tagCustom36h11_create(), tag25h9_destroy)
    //
    // Then this is a huge footgun because the created tag family does not match the one that will be destroyed when the
    // destructor is called. Because of the nature of the generated code there is no way to enforce that matching pairs
    // or passed, or that the proper destructor is called automatically.
    AprilTagFamily(apriltag_family_t* _tag_family, std::function<void(apriltag_family_t*)> _tag_family_destroy)
        : tag_family{_tag_family}, tag_family_destroy{std::move(_tag_family_destroy)} {}

    ~AprilTagFamily() { tag_family_destroy(tag_family); }

    apriltag_family_t* tag_family;

   private:
    std::function<void(apriltag_family_t*)> tag_family_destroy;
};

struct AprilTagDetections {
    explicit AprilTagDetections(zarray_t* _detections) : detections{_detections} {}

    ~AprilTagDetections() { apriltag_detections_destroy(detections); }

    // WARN(Jack): This will not check out of bounds! It has assertions in the apriltag library but those will not exist
    // in a release build.
    apriltag_detection_t operator[](int const i) const {
        apriltag_detection_t* det;
        zarray_get(detections, i, &det);

        return *det;
    }

    zarray_t* detections;
};

struct AprilTagDetector {
    struct AprilTagDetectorSettings {
        double decimate;
        double blur;
        int threads;
        bool debug;
        bool refine_edges;
    };

    AprilTagDetector(AprilTagFamily const& tag_family_handler, AprilTagDetectorSettings const& settings)
        : tag_detector{apriltag_detector_create()} {
        apriltag_detector_add_family(tag_detector, tag_family_handler.tag_family);

        tag_detector->quad_decimate = settings.decimate;
        tag_detector->quad_sigma = settings.blur;
        tag_detector->nthreads = settings.threads;
        tag_detector->debug = settings.debug;
        tag_detector->refine_edges = settings.refine_edges;
    }

    // WARN(Jack): Must be grayscale image
    AprilTagDetections Detect(cv::Mat const& gray) const {
        image_u8_t raw_gray{gray.cols, gray.rows, gray.cols, gray.data};
        zarray_t* raw_detections{apriltag_detector_detect(tag_detector, &raw_gray)};

        return AprilTagDetections{raw_detections};
    }

    ~AprilTagDetector() { apriltag_detector_destroy(tag_detector); }

   private:
    apriltag_detector_t* tag_detector;
};

}  // namespace reprojection_calibration::feature_extraction