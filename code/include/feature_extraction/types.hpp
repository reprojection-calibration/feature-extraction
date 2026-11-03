#pragma once

#include <Eigen/Dense>

// TODO(Jack): We need to remove all these types and copy and pasted helper methods from the core calibration repo and
// use these instead.

namespace reprojection::feature_extraction {

struct Bundle {
    Bundle operator()(Eigen::ArrayXi const& valid_ids) const {
        return Bundle{pixels(valid_ids, Eigen::all), points(valid_ids, Eigen::all)};
    }

    Eigen::MatrixX2d pixels;
    Eigen::MatrixX3d points;
};

struct ExtractedTarget {
    ExtractedTarget operator()(Eigen::ArrayXi const& valid_ids) const {
        return ExtractedTarget{bundle(valid_ids), indices(valid_ids, Eigen::all)};
    }

    Bundle bundle;
    Eigen::ArrayX2i indices;
};

/** \brief Supported calibration targets.
 *
 *  \par Unit Dimension Measurement
 *  To mitigate measurement error measure the board's entire height/width and divide by the corresponding the number of
 *  unit lengths.
 *
 *  \note Reprojection does not support the Kalibr style Aprilgrid, and instead uses the new and improved Aprilgrid3.
 */
enum class TargetType {
    /** \par Unit Dimension
     *  The side length of a single checkerboard square.
     *  \par Size
     *  The number of rows and columns as you would count normally. Note that the internal algorithm uses this value
     *  minus one because the feature extractor actually extracts the internal checkerboard corners.
     */
    Checkerboard,
    /** \par Unit Dimension
     *  The distance between the center's of two row/column adjacent circles. For asymmetric circle grids this cannot be
     *  measured directly.
     *  \par Size
     *  The total number of rows and columns regardless if symmetric or asymmetric. Please note that the standard OpenCv
     *  asymmetric circle grid specification counts only the asymmetric rows but all the columns to specify the size. Do
     *  not do this!
     */
    CircleGrid,
    /** \par Unit Dimension
     *  The length of a single square between the corner sharpening elements. If measuring the total internal target
     *  height/width (i.e. excluding the outer sharpening elements) make sure to account for the internal sharpening
     *  elements widths which are each 1/3 `unit_dimension` width.
     *  \par Size
     *  The total number of april tag rows and columns.
     */
    Aprilgrid3,
};

struct TargetInfo {
    TargetType target_type;
    int height;
    int width;
    double unit_dimension;
    // TODO(Jack): This annoys me that this is here because it only actually really applies to the circle grid target.
    // You can have a symmetric or asymmetric circle grid but that is not possible for the other targets.
    bool asymmetric;
};

inline std::string ToString(TargetType const target_type) {
    if (target_type == TargetType::Aprilgrid3) {
        return "aprilgrid3";
    } else if (target_type == TargetType::Checkerboard) {
        return "checkerboard";
    } else if (target_type == TargetType::CircleGrid) {
        return "circle_grid";
    } else {
        throw std::runtime_error("LIBRARY IMPLEMENTATION ERROR - Unrecognized argument passed to ToString(TargetType)");
    }
}

}  // namespace reprojection::feature_extraction