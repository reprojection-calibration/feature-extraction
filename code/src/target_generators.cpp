#include "target_generators.hpp"

namespace reprojection_calibration::feature_extraction {

// TODO(Jack): Consider using cv:Size instead of rows and cols.
cv::Mat GenerateCheckerboard(int const rows, int const cols, int const unit_dimension_pixels) {
    // We have to add the one unit_dimension_pixels buffer around the entire checkboard area (i.e.
    // 2*unit_dimension_pixels).
    int const height{(unit_dimension_pixels * (rows + 1)) + (2 * unit_dimension_pixels)};
    int const width{(unit_dimension_pixels * (cols + 1)) + (2 * unit_dimension_pixels)};
    cv::Mat checkerboard{255 * cv::Mat::ones(height, width, CV_8UC1)};  // Start with white image

    // TODO(Jack): Clean up this copy and pasted madness to draw the checkboard with alternating rows and columns!
    for (int row = 0; row < rows + 1; row++) {
        for (int col = 0; col < cols + 1; col++) {
            if (row % 2 == 0) {
                if (col % 2 == 0) {
                    cv::Point const top_left_corner{unit_dimension_pixels + (unit_dimension_pixels * col),
                                                    unit_dimension_pixels + (unit_dimension_pixels * row)};
                    cv::Point const bottom_right_corner{top_left_corner.x + unit_dimension_pixels,
                                                        top_left_corner.y + unit_dimension_pixels};
                    cv::rectangle(checkerboard, top_left_corner, bottom_right_corner, (0), -1);
                }

            } else {
                if (col % 2 != 0) {
                    cv::Point const top_left_corner{unit_dimension_pixels + (unit_dimension_pixels * col),
                                                    unit_dimension_pixels + (unit_dimension_pixels * row)};
                    cv::Point const bottom_right_corner{top_left_corner.x + unit_dimension_pixels,
                                                        top_left_corner.y + unit_dimension_pixels};
                    cv::rectangle(checkerboard, top_left_corner, bottom_right_corner, (0), -1);
                }
            }
        }
    }

    return checkerboard;
}

}  // namespace reprojection_calibration::feature_extraction