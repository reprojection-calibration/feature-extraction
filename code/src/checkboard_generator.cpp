#include "checkboard_generator.hpp"

namespace reprojection_calibration::feature_extraction {

cv::Mat GenerateCheckboard(int const rows, int const cols, int const unit_pixel_dimension) {
    // We have to add the one unit_pixel_dimension buffer around the entire checkboard area (i.e.
    // 2*unit_pixel_dimension).
    int const height{(unit_pixel_dimension * (rows + 1)) + (2 * unit_pixel_dimension)};
    int const width{(unit_pixel_dimension * (cols + 1)) + (2 * unit_pixel_dimension)};
    cv::Mat checkerboard{255 * cv::Mat::ones(height, width, CV_8UC1)};  // Start with white image

    // TODO(Jack): Clean up this copy and pasted madness to draw the checkboard with alternating rows and columns!
    for (int row = 0; row < rows + 1; row++) {
        for (int col = 0; col < cols + 1; col++) {
            if (row % 2 == 0) {
                if (col % 2 == 0) {
                    cv::Point const top_left_corner{unit_pixel_dimension + (unit_pixel_dimension * col),
                                                    unit_pixel_dimension + (unit_pixel_dimension * row)};
                    cv::Point const bottom_right_corner{top_left_corner.x + unit_pixel_dimension,
                                                        top_left_corner.y + unit_pixel_dimension};
                    cv::rectangle(checkerboard, top_left_corner, bottom_right_corner, (0), -1);
                }

            } else {
                if (col % 2 != 0) {
                    cv::Point const top_left_corner{unit_pixel_dimension + (unit_pixel_dimension * col),
                                                    unit_pixel_dimension + (unit_pixel_dimension * row)};
                    cv::Point const bottom_right_corner{top_left_corner.x + unit_pixel_dimension,
                                                        top_left_corner.y + unit_pixel_dimension};
                    cv::rectangle(checkerboard, top_left_corner, bottom_right_corner, (0), -1);
                }
            }
        }
    }

    return checkerboard;
}

}  // namespace reprojection_calibration::feature_extraction