#include <gtest/gtest.h>

#include "feature_extraction/target_extractors.hpp"

using namespace reprojection_calibration::feature_extraction;

TEST(XXX, ZZZ) {
    int const n_rows = 3;
    int const n_cols = 5;

    auto const rows = Eigen::ArrayXi::LinSpaced(n_cols * n_rows, 0, n_rows - 1);
    auto const cols = Eigen::ArrayXi::LinSpaced(n_cols, 0, n_cols).colwise().replicate(n_rows);

    Eigen::ArrayX2i indices(n_rows * n_cols, 2);
    indices.col(0) = rows;
    indices.col(1) = cols;

    std::cout << indices << std::endl;

    EXPECT_EQ(1, 2);
}
