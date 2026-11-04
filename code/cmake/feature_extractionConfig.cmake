include(CMakeFindDependencyMacro)

find_dependency(Ceres REQUIRED)
find_dependency(Eigen3 REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/featureExtractionTargets.cmake")