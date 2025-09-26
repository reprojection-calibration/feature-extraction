cmake_minimum_required(VERSION 3.28)

project(
        feature-extraction
        VERSION 0.0.1
        LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Werror -Wall -Wextra -Wpedantic)

# Taken from https://github.com/amy-tabb/apriltag-cmake-example/tree/master
find_package(PkgConfig)
pkg_search_module(apriltag REQUIRED apriltag)
set(apriltag_INCLUDE_DIRS "${apriltag_INCLUDE_DIRS}/apriltag")

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

set(SRC_FILES
        src/grid_utilities.cpp
        src/target_generators_april_tag.cpp
        src/target_generators.cpp
        src/target_extractors.cpp
        src/generated_apriltag_code/tagCustom36h11.c
)
add_library(${PROJECT_NAME} SHARED
        ${SRC_FILES}
)
target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>
        $<INSTALL_INTERFACE:include/>
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PUBLIC
        Eigen3::Eigen
        ${apriltag_LIBRARIES}
        ${OpenCV_LIBS}
)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)
enable_testing()
include(GoogleTest)

set(TESTS
        src/grid_utilities.test.cpp
        src/target_extractors_april_tag.test.cpp
        src/target_extractors.test.cpp
        src/target_generators_april_tag.test.cpp
        src/target_generators.test.cpp
        test/feature_extraction.test.cpp
)
foreach (TEST IN LISTS TESTS)
    get_filename_component(TEST_NAME ${TEST} NAME)
    add_executable(${TEST_NAME} ${TEST})
    target_link_libraries(${TEST_NAME}
            GTest::gtest_main
            ${PROJECT_NAME}
    )
    gtest_discover_tests(${TEST_NAME})
endforeach ()
