cmake_minimum_required(VERSION 3.28)

project(
        feature-extraction
        VERSION 0.0.1
        LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Werror -Wall -Wextra -Wpedantic)

# Taken from https://github.com/amy-tabb/apriltag-cmake-example/tree/master (did not include the link_directories() command here)
find_package(PkgConfig)
pkg_search_module(apriltag REQUIRED apriltag)
set(apriltag_INCLUDE_DIRS "${apriltag_INCLUDE_DIRS}/apriltag")

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

set(SRC_FILES
        src/april_tag_cpp_wrapper.cpp
        src/generated_apriltag_code/tagCustom36h11.c
        src/utilities.cpp
        src/target_extraction.cpp
        src/target_extractors.cpp
        src/target_generators.cpp
)
add_library(${PROJECT_NAME} SHARED
        ${SRC_FILES}
)
target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>
        $<INSTALL_INTERFACE:include/>
        ${OpenCV_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIR}
)
target_link_libraries(${PROJECT_NAME} PUBLIC
        Eigen3::Eigen
        ${apriltag_LIBRARIES}
        ${OpenCV_LIBS}
        ${YAML_CPP_LIBRARIES}
)
# For inexplicable reasons, the docker build-stage build will fail if we do not include this link directory...
# Without this it works fine in CLion, but when we build it manually it gives us problems. I should note that I
# diverged from the original "tip" about using april grid in cmake found here (https://github.com/amy-tabb/apriltag-cmake-example/tree/master).
# Because I only wanted to use target specific commands, I omitted the link_directories() command and instead add
# the april grid libraries via a target_link_libraries() command to our library.
target_link_directories(${PROJECT_NAME}  PUBLIC
        ${apriltag_LIBDIR}
)

set(EXAMPLES
        examples/webcam_demo.cpp
)
foreach (EXAMPLE IN LISTS EXAMPLES)
    get_filename_component(EXAMPLE_NAME ${EXAMPLE} NAME_WLE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE})
    target_link_libraries(${EXAMPLE_NAME}
            ${PROJECT_NAME}
    )
endforeach ()

message(STATUS "Starting to FetchContent googletest - this requires a network connection. ")
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)
enable_testing()
include(GoogleTest)

set(TESTS
        src/april_tag_cpp_wrapper.test.cpp
        src/target_extractors.test.cpp
        src/target_generators.test.cpp
        test/target_extraction.test.cpp
        src/utilities.test.cpp
)
foreach (TEST IN LISTS TESTS)
    get_filename_component(TEST_NAME ${TEST} NAME_WLE)
    add_executable(${TEST_NAME} ${TEST})
    target_link_libraries(${TEST_NAME}
            GTest::gtest_main
            ${PROJECT_NAME}
    )
    gtest_discover_tests(${TEST_NAME})
endforeach ()
