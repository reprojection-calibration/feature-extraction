cmake_minimum_required(VERSION 3.20)

project(
        feature_extraction
        VERSION "0.0.0"
        LANGUAGES C CXX
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(apriltag REQUIRED)


set(SRC_FILES
        src/april_tag_cpp_wrapper.cpp
        src/generated_apriltag_code/tagCustom36h11.c
        src/target_extraction.cpp
        src/target_extractors.cpp
        src/target_generators.cpp
        src/utilities.cpp
)
set(INCLUDE_DIRECTORIES
        ${OpenCV_INCLUDE_DIRS}
)
set(PRIVATE_LINK_LIBRARIES
        apriltag::apriltag
        eigen_utilities
        types_internal
)

add_library(${PROJECT_NAME} SHARED
        ${SRC_FILES}
)
target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
        $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
target_include_directories(${PROJECT_NAME} PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>
        ${INCLUDE_DIRECTORIES}
)
target_link_libraries(${PROJECT_NAME}
        PRIVATE
            ${PRIVATE_LINK_LIBRARIES}
        PUBLIC
            ${OpenCV_LIBS}
)

# For inexplicable reasons, the docker build-stage build will fail if we do not include this link directory...
# Without this it works fine in CLion, but when we build it manually it gives us problems. I should note that I
# diverged from the original "tip" about using april grid in cmake found here (https://github.com/amy-tabb/apriltag-cmake-example/tree/master).
# Because I only wanted to use target specific commands, I omitted the link_directories() command and instead add
# the april grid libraries via a target_link_libraries() command to our library.
target_link_directories(${PROJECT_NAME} PUBLIC
        ${apriltag_LIBDIR}
)

set(TESTS
        src/april_tag_cpp_wrapper.test.cpp
        src/target_extractors.test.cpp
        src/target_generators.test.cpp
        test/target_extraction.test.cpp
        src/utilities.test.cpp
)

if (BUILD_TESTING)
    foreach (TEST IN LISTS TESTS)
        get_filename_component(TEST_NAME ${TEST} NAME_WLE)
        set(TEST_NAME ${PROJECT_NAME}.${TEST_NAME})

        add_executable(${TEST_NAME} ${TEST})
        target_include_directories(${TEST_NAME} PRIVATE
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>
                ${INCLUDE_DIRECTORIES}
        )
        target_link_libraries(${TEST_NAME} PRIVATE
                GTest::gtest_main
                ${PROJECT_NAME}
                ${PRIVATE_LINK_LIBRARIES}
        )

        if (REPROJECTION_ENABLE_COVERAGE)
            target_compile_options(${TEST_NAME} PRIVATE --coverage)
            target_link_options(${TEST_NAME} PRIVATE --coverage)
        endif ()

        gtest_discover_tests(${TEST_NAME})
    endforeach ()
endif ()