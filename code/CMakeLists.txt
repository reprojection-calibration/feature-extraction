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
        Eigen3::Eigen
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


install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION include/${PROJECT_NAME}
)
install(TARGETS ${PROJECT_NAME}
        EXPORT featureExtractionTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

install(EXPORT featureExtractionTargets
        FILE feature_extractionTargets.cmake
        NAMESPACE reprojection::
        DESTINATION lib/cmake/feature_extraction
)
install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/feature_extractionConfig.cmake
        DESTINATION lib/cmake/feature_extraction
)

include(CTest)
if (BUILD_TESTING)
    find_package(GTest REQUIRED)
endif ()

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

        gtest_discover_tests(${TEST_NAME})
    endforeach ()
endif ()

if (BUILD_DOCS)
    find_package(Doxygen REQUIRED)

    set(DOXYGEN_ALWAYS_DETAILED_SEC YES) # Required to display the inline code when their is only a brief descriptor.
    set(DOXYGEN_HIDE_UNDOC_MEMBERS YES)
    set(DOXYGEN_HIDE_UNDOC_CLASSES YES)
    set(DOXYGEN_HTML_OUTPUT "${PROJECT_VERSION}")
    set(DOXYGEN_OUTPUT_DIRECTORY "docs")
    set(DOXYGEN_WARN_IF_UNDOCUMENTED "NO")
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_SOURCE_DIR}/README.md")
    set(DOXYGEN_PROJECT_NAME "Feature Extraction")

    doxygen_add_docs(
            docs
            "${PROJECT_SOURCE_DIR}"
    )
endif ()