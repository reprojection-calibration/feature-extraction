FROM ubuntu:24.04 AS configuration-stage

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]
WORKDIR /buildroot


FROM configuration-stage AS base-stage

ARG INSTALL_BUILD_DEPENDENCIES=building/install_scripts/install_build_dependencies.sh
RUN --mount=type=bind,source=${INSTALL_BUILD_DEPENDENCIES},target=/temporary/${INSTALL_BUILD_DEPENDENCIES} \
    /temporary/${INSTALL_BUILD_DEPENDENCIES}

ARG INSTALL_LIBRARY_DEPENDENCIES=building/install_scripts/install_library_dependencies.sh
RUN --mount=type=bind,source=${INSTALL_LIBRARY_DEPENDENCIES},target=/temporary/${INSTALL_LIBRARY_DEPENDENCIES} \
    /temporary/${INSTALL_LIBRARY_DEPENDENCIES}

ARG INSTALL_APRIL_TAGS=building/install_scripts/install_april_tags.sh
RUN --mount=type=bind,source=${INSTALL_APRIL_TAGS},target=/temporary/${INSTALL_APRIL_TAGS} \
    /temporary/${INSTALL_APRIL_TAGS}

FROM configuration-stage AS code-checks-stage

ARG INSTALL_CODE_CHECK_DEPENDENCIES=building/install_scripts/install_code_check_dependencies.sh
RUN --mount=type=bind,source=${INSTALL_CODE_CHECK_DEPENDENCIES},target=/temporary/${INSTALL_CODE_CHECK_DEPENDENCIES} \
    /temporary/${INSTALL_CODE_CHECK_DEPENDENCIES}

ARG CODE_CHECKS=building/scripts/code_checks.sh
RUN --mount=type=bind,source=${CODE_CHECKS},target=/temporary/${CODE_CHECKS} \
    --mount=type=bind,source=.clang-format,target=/temporary/.clang-format \
    --mount=type=bind,source=building/,target=/temporary/building \
    --mount=type=bind,source=code/,target=/temporary/code \
    /temporary/${CODE_CHECKS}

FROM base-stage AS build-stage

ARG BUILD_LIBRARY=building/scripts/build_library.sh
RUN --mount=type=bind,source=${BUILD_LIBRARY},target=/temporary/${BUILD_LIBRARY} \
    --mount=type=bind,source=code/,target=/temporary/code \
    /temporary/${BUILD_LIBRARY}

ENTRYPOINT ["/buildroot/build/webcam_demo", "-c", "/temporary/code/examples/target_config.yaml"]

FROM base-stage AS docs-stage

ARG INSTALL_DOXYGEN=building/install_scripts/install_doxygen.sh
RUN --mount=type=bind,source=${INSTALL_DOXYGEN},target=/temporary/${INSTALL_DOXYGEN} \
        /temporary/${INSTALL_DOXYGEN}

RUN --mount=type=bind,source=code/,target=/temporary/code \
    cmake -B "/buildroot/build-docs" -S "/temporary/code" \
        -DBUILD_DOCS=ON; \
    cmake --build "/buildroot/build-docs" --target docs


FROM nginx:1.29.2 AS docs-report-stage

# WARN(Jack): Hardcoding the doc version here seems hacky. If we just use this for local development it can work but I
# would rather have something like a version env var.
COPY --from=docs-stage  /buildroot/build-docs/docs/0.0.0  /usr/share/nginx/html

FROM base-stage AS development-stage

RUN apt-get update; apt-get install --no-install-recommends --yes gdb; rm --force --recursive /var/lib/apt/lists/*