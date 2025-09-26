#!/bin/bash

set -eoux pipefail

BUILD_DIRECTORY=/buildroot/build

ls  /usr/lib/x86_64-linux-gnu/

cmake -B "${BUILD_DIRECTORY}" -G Ninja -S /temporary/code
cmake --build "${BUILD_DIRECTORY}"

ctest --output-on-failure --progress --test-dir "${BUILD_DIRECTORY}"