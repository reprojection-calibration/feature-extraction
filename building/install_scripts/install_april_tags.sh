#!/bin/bash

set -eoux pipefail

git clone https://github.com/AprilRobotics/apriltag.git
cd apriltag
git checkout v3.4.5 # Just the most recent tag - only so that we have a fixed version not because we actually need this one

cmake -B build/ -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build/ --target install
cmake --install build/