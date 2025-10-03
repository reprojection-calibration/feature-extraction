#!/bin/bash

set -eou pipefail

SCRIPT_FOLDER="$(dirname "$(realpath -s "$0")")"
TAG=feature-extraction:release

echo "Running container from image '$TAG'..."
xhost +
docker run --name webcam_demo --device=/dev/video0:/dev/video0 -e DISPLAY=:0.0 -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev:/dev --privileged --rm --volume ${SCRIPT_FOLDER}/../../:/temporary ${TAG}