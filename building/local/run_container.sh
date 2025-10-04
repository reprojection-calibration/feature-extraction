#!/bin/bash

set -eou pipefail

# NOTE(Jack): This usage instruction is not fully consistent with the run scripts usage because it was copy and pasted from the build script.
usage() {
    echo "Usage: $0 -t <target-stage>"
    echo "  -t <target-stage>     : Target build stage (e.g., build, development)"
    exit 1
}

TARGET_STAGE=development

while getopts ":t:" opt; do
  case ${opt} in
    t ) TARGET_STAGE=$OPTARG ;;
    * ) usage ;;
  esac
done

IMAGE=feature-extraction
TAG=${IMAGE}:${TARGET_STAGE}

echo "Running container from image with tag '$TAG'..."
docker run --entrypoint="" --interactive --rm --tty "${TAG}" /bin/bash