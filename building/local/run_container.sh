#!/bin/bash

set -eou pipefail

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
SCRIPT_FOLDER="$(dirname "$(realpath -s "$0")")"
TAG=${IMAGE}:${TARGET_STAGE}

echo "Running container of image with tag '$TAG'..."
docker run --entrypoint="" --interactive --rm --tty ${TAG} /bin/bash