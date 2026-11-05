#!/bin/bash

set -eoux pipefail

# NOTE(Jack): Copy and pasted from reprojection
download_and_extract() {
    local url=$1
    local dest=$2
    local sha256=$3
    local strip_components=${4:-0}

    local file
    file="${dest}/$(basename "${url}")"

    wget --directory-prefix="${dest}" "${url}"
    echo "${sha256}  ${file}" | sha256sum --check --status
    tar -xf "${file}" -C "${dest}" --strip-components="${strip_components}"
}

version="1.18.0"
sha256="14fa81bdc34171edb5f1f02b1d60e74802f0439b77fa44e592565d517d72df90"
name="doxygen-${version}.linux.bin.tar.gz"

buildroot="/buildroot"
download_and_extract \
    "https://www.doxygen.nl/files/${name}" \
    "${buildroot}" \
    "${sha256}"

install \
  "${buildroot}/doxygen-${version}/bin/doxygen" \
  /usr/local/bin/doxygen