#!/usr/bin/env bash
# Download and unzip a dataset (Kaggle copy) into data/raw/<dataset>/.
#
#   scripts/download_data.sh [inbreast | cbis-ddsm | all]    (default: inbreast)
#
# Safe to run again: it skips the download when the zip is already complete and
# the unzip when the dataset is already extracted. An interrupted download is
# resumed, a few times automatically and otherwise by running the script again.
# The data is not part of this repository; do not redistribute it (the license
# of the INbreast copy is unclear).
#
# Environment variables (mainly for testing):
#   DATA_RAW_DIR            parent of the dataset folders (default: <repo>/data/raw)
#   DOWNLOAD_ATTEMPTS       tries of a download before giving up (default: 5)
#   DOWNLOAD_RETRY_SECONDS  wait between the tries (default: 10)
#   INBREAST_URL, CBIS_DDSM_URL                    download URLs
#   INBREAST_ZIP_BYTES, CBIS_DDSM_ZIP_BYTES        expected sizes of the zips in bytes
#   INBREAST_DCM_COUNT, CBIS_DDSM_JPEG_COUNT       expected number of images after unzipping

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_root="${DATA_RAW_DIR:-$repo_root/data/raw}"
attempts="${DOWNLOAD_ATTEMPTS:-5}"
retry_seconds="${DOWNLOAD_RETRY_SECONDS:-10}"

usage() {
    echo "usage: $(basename "$0") [inbreast | cbis-ddsm | all]  (default: inbreast)"
}

if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi
case "${1:-inbreast}" in
    inbreast | cbis-ddsm) datasets=("${1:-inbreast}") ;;
    all) datasets=(inbreast cbis-ddsm) ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        echo "error: unknown dataset '$1'" >&2
        usage >&2
        exit 2
        ;;
esac

for tool in curl unzip; do
    if ! command -v "$tool" > /dev/null; then
        echo "error: '$tool' is required but not installed" >&2
        exit 1
    fi
done

# The settings of one dataset: where its images are after unzipping, and how
# many there are (at the given depth below that folder).
configure() {
    case "$1" in
        inbreast)
            label="INbreast"
            raw_dir="$raw_root/inbreast"
            url="${INBREAST_URL:-https://www.kaggle.com/api/v1/datasets/download/ramanathansp20/inbreast-dataset}"
            zip_path="$raw_dir/inbreast-dataset.zip"
            expected_bytes="${INBREAST_ZIP_BYTES:-2074630259}"
            image_dir="$raw_dir/INbreast Release 1.0/AllDICOMs"
            image_pattern='*.dcm'
            image_depth=1
            expected_images="${INBREAST_DCM_COUNT:-410}"
            image_kind="DICOM files"
            ;;
        cbis-ddsm)
            label="CBIS-DDSM"
            raw_dir="$raw_root/cbis-ddsm"
            url="${CBIS_DDSM_URL:-https://www.kaggle.com/api/v1/datasets/download/awsaf49/cbis-ddsm-breast-cancer-image-dataset}"
            zip_path="$raw_dir/cbis-ddsm-breast-cancer-image-dataset.zip"
            expected_bytes="${CBIS_DDSM_ZIP_BYTES:-5318997088}"
            image_dir="$raw_dir/jpeg" # jpeg/<series>/<image>.jpg
            image_pattern='*.jpg'
            image_depth=2
            expected_images="${CBIS_DDSM_JPEG_COUNT:-10237}"
            image_kind="JPEG files"
            ;;
    esac
}

# Prints 0 when the folder does not exist (find fails, which pipefail would
# otherwise turn into a silent exit).
count_images() {
    {
        find "$image_dir" -mindepth "$image_depth" -maxdepth "$image_depth" \
            -name "$image_pattern" 2> /dev/null || true
    } | wc -l
}

file_size() {
    stat -c %s "$1"
}

fetch() {
    local attempt
    for ((attempt = 1; attempt <= attempts; attempt++)); do
        # -C - resumes a partial download, -f fails on HTTP errors, -L follows redirects.
        if curl --fail --location --continue-at - --output "$zip_path" "$url"; then
            return 0
        fi
        if ((attempt < attempts)); then
            echo "Download interrupted (try $attempt of $attempts), resuming in $retry_seconds s" >&2
            sleep "$retry_seconds"
        fi
    done
    echo "error: the download failed $attempts times; run the script again to resume it." >&2
    return 1
}

download() {
    configure "$1"
    if [[ "$(count_images)" -eq "$expected_images" ]]; then
        echo "$label is already in $raw_dir ($expected_images $image_kind), nothing to do."
        return 0
    fi

    mkdir -p "$raw_dir"

    if [[ -f "$zip_path" && "$(file_size "$zip_path")" -eq "$expected_bytes" ]]; then
        echo "Zip already downloaded: $zip_path"
    else
        if [[ -f "$zip_path" && "$(file_size "$zip_path")" -gt "$expected_bytes" ]]; then
            echo "error: $zip_path is larger than the expected $expected_bytes bytes." >&2
            echo "The dataset may have changed; the file was kept." >&2
            return 1
        fi
        echo "Downloading $label (about $((expected_bytes / 1000000)) MB) to $zip_path"
        fetch
        local actual_bytes
        actual_bytes="$(file_size "$zip_path")"
        if [[ "$actual_bytes" -ne "$expected_bytes" ]]; then
            echo "error: $zip_path is $actual_bytes bytes, expected $expected_bytes." >&2
            echo "The download is incomplete or the dataset changed; the file was kept." >&2
            return 1
        fi
    fi

    echo "Unzipping into $raw_dir"
    unzip -q -o "$zip_path" -d "$raw_dir"

    local found
    found="$(count_images)"
    if [[ "$found" -ne "$expected_images" ]]; then
        echo "error: found $found $image_kind after unzipping, expected $expected_images." >&2
        return 1
    fi

    echo "Done: $found $image_kind in $image_dir"
    echo "The zip is kept at $zip_path; delete it to free about $((expected_bytes / 1000000)) MB."
}

for dataset in "${datasets[@]}"; do
    download "$dataset"
done
