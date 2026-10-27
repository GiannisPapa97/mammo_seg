#!/usr/bin/env bash
# Download and unzip the INbreast dataset (Kaggle copy) into data/raw/inbreast/.
#
# Safe to run again: it skips the download when the zip is already complete and
# the unzip when the dataset is already extracted. The data is not part of this
# repository and its license is unclear, so do not redistribute it.
#
# Environment variables (mainly for testing):
#   DATA_RAW_DIR          target directory (default: <repo>/data/raw/inbreast)
#   INBREAST_URL          download URL
#   INBREAST_ZIP_BYTES    expected size of the zip in bytes
#   INBREAST_DCM_COUNT    expected number of DICOM files after unzipping

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_dir="${DATA_RAW_DIR:-$repo_root/data/raw/inbreast}"
url="${INBREAST_URL:-https://www.kaggle.com/api/v1/datasets/download/ramanathansp20/inbreast-dataset}"
expected_bytes="${INBREAST_ZIP_BYTES:-2074630259}"
expected_dcm="${INBREAST_DCM_COUNT:-410}"

zip_path="$raw_dir/inbreast-dataset.zip"
dicom_dir="$raw_dir/INbreast Release 1.0/AllDICOMs"

for tool in curl unzip; do
    if ! command -v "$tool" > /dev/null; then
        echo "error: '$tool' is required but not installed" >&2
        exit 1
    fi
done

# Prints 0 when the folder does not exist (find fails, which pipefail would
# otherwise turn into a silent exit).
count_dicoms() {
    { find "$dicom_dir" -maxdepth 1 -name '*.dcm' 2> /dev/null || true; } | wc -l
}

file_size() {
    stat -c %s "$1"
}

if [[ -d "$dicom_dir" && "$(count_dicoms)" -eq "$expected_dcm" ]]; then
    echo "INbreast is already in $raw_dir ($expected_dcm DICOM files), nothing to do."
    exit 0
fi

mkdir -p "$raw_dir"

if [[ -f "$zip_path" && "$(file_size "$zip_path")" -eq "$expected_bytes" ]]; then
    echo "Zip already downloaded: $zip_path"
else
    echo "Downloading INbreast (about $((expected_bytes / 1000000)) MB) to $zip_path"
    # -C - resumes a partial download, -f fails on HTTP errors, -L follows redirects.
    curl --fail --location --continue-at - --output "$zip_path" "$url"
    actual_bytes="$(file_size "$zip_path")"
    if [[ "$actual_bytes" -ne "$expected_bytes" ]]; then
        echo "error: $zip_path is $actual_bytes bytes, expected $expected_bytes." >&2
        echo "The download is incomplete or the dataset changed; the file was kept." >&2
        exit 1
    fi
fi

echo "Unzipping into $raw_dir"
unzip -q -o "$zip_path" -d "$raw_dir"

found="$(count_dicoms)"
if [[ "$found" -ne "$expected_dcm" ]]; then
    echo "error: found $found DICOM files after unzipping, expected $expected_dcm." >&2
    exit 1
fi

echo "Done: $found DICOM files in $dicom_dir"
echo "The zip is kept at $zip_path; delete it to free about $((expected_bytes / 1000000)) MB."
