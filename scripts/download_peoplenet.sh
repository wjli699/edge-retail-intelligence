#!/usr/bin/env bash
# Download NVIDIA PeopleNet v2.3.3 (INT8 ONNX) from NGC.
# Requires either:
#   (a) NGC CLI installed and authenticated:  ngc config set
#   (b) An NGC API key exported as:           export NGC_API_KEY=<key>
# Free NGC account at https://ngc.nvidia.com

set -euo pipefail

DEST="$(cd "$(dirname "$0")/../models/peoplenet" && pwd)"
MODEL_VERSION="pruned_quantized_decrypted_v2.3.3"
ONNX_FILE="resnet34_peoplenet_int8.onnx"
NGC_MODEL="nvidia/tao/peoplenet:${MODEL_VERSION}"
NGC_API="https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/${MODEL_VERSION}/files"

mkdir -p "$DEST"

if [ -f "${DEST}/${ONNX_FILE}" ]; then
  echo "[info] ${ONNX_FILE} already present in ${DEST}, skipping download."
  exit 0
fi

# --- Attempt 1: NGC CLI -------------------------------------------------
if command -v ngc &>/dev/null; then
  echo "[info] Using NGC CLI to download ${NGC_MODEL}"
  ngc registry model download-version "${NGC_MODEL}" \
      --dest "${DEST}" \
      --file "${ONNX_FILE}"
  # NGC CLI may nest files under a versioned subdirectory (name varies by CLI/NGC)
  if [ ! -f "${DEST}/${ONNX_FILE}" ]; then
    FOUND="$(find "${DEST}" -name "${ONNX_FILE}" -type f 2>/dev/null | head -n1)"
    if [ -n "${FOUND}" ] && [ "${FOUND}" != "${DEST}/${ONNX_FILE}" ]; then
      mv -f "${FOUND}" "${DEST}/"
    fi
  fi
  find "${DEST}" -mindepth 1 -type d -empty -delete 2>/dev/null || true
  echo "[ok] Model saved to ${DEST}/${ONNX_FILE}"
  exit 0
fi

# --- Attempt 2: wget with NGC API key -----------------------------------
if [ -z "${NGC_API_KEY:-}" ]; then
  echo "[error] NGC CLI not found and NGC_API_KEY is not set."
  echo "        Install NGC CLI (https://ngc.nvidia.com/setup/installers/cli)"
  echo "        or export NGC_API_KEY=<your-api-key> and re-run."
  exit 1
fi

echo "[info] Downloading ${ONNX_FILE} via NGC REST API..."
wget --header="Authorization: ApiKey ${NGC_API_KEY}" \
     --progress=bar:force \
     -O "${DEST}/${ONNX_FILE}" \
     "${NGC_API}/${ONNX_FILE}"

echo "[ok] Model saved to ${DEST}/${ONNX_FILE}"
echo ""
echo "NOTE: The TensorRT engine (${ONNX_FILE%.onnx}.engine) will be auto-generated"
echo "      by nvinfer on the first pipeline run. This may take several minutes on"
echo "      Jetson Orin NX — subsequent runs use the cached engine file."
