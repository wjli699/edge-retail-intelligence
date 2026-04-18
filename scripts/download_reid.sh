#!/usr/bin/env bash
# Download NVIDIA TAO ReIdentificationNet (ResNet-50, Market-1501) from NGC.
# Requires NGC CLI authenticated, or NGC_API_KEY env var.
set -euo pipefail

DEST="$(cd "$(dirname "$0")/../models/reid" && pwd)"
MODEL_VERSION="deployable_v1.0"
ONNX_FILE="resnet50_market1501.onnx"
NGC_MODEL="nvidia/tao/reidentificationnet:${MODEL_VERSION}"
NGC_API="https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/${MODEL_VERSION}/files"

mkdir -p "$DEST"

if [ -f "${DEST}/${ONNX_FILE}" ]; then
  echo "[info] ${ONNX_FILE} already present in ${DEST}, skipping."
  exit 0
fi

if command -v ngc &>/dev/null; then
  echo "[info] Using NGC CLI to download ${NGC_MODEL}"
  ngc registry model download-version "${NGC_MODEL}" \
      --dest "${DEST}" \
      --file "${ONNX_FILE}"
  if [ ! -f "${DEST}/${ONNX_FILE}" ]; then
    FOUND="$(find "${DEST}" -name "${ONNX_FILE}" -type f 2>/dev/null | head -n1)"
    [ -n "${FOUND}" ] && mv -f "${FOUND}" "${DEST}/"
  fi
  find "${DEST}" -mindepth 1 -type d -empty -delete 2>/dev/null || true
  echo "[ok] Model saved to ${DEST}/${ONNX_FILE}"
  exit 0
fi

if [ -z "${NGC_API_KEY:-}" ]; then
  echo "[error] NGC CLI not found and NGC_API_KEY is not set."
  echo "        Install NGC CLI or: export NGC_API_KEY=<key>"
  exit 1
fi

echo "[info] Downloading ${ONNX_FILE} via NGC REST API..."
wget --header="Authorization: ApiKey ${NGC_API_KEY}" \
     --progress=bar:force \
     -O "${DEST}/${ONNX_FILE}" \
     "${NGC_API}/${ONNX_FILE}"

echo "[ok] Model saved to ${DEST}/${ONNX_FILE}"
echo "NOTE: TRT engine will be auto-generated on first run (saved alongside ONNX)."
