#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
REPO_ROOT="${SCRIPT_DIR}/.."
PROFILE_NAME=$1
TRACKER_MODE=$2
BAG_PATH="$(realpath "$3")"
BAG_START=0
if (( $# >= 4 )); then
  BAG_START=$4
fi


BAG_DIR="$(dirname "${BAG_PATH}")"
BAG_NAME="$(basename "${BAG_PATH}")"
CONFIG="$SCRIPT_DIR/../config/${PROFILE_NAME}/estimator_config.yaml"
TEST_ARTIFACT_DIR="${BAG_DIR}/openvins_slg_loop_${BAG_NAME}"
EST_OUT="${TEST_ARTIFACT_DIR}/traj_${TRACKER_MODE}.txt"
PCD_OUT="${TEST_ARTIFACT_DIR}/points_${TRACKER_MODE}.pcd"
TIME_OUT="${TEST_ARTIFACT_DIR}/timing_${TRACKER_MODE}.txt"
EVAL_OUT="${TEST_ARTIFACT_DIR}/metrics_${TRACKER_MODE}.txt"
MP4_OUT="${TEST_ARTIFACT_DIR}/viz_${TRACKER_MODE}.mp4"

if [[ "${TRACKER_MODE}" == "slg" ]]; then
  USE_KLT=false
else
  USE_KLT=true
fi

mkdir -p "${TEST_ARTIFACT_DIR}"

roslaunch ov_lightglue serial_new.launch \
  config:=${PROFILE_NAME} \
  dataset:="${BAG_NAME}" \
  bag:="${BAG_PATH}" \
  config_path:="${CONFIG}" \
  dosave:=true \
  dopcd:=true \
  dotime:=false \
  doeval:=false \
  doviz:=false \
  path_est:="${EST_OUT}" \
  path_pcd:="${PCD_OUT}" \
  path_time:="${TIME_OUT}" \
  path_metrics:="${EVAL_OUT}" \
  path_viz:="${MP4_OUT}" \
  slg_use_gpu:=true \
  bag_start:="${BAG_START}" \
  use_klt:="${USE_KLT}" \
  slg_superpoint_onnx_path:="$REPO_ROOT/onnx/weights_latest/superpoint.onnx" \
  slg_lightglue_onnx_path:="$REPO_ROOT/onnx/weights_latest/superpoint_lightglue_fused_cpu.onnx"
