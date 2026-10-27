#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NODE_ARGS=("$@")
set --
source /opt/ros/noetic/setup.bash
source "$HOME/haptic_ws/devel/setup.bash"
exec python3 "$SCRIPT_DIR/lidar_node.py" "${NODE_ARGS[@]}"
