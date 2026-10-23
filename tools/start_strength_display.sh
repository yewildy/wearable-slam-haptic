#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/noetic/setup.bash
source "$HOME/ws_livox/devel/setup.bash"
source "$HOME/haptic_ws/devel/setup.bash"
exec python3 "$SCRIPT_DIR/start_strength_display.py" "$@"
