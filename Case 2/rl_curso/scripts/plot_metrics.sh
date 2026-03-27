#!/usr/bin/env bash
set -euo pipefail

source "$HOME/venvs/robot_nav_rl/bin/activate"

python -m robot_navigation.scripts.plot_metrics \
    --window 20 \
    --logdir runs/train/smallhouse/20260320-011951 \
    --env-name smallhouse \
