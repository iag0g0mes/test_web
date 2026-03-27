#!/usr/bin/env bash
set -euo pipefail

source "$HOME/venvs/robot_nav_rl/bin/activate"

python -m robot_navigation.scripts.eval_sac \
    --model runs/train/smallhouse/20260320-011951/final_model.zip \
    --episodes 10 \
    --seed 1 \
    --env-name bookstore \
    --deterministic \
    --ros-args --params-file $(ros2 pkg prefix --share robot_navigation)/params/ros_test.yaml

