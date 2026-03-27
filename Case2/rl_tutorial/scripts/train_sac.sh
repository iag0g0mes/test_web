#!/usr/bin/env bash
set -euo pipefail

source "$HOME/venvs/robot_nav_rl/bin/activate"

python -m robot_navigation.scripts.train_sac\
    --total-timesteps 800000\
    --seed 1\
    --logdir runs/train\
    --checkpoint-freq 100000\
    --env-name smallhouse\
    --learning-starts 20000\
    --lr 0.0001\
    --buffer-size 400000\
    --batch-size 256\
    --ros-args --params-file $(ros2 pkg prefix --share robot_navigation)/params/ros.yaml
