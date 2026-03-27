import argparse
import csv
import os
from typing import List, Dict
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from robot_navigation.envs.utils import print_config_rich
@dataclass
class RunningParams:
    logdir:str="~/rl/checkpoints"
    window:int=3
    env_name:str="smallhouse"

def parse_args() -> RunningParams:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--logdir",    type=str, default="~/runs/eval")
    parser.add_argument("--window",    type=int, default=3)
    parser.add_argument("--env-name",  type=str, default="smallhouse")
    
    args = parser.parse_args()
    
    params = RunningParams()
    params.logdir = str(args.logdir)
    params.env_name = str(args.env_name)
    params.window = int(args.window)
    
    print_config_rich(params)
    
    return params
    
    
def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) == 0:
        return x
    w = max(1, int(w))
    if len(x) < w:
        return np.ones_like(x) * np.mean(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def load_metrics(csv_path: str) -> Dict[str, np.ndarray]:
    cols: Dict[str, List[float]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return {k: np.array(v, dtype=np.float32) for k, v in cols.items()}


def main():
    params = parse_args()

    metrics_path = os.path.join(params.logdir, "metrics.csv")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Missing: {metrics_path}")

    m = load_metrics(metrics_path)
    steps = m["global_step"]
    rewards = m["episode_reward"]
    success = m["success"]
    collision = m["collision"]
    ep_len = m["episode_len"]

    out_dir = os.path.join(params.logdir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Success rate vs steps (smoothed)
    sr = moving_average(success, params.window)
    sr_steps = steps[len(steps) - len(sr):]
    plt.figure()
    plt.plot(sr_steps, sr)
    plt.xlabel("env steps")
    plt.ylabel(f"success rate (MA{params.window})")
    plt.title("Success rate vs steps")
    plt.grid(True)
    p1 = os.path.join(out_dir, "success_rate_vs_steps.png")
    plt.savefig(p1, dpi=160, bbox_inches="tight")
    plt.close()

    # Reward vs steps (smoothed)
    rw = moving_average(rewards, params.window)
    rw_steps = steps[len(steps) - len(rw):]
    plt.figure()
    plt.plot(rw_steps, rw)
    plt.xlabel("env steps")
    plt.ylabel(f"episode reward (MA{params.window})")
    plt.title("Reward vs steps")
    plt.grid(True)
    p2 = os.path.join(out_dir, "reward_vs_steps.png")
    plt.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close()

    # Additional metric: collision rate vs steps (smoothed)
    cr = moving_average(collision, params.window)
    cr_steps = steps[len(steps) - len(cr):]
    plt.figure()
    plt.plot(cr_steps, cr)
    plt.xlabel("env steps")
    plt.ylabel(f"collision rate (MA{params.window})")
    plt.title("Collision rate vs steps")
    plt.grid(True)
    p3 = os.path.join(out_dir, "collision_rate_vs_steps.png")
    plt.savefig(p3, dpi=160, bbox_inches="tight")
    plt.close()

    # episode length vs steps
    el = moving_average(ep_len, params.window)
    el_steps = steps[len(steps) - len(el):]
    plt.figure()
    plt.plot(el_steps, el)
    plt.xlabel("env steps")
    plt.ylabel(f"episode length (MA{params.window})")
    plt.title("Episode length vs steps")
    plt.grid(True)
    p4 = os.path.join(out_dir, "episode_len_vs_steps.png")
    plt.savefig(p4, dpi=160, bbox_inches="tight")
    plt.close()

    print("[plot] wrote:")
    print(" ", p1)
    print(" ", p2)
    print(" ", p3)
    print(" ", p4)


if __name__ == "__main__":
    main()