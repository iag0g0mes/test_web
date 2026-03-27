import argparse
import os
from typing import Dict, List
from dataclasses import dataclass
import numpy as np

from stable_baselines3 import SAC

from robot_navigation.envs.indoor_env import IndoorEnv, EnvConfig
from robot_navigation.envs.indoor_rewards import RewardConfig
from robot_navigation.envs.utils import print_config_rich
from robot_navigation.envs.ros_interface import RosInterface

@dataclass
class RunningParams:
    model:str = "None"
    episodes:int = 10
    seed:int=0
    logdir:str="~/rl/checkpoints"
    deterministic:bool=True
    env_name:str="smallhouse"
    

def parse_args() -> RunningParams:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-name", type=str, default="smallhouse")
    parser.add_argument("--deterministic", action="store_true")
    
    args, ros_args = parser.parse_known_args()
    
    params = RunningParams()
    params.model = str(args.model)
    params.episodes = int(args.episodes)
    params.seed = int(args.seed)
    params.deterministic = bool(args.deterministic)
    params.env_name = str(args.env_name)
    
    if not os.path.exists(params.model):
        raise FileNotFoundError(f"Model path not found: {params.model}")
    
    params.logdir = os.path.dirname(params.model)
    
    print_config_rich(params)
    
    return params, ros_args

def main():
    
    print("[Reinforcement Learning Navigation][eval] running...")
    params, ros_args = parse_args()

    print("[Reinforcement Learning Navigation][eval] creating ROS interface...")
    ros = RosInterface(args=ros_args)
    
    print("[Reinforcement Learning Navigation][eval] creating Env interface...")
    env = None
    match params.env_name:
        case "smallhouse" | "bookstore":
            env = IndoorEnv(
                    ros    = ros,
                    params = params
                )
        case _:
            raise ValueError(f"Environment type unknown!! {params.env_name}")
    
    assert env is not None, f"Failed to create the environment: {params.env_name}"
    
    
    print("[Reinforcement Learning Navigation][eval] loading SAC...")
    model = SAC.load(params.model, device="auto")

    print("[Reinforcement Learning Navigation][eval] creating logger...")
    csv_path = os.path.join(params.logdir, "eval_metrics.csv")
    print(f"[Reinforcement Learning Navigation][eval] logger path: {csv_path}")
    
    rows: List[str] = []
    rows.append("episode,episode_reward,episode_len,success,collision,mean_dist,final_dist\n")


    print("[Reinforcement Learning Navigation][eval] running test...")
    successes = 0
    collisions = 0
    rewards = []
    
    try:
        for ep in range(1, params.episodes + 1):
            obs, info = env.reset(seed=params.seed + ep)
            done = False
            ep_r = 0.0
            ep_l = 0

            while not done:
                action, _ = model.predict(obs, deterministic=params.deterministic)
                obs, r, terminated, truncated, info = env.step(action)
                ep_r += float(r)
                ep_l += 1
                done = bool(terminated or truncated)

            success = int(bool(info.get("is_success", False)))
            collision = int(bool(info.get("is_collision", False)))
            mean_dist = float(info.get("mean_dist", np.nan))
            final_dist = float(info.get("final_dist", np.nan))

            successes += success
            collisions += collision
            rewards.append(ep_r)

            rows.append(
                f"{ep},{ep_r:.6f},{ep_l},{success},{collision},{mean_dist:.6f},{final_dist:.6f}\n"
            )
            print(f"[Reinforcement Learning Navigation][eval] ep={ep} R={ep_r:.2f} L={ep_l} success={success} collision={collision}")

        with open(csv_path, "w", encoding="utf-8") as f:
            f.writelines(rows)

        summary = {
            "episodes": params.episodes,
            "success_rate": float(successes / params.episodes),
            "collision_rate": float(collisions / params.episodes),
            "mean_reward": float(np.mean(np.array(rewards, dtype=np.float32))) if rewards else 0.0,
        }
        summary_path = os.path.join(params.logdir, "summary_eval.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")

        print(f"[Reinforcement Learning Navigation][eval] Wrote: {csv_path}")
        print(f"[Reinforcement Learning Navigation][eval] Wrote: {summary_path}")
        
    finally:
        env.close()
        

if __name__ == "__main__":
    main()