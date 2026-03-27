import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import yaml
import os

import gymnasium as gym
import numpy as np

from .ros_interface import RosInterface
from .indoor_rewards import RewardConfig, compute_reward
from .utils import (wrap_angle, 
                    downsample_scan, 
                    sanitize_scan, 
                    tuple_goals, 
                    tuple_starts,
                    print_config_rich)



@dataclass(frozen=True)
class EnvConfig:
    dt: float = 0.1
    max_steps: int = 300
    n_scan: int = 72

    v_max: float = 0.6
    w_max: float = 1.5

    goal_threshold: float = 0.35
    collision_threshold: float = 0.18

    scan_range_min: float = 0.12
    scan_range_max: float = 5.0

    max_goal_dist: float = 50.0
    min_goal_dist: float = 3.0

    robot_name: str = "robot"
    
    starts_path:str = "starts.yaml"
    goals_path:str = "starts.yaml"
    
    # (x, y, yaw)
    starts: Tuple[Tuple[float, float, float], ...] =\
        ((0.0, 0.0, 0.0),)

    # (x, y)
    goals: Tuple[Tuple[float, float], ...] =\
        ((4.0, 4.0),)


class IndoorEnv(gym.Env):
    """
    Gymnasium Env for ROS2+Gazebo navigation:
      action: continuous (v, w) via /cmd_vel
      obs: downsampled /scan + goal (dist + sin/cos heading) + (v,w)
      reward: progress-to-goal + time penalty + collision penalty + goal bonus
    """

    metadata = {"render_modes": []}

    def __init__(self, ros:RosInterface, params):
        super().__init__()

        self.ros = ros
        self.params = params
        
        self.np_random = np.random.default_rng(self.params.seed)
        
        self.env_cfg, self.reward_cfg = self._load_configs(self.ros, self.params)
        
        print_config_rich(self.reward_cfg)      
        print_config_rich(self.env_cfg)   
        
        # 1) LiDAR points in Scan format (distance of each point)
        # 2) distance to the goal
        # 3) sin(angle to the goal)
        # 4) cos(angle to the goal)
        # 5) linear velocity
        # 6) angular velocity
        obs_dim = self.env_cfg.n_scan + 5  # scan + [dist_norm, sin(angle), cos(angle), v, w]
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        
        # linear velocity + angular velocity
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._goal_xy: Optional[Tuple[float, float]] = None
        self._prev_dist: float = 0.0
        self._step: int = 0
        self._dist_hist: List[float] = []

        self.ros.wait_for_topics(timeout_sec=15.0)
    
    def _load_configs(self, ros:RosInterface, params:Any) -> Tuple[EnvConfig, RewardConfig]:
        
        env_file_path = self.ros.params.env_config
        
        if not os.path.exists(env_file_path):
            raise FileNotFoundError(f"Environment config file not found at: {env_file_path}")
        
        with open(env_file_path, "r", encoding="utf-8") as f:
            config: Dict[str, Any] = yaml.safe_load(f) or {}
        
        env_dict = config.get("environment", {})
        
        if "starts_path" in env_dict:
            starts_path = os.path.join(ros.share_directory, env_dict["starts_path"])
            with open(starts_path, "r", encoding="utf-8") as f:
                env_dict["starts"] = tuple_starts(yaml.safe_load(f)["waypoints"] or {})
                
        if "goals_path" in env_dict:
            goals_path = os.path.join(ros.share_directory, env_dict["goals_path"])
            with open(goals_path, "r", encoding="utf-8") as f:
                env_dict["goals"] = tuple_goals(yaml.safe_load(f)["waypoints"] or {})
        
        env_cfg = EnvConfig(**env_dict)
        
        reward_dict = config.get("reward", {})
        reward_cfg = RewardConfig(**reward_dict)
    
        return env_cfg, reward_cfg
    
    def _sample_start(self) -> Tuple[float, float, float]:
        """
            randomly sample a start position from a list of possible starts in env_config

        Returns:
            Tuple[float, float, float]: (x, y, yaw) start of the robot
        """
        
        candidates = list(self.env_cfg.starts)
        self.np_random.shuffle(candidates)            
        return candidates[0]
    

    def _sample_goal(self, robot_xy: Tuple[float, float]) -> Tuple[float, float]:
        """
            randomly sample a goal from a list of possible goals in env_config
        
        - condition: the distance between the current position (start) and the goal
                        must be greater than a threshold (min_goal_dist) in env_config

        Args:
            robot_xy (Tuple[float, float]): current position of the robot

        Returns:
            Tuple[float, float]: (x, y) goal of the robot
        """
        rx, ry = robot_xy
        candidates = list(self.env_cfg.goals)
        self.np_random.shuffle(candidates)
        
        for gx, gy in candidates:
            d = math.hypot(gx - rx, gy - ry)
            if (d > self.env_cfg.min_goal_dist) and\
               (d < self.env_cfg.max_goal_dist):
                return (gx, gy)
            
        return candidates[0]

    def _compute_goal_features(self, robot_x: float, robot_y: float, robot_yaw: float) -> Tuple[float, float, float]:
        """
            computes the distance and angle to the goal
        
        - condition: the goal must exist (self._goal_xy is not None)

        Args:
            robot_x (float): position in x-axis
            robot_y (float): position in y-axis
            robot_yaw (float): orientation in z-axis (yaw)

        Returns:
            Tuple[float, float, float]: distance, normalized_distance, and angle to the goal
        """
        assert self._goal_xy is not None
        
        # goal pose
        gx, gy = self._goal_xy
        
        # 2) distance to goal
        dx = gx - robot_x
        dy = gy - robot_y
        dist = math.hypot(dx, dy)
        
        # 3 and 4) angle to the goal
        heading = wrap_angle(math.atan2(dy, dx) - robot_yaw)
        
        # 2) normalization of the distance
        dist_norm = float(np.clip(dist / self.env_cfg.max_goal_dist, 0.0, 1.0))
        
        return dist, dist_norm, heading

    def _get_obs(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
            get the observations from the ros_interface, and preprocess them

        - LiDAR scan:
            + sanitization
            + downsample
        - Odometry
            + convert to distance and angle to the goal
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: observation vector and info (logger)
        """
        scan = self.ros.get_scan()
        odom = self.ros.get_odom()

        # scan to ROI (Region of Interest)
        ranges = sanitize_scan(scan, self.env_cfg.scan_range_min, self.env_cfg.scan_range_max)
        
        # downsampling
        ranges_ds = downsample_scan(ranges, self.env_cfg.n_scan)
        
        # normalize
        ranges_norm = (ranges_ds / self.env_cfg.scan_range_max).astype(np.float32)

        # distance + angle to the goal
        dist, dist_norm, heading = self._compute_goal_features(odom.x, odom.y, odom.yaw)

        # observation
        obs = np.concatenate(
            [
                ranges_norm,
                np.array(
                    [
                        dist_norm,
                        math.sin(heading),
                        math.cos(heading),
                        np.clip(odom.v / self.env_cfg.v_max, -1.0, 1.0),
                        np.clip(odom.w / self.env_cfg.w_max, -1.0, 1.0),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

        # logger
        info = {
            "dist_to_goal": float(dist),
            "min_scan": float(np.min(ranges_ds)),
            "robot_x": float(odom.x),
            "robot_y": float(odom.y),
            "robot_yaw": float(odom.yaw),
            "goal_x": float(self._goal_xy[0]) if self._goal_xy else 0.0,
            "goal_y": float(self._goal_xy[1]) if self._goal_xy else 0.0,
            "v": float(odom.v),
            "w": float(odom.w),
        }
        return obs, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
            reset the current episode and start a new one
        
        - actions:
        1) stop the robot
        2) sample new start position
        3) reset simulation
        4) set new start position
        5) wait simulation physics settle
        6) sample new goal
        7) first observation
        
        Args:
            seed (Optional[int], optional): seed for random sampling. Defaults to None.
            options (Optional[dict], optional): optinial variables (logger). Defaults to None.

        Returns:
            Tuple[nd.array, dict]: first observation of new episode, additional info (logger)
        """
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # stop the current robot
        self.ros.stop_robot()
        
        # 1) reset the world

        # 2) try to sample a safe start
        
        # 3) let physics settle
        
        # 4) new goal
       
        # 5) new observation + logger
        
        return obs, info

    def step(self, action: np.ndarray):
        """
            individual step for a episode
        
        - actios:
        1) get new action
        2) send it to the robot
        3) wait a little
        4) calculate the reward
        5) logger
        6) check if the robot reach any stop condition 
            + collision
            + reach the goal (success)

        Args:
            action (np.ndarray): [linear velocity, angular velocity]

        Returns:
            Tuple[np.ndarray, float, bool, bool dict]: results of the step
        """
        self._step += 1

        # sanitization of actions
        a = np.clip(np.array(action, dtype=np.float32), -1.0, 1.0)
        
        # 1) normalziation


        # 2) publish action

        # 3) get new observation

        # 4) reward

        # distance
        
        # collision
        
        # success

        # 5) estimate the reward
        
        # 6) logger

        # 7) stop conditions
        
        return obs, float(reward), terminated, early_stop, info

    def close(self):
        """
            end environment
        """
        try:
            self.ros.close()
        finally:
            pass