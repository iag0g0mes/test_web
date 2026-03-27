from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    # positive reinforcement
    goal_bonus: float = 30.0
    progress_scale: float = 3.0
    
    # penalty (negative) reinforcement
    time_penalty: float = -0.01
    collision_penalty: float = -15.0
    ang_vel_penalty_scale: float = -0.01
    reverse_vel_penalty: float = -0.05
    obstacle_penalty_scale:float=-1.0
    
    # params
    safe_dist: float = 0.50

def compute_reward(cfg: RewardConfig, 
                   prev_dist: float, 
                   dist: float, 
                   min_scan:float,
                   ang_vel: float,
                   lon_vel: float, 
                   collision: bool, 
                   success: bool) -> float:
    reward = 0.0
    reward += cfg.progress_scale * (prev_dist - dist)
    reward += cfg.time_penalty
    reward += cfg.ang_vel_penalty_scale * abs(ang_vel)
    
    if min_scan < cfg.safe_dist:
        reward += cfg.obstacle_penalty_scale * (cfg.safe_dist - min_scan) / cfg.safe_dist
    
    if lon_vel < 0.0:
        reward += cfg.reverse_vel_penalty * abs(lon_vel)

    if collision:
        reward += cfg.collision_penalty
    if success:
        reward += cfg.goal_bonus

    return float(reward)