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

    # 1) progresso

    # 2) penalidade de tempo

    # 3) penalidade de velocidade angular

    # 4) penalidade de distancia para obstaculos

    # 5) colisao

    # 6) sucesso

    return float(reward)