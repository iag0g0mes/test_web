
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeMetricsCallback(BaseCallback):
    """
    Writes per-episode metrics to CSV and also logs scalars for TensorBoard
    """
    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.csv_path = csv_path
        self._file_logger = None
        self._episode = 0
        self._header_written = False

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self._file_logger = open(self.csv_path, "w", encoding="utf-8")

    def _write_header(self) -> None:
        if self._file_logger is None or self._header_written:
            return
        self._file_logger.write(
            "global_step,episode,episode_reward,episode_len,success,collision,mean_dist,final_dist\n"
        )
        self._file_logger.flush()
        self._header_written = True

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)
        rewards = self.locals.get("rewards", None)

        if infos is None or dones is None:
            return True

        # VecEnv with n_envs=1 => lists length 1
        for i, done in enumerate(dones):
            if not done:
                continue

            info = infos[i]
            if "episode" not in info:
                continue

            self._episode += 1
            self._write_header()

            ep_r = float(info["episode"]["r"])
            ep_l = int(info["episode"]["l"])
            success = int(bool(info.get("is_success", False)))
            collision = int(bool(info.get("is_collision", False)))
            mean_dist = float(info.get("mean_dist", np.nan))
            final_dist = float(info.get("final_dist", np.nan))

            self._file_logger.write(
                f"{self.num_timesteps},{self._episode},{ep_r:.6f},{ep_l},{success},{collision},{mean_dist:.6f},{final_dist:.6f}\n"
            )
            self._file_logger.flush()

            self.logger.record("nav/episode_reward", ep_r)
            self.logger.record("nav/episode_len", ep_l)
            self.logger.record("nav/success", success)
            self.logger.record("nav/collision", collision)
            if not np.isnan(mean_dist):
                self.logger.record("nav/mean_dist", mean_dist)
            if not np.isnan(final_dist):
                self.logger.record("nav/final_dist", final_dist)

        return True

    def _on_training_end(self) -> None:
        if self._file_logger:
            self._file_logger.close()
            self._file_logger = None