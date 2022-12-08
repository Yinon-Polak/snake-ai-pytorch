from collections import deque

import gym
import wandb
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs


class CostumeWandbEnvLogger(gym.Wrapper):
    def __init__(self, env):
        super(CostumeWandbEnvLogger, self).__init__(env)

    def step(self, action):
        observations, rewards, dones, infos = super(CostumeWandbEnvLogger, self).step(
            action
        )
        return observations, rewards, dones, infos


class CostumeWandbVecEnvLogger(VecEnvWrapper):
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)
        self.last_scores = deque(maxlen=1000)
        self.total_score = 0
        self._n_games = 0

    def reset(self) -> VecEnvObs:
        return self.venv.reset()

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        for i, done in enumerate(dones):
            if done:
                self._n_games += 1
                score = infos[i]['score']
                self.total_score += score
                self.last_scores.append(score)

                ma = sum(self.last_scores) / len(self.last_scores)
                mean_score = self.total_score / self._n_games
                wandb.log(
                    {
                        'score': score,
                        'ma_1000': ma,
                        'mean_score': mean_score,
                    }
                )

        return obs, rewards, dones, infos