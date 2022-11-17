import dataclasses
import logging
import multiprocessing
import threading
from dataclasses import dataclass

import torch
import random
import numpy as np
from collections import deque
from typing import Tuple, List, Optional, Deque

from src.collision_type import CollisionType
from src.game import SnakeGameAI, Direction, Point, BLOCK_SIZE, head_to_global_direction
from src.model import Linear_QNet, QTrainer

import wandb

from src.utils.agent_utils.proximity_calculator import ProximityCalculator
from src.utils.utils import flatten

DEFAULT_AGENT_KWARGS = {
    'n_features': 11,
    'max_games': 4000,
    'gamma': 0.9,
    'lr': 0.001,
    'batch_size': 1_000,
    'max_memory': 100_000,
    'n_steps_collision_check': 0,
    'collision_types': [CollisionType.BOTH],
    'model_hidden_size_l1': 256,
    'n_steps_proximity_check': -1,
    'starting_epsilon': 80,
    'random_scale': 200,
    'max_update_end_steps': 0,
    'max_update_start_steps': 0,
    'min_len_snake_at_update': 0,
    'convert_proximity_to_bool': False,
    'scheduler_step_size': 100_000,
    'scheduler_gamma': 0.1,
    'override_proximity_to_bool': True,
    'init_kaiming_normal': False
}


class Agent:

    def __init__(self, game: SnakeGameAI, **kwargs):
        """

        :param n_games:
        :param n_features: corrently, must be mannually calculated, for santy tracking
        :param max_games:
        :param epsilon: randomness
        :param gamma: discount rate
        :param lr:
        :param batch_size:
        :param max_memory:
        :param n_steps:
        :param max_update_steps:
        :param kwargs:
        """
        if kwargs:
            DEFAULT_AGENT_KWARGS.update(kwargs)
            kwargs = DEFAULT_AGENT_KWARGS
        else:
            kwargs = DEFAULT_AGENT_KWARGS

        self.k = kwargs

        self.n_games: int = 0
        # self.n_features: int = kwargs["n_features"]
        self.max_games: int = kwargs['max_games']
        self.gamma: float = kwargs['gamma']
        self.lr: float = kwargs['lr']
        self.batch_size: int = kwargs['batch_size']
        self.max_memory: int = kwargs['max_memory']
        self.n_steps_collision_check: int = kwargs['n_steps_collision_check']
        self.collision_types: List[CollisionType] = kwargs['collision_types']
        self.model_hidden_size_l1: int = kwargs['model_hidden_size_l1']
        # self.non_zero_memory: Deque[int] = kwargs.get('non_zero_memory', deque(maxlen=self.max_memory))
        self.n_steps_proximity_check: int = kwargs['n_steps_proximity_check']
        self.random_scale: int = kwargs['random_scale']
        self.starting_epsilon: int = kwargs['starting_epsilon']  # self.n_games_exploration
        self.max_update_end_steps: int = kwargs['max_update_end_steps']
        self.max_update_start_steps: int = kwargs['max_update_start_steps']
        self.convert_proximity_to_bool: bool = kwargs['convert_proximity_to_bool']
        self.override_proximity_to_bool: bool = kwargs['override_proximity_to_bool']
        self.min_len_snake_at_update: int = kwargs['min_len_snake_at_update']
        self.scheduler_step_size: int = kwargs['scheduler_step_size']
        self.scheduler_gamma: int = kwargs['scheduler_gamma']

        self.init_kaiming_normal: bool = kwargs['init_kaiming_normal']

        self.memory = deque(maxlen=self.max_memory)  # popleft()

        self.last_scores = deque(maxlen=1000)

        self.n_features = len(self.get_state(game))
        self.k['n_features'] = self.n_features
        self.model = Linear_QNet(input_size=self.n_features, hidden_size=self.model_hidden_size_l1, output_size=3, init_kaiming_normal=self.init_kaiming_normal)
        self.trainer = QTrainer(self.model, lr=self.lr, gamma=self.gamma, scheduler_step_size=self.scheduler_step_size, scheduler_gamma=self.scheduler_gamma)
        self.should_update_rewards = self.max_update_start_steps > 0 or self.max_update_end_steps > 0

    @staticmethod
    def get_sorounding_points(point: Point, c: int = 1) -> Tuple[Point, Point, Point, Point]:
        point_l = Point(point.x - c * BLOCK_SIZE, point.y)
        point_r = Point(point.x + c * BLOCK_SIZE, point.y)
        point_u = Point(point.x, point.y - c * BLOCK_SIZE)
        point_d = Point(point.x, point.y + c * BLOCK_SIZE)
        return point_l, point_r, point_u, point_d

    @staticmethod
    def get_point_ahead(direction: Direction, point_l2, point_r2, point_u2, point_d2) -> Point:
        if direction == Direction.LEFT: return point_l2
        if direction == Direction.RIGHT: return point_r2
        if direction == Direction.UP: return point_u2
        if direction == Direction.DOWN: return point_d2

    def is_collisions(
            self,
            game: SnakeGameAI,
            direction: Direction,
            collision_type: CollisionType,
            point_r: Point,
            point_d: Point,
            point_l: Point,
            point_u: Point,
    ) -> List[bool]:
        dir_l, dir_r, dir_u, dir_d = self.get_direction_bool_vector(direction)
        return [
            (dir_r and game.is_collision(collision_type, point_r)) or
            (dir_l and game.is_collision(collision_type, point_l)) or
            (dir_u and game.is_collision(collision_type, point_u)) or
            (dir_d and game.is_collision(collision_type, point_d)),

            # Danger right
            (dir_u and game.is_collision(collision_type, point_r)) or
            (dir_d and game.is_collision(collision_type, point_l)) or
            (dir_l and game.is_collision(collision_type, point_u)) or
            (dir_r and game.is_collision(collision_type, point_d)),

            # Danger left
            (dir_d and game.is_collision(collision_type, point_r)) or
            (dir_u and game.is_collision(collision_type, point_l)) or
            (dir_r and game.is_collision(collision_type, point_u)) or
            (dir_l and game.is_collision(collision_type, point_d)),
        ]

    @staticmethod
    def get_direction_bool_vector(direction) -> List[bool]:
        dir_l = direction == Direction.LEFT
        dir_r = direction == Direction.RIGHT
        dir_u = direction == Direction.UP
        dir_d = direction == Direction.DOWN
        return [
            dir_l,
            dir_r,
            dir_u,
            dir_d,
        ]

    # oriatation features - get collisotions one step ahead in each snake direction [stright, right, left]
    # syntax: point_ snake_direction_ [stright, right, left], global_ { Direction }
    def get_is_collisions_wrapper(
            self,
            game: SnakeGameAI,
            turn: List[int],
            collision_type: CollisionType,
            point_l1: Point,
            point_r1: Point,
            point_u1: Point,
            point_d1: Point,
            n_steps: int = 1,
    ) -> List[bool]:
        some_direction = head_to_global_direction(game.direction, turn)

        collisions_vectors_dist = []
        for _ in range(n_steps):
            point_ahead = self.get_point_ahead(
                some_direction,
                point_l1, point_r1, point_u1, point_d1,
            )

            point_l1, point_r1, point_u1, point_d1 = self.get_sorounding_points(point_ahead)
            collisions_vec_dist = self.is_collisions(game,
                                                     direction=some_direction,
                                                     collision_type=collision_type,
                                                     point_r=point_r1,
                                                     point_d=point_d1,
                                                     point_l=point_l1,
                                                     point_u=point_u1, )
            collisions_vectors_dist.append(collisions_vec_dist)

        return flatten(collisions_vectors_dist)

    def calc_all_directions_collisions(
            self,
            game: SnakeGameAI,
            collision_type: CollisionType,
            point_l1: Point,
            point_r1: Point,
            point_u1: Point,
            point_d1: Point,
            n_steps: int,
    ) -> List[bool]:

        if n_steps == -1:
            return []

        collisions_vec_dist_0 = self.is_collisions(game,
                                                   direction=game.direction,
                                                   collision_type=collision_type,
                                                   point_r=point_r1,
                                                   point_d=point_d1,
                                                   point_l=point_l1,
                                                   point_u=point_u1, )

        if n_steps == 0:
            return collisions_vec_dist_0

        collisions_vec_ahead = [
            self.get_is_collisions_wrapper(
                game,
                direction,
                collision_type,
                point_l1, point_r1, point_u1, point_d1,
                n_steps,
            ) for direction in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        ]

        return [*collisions_vec_dist_0, *flatten(collisions_vec_ahead)]


    def clac_collision_vec_by_type(
            self,
            game: SnakeGameAI,
            collision_types: List[CollisionType],
            point_l1: Point,
            point_r1: Point,
            point_u1: Point,
            point_d1: Point,
            n_steps: int,
    ) -> List[bool]:
        collisions_vec = [
            self.calc_all_directions_collisions(
                game,
                collision_type,
                point_l1,
                point_r1,
                point_u1,
                point_d1,
                self.n_steps_collision_check
            ) for collision_type in collision_types
        ]

        return flatten(collisions_vec)

    def get_state(self, game) -> np.array:

        dir_l, dir_r, dir_u, dir_d = self.get_direction_bool_vector(game.direction)
        head = game.snake[0]
        point_l1, point_r1, point_u1, point_d1 = self.get_sorounding_points(head, c=1)

        collisions_vec = self.clac_collision_vec_by_type(
            game,
            self.collision_types,
            point_l1,
            point_r1,
            point_u1,
            point_d1,
            self.n_steps_collision_check
        )

        # distance to body
        distance_to_body_vec = ProximityCalculator().calc_proximity(
            game,
            self.n_steps_proximity_check,
            point_l1, point_r1, point_u1, point_d1,
        )

        if self.convert_proximity_to_bool:
            if self.override_proximity_to_bool:
                distance_to_body_vec = [prox < 1 for prox in distance_to_body_vec]
            else:
                bool_proximity_vec = [prox < 1 for prox in distance_to_body_vec]
                distance_to_body_vec.extend(bool_proximity_vec)

        snake_len = len(game.snake)

        state = [
            len(game.last_trail) == 0,
            len(game.last_trail) / game.n_blocks,
            snake_len / game.n_blocks,

            # distance to body
            *distance_to_body_vec,

            # is collision
            *collisions_vec,

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
        ]

        return torch.tensor(state, dtype=torch.float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    # def remember_no_zero(self, state, action, reward, next_state, done):
    #     self.non_zero_memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self, game: SnakeGameAI, reward):
        if self.should_update_rewards and len(game.snake) > self.min_len_snake_at_update:
            updated_records = self._update_rewards(game, reward, return_updated_records=True)
            self.trainer.reset_state_to_last_checkpoint()

            # retrain with updated rewards
            for state, action, reward, next_state, done in updated_records:
                self.trainer.train_step(state, action, reward, next_state, done)

        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

        if self.should_update_rewards:
            self.trainer.save_state_checkpoint()

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def _update_rewards(self, game: SnakeGameAI, last_reward: int, return_updated_records):
        len_last_trail = len(game.last_trail)
        last_records = []
        for _ in range(len_last_trail):
            last_records.insert(0, self.memory.pop())

        modified_last_record = []
        for i, record in enumerate(last_records):
            (state, action, reward, next_state, done) = record
            if i < self.max_update_start_steps or i >= (len_last_trail - self.max_update_end_steps):
                reward = last_reward  # reward
            modified_last_record.append((state, action, reward, next_state, done))
            # self.remember_no_zero(state, action, reward, next_state, done)

        for record in modified_last_record:
            self.memory.append(record)

        if return_updated_records:
            return modified_last_record

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        final_move = [0, 0, 0]
        epsilon = self.starting_epsilon - self.n_games
        if epsilon > 0 and random.randint(0, self.random_scale) < epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move



@dataclass
class RunSettings:
    project: str
    group: str
    note: str
    agent_kwargs: dict
    wandb_mode: str
    wandb_setttings: Optional[wandb.Settings] = None
    index: Optional[int] = 0

    def generate_instances(self, n: int):
        return [dataclasses.replace(self, index=i) for i in range(n)]


def train(run_settings: Optional[RunSettings] = None):
    game = SnakeGameAI()

    if not run_settings:
        wandb.init()
        agent = Agent(game, **wandb.config)
        # wandb.config(agent.k)

    else:
        agent = Agent(game, **run_settings.agent_kwargs)
        wandb.init(
            reinit=True,
            project=run_settings.project,
            group=run_settings.group,
            name=str(run_settings.index),
            notes=run_settings.note,
            config=agent.k,
            settings=run_settings.wandb_setttings,
            mode=run_settings.wandb_mode,
        )

    total_score = 0
    record = 0

    # try:
    #     wandb.bwatch(agent.model)
    # except Exception as e:
    #     print(e)
    #     wandb.watch(agent.model)
    min_iter_no_learning = 200
    mean_score = 0
    while agent.n_games < agent.max_games:
        # if agent.n_games > min_iter_no_learning and mean_score < 1:
        #     logging.info(f"breaking learning loop, agent.n_games > {min_iter_no_learning} and mean_score < 1")
        #     break

        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory
            agent.n_games += 1
            agent.train_long_memory(game, reward)
            agent.trainer.lr_scheduler.step()
            game.reset()

            if score > record:
                record = score
                agent.model.save('best.pth')
                torch.save(agent.model.activations_layer_1, f'./model/activations_init_kaiming_normal_{agent.init_kaiming_normal}.pt')

            total_score += score
            agent.last_scores.append(score)
            ma = sum(agent.last_scores) / len(agent.last_scores)
            mean_score = total_score / agent.n_games

            logging.info('Game', agent.n_games, 'Score', score, 'mean_score', mean_score, 'Record:', record)
            linear1_n_non_active = torch.sum(agent.model.activations_layer_1 <= 0)


            if agent.n_games == 1:
                wandb.log({
                    'score': 0,
                    'mean_score': 0,
                    'ma_1000_score': 0,
                    'linear1_n_non_active': torch.sum(agent.model.initial_activations <= 0),
                })

            wandb.log({
                'score': score,
                'mean_score': mean_score,
                'ma_1000_score': ma,
                'linear1_n_non_active': linear1_n_non_active,
            })

    # wandb.finish()
    return agent


if __name__ == '__main__':
    # wandb_mode = "disabled"
    wandb_mode = "online"

    run_settings = [
        RunSettings(
                "lr optimization",
                "continue ; lr: 0.0001 ; starting_epsilon: 8000 ; random_scale: 20_000",
                "",
                {
                    "max_games": 100_000,
                    "n_steps_collision_check": 0,
                    "n_steps_proximity_check": 0,
                    "convert_proximity_to_bool": True,
                    "lr": 0.0001,
                    "starting_epsilon": 8_000,
                    "random_scale": 20_000
                },
                wandb_mode
        ),
        # RunSettings(
        #     "lr optimization",
        #     "lr: 0.0001 ; starting_epsilon: 800 ; random_scale: 2_000",
        #     "",
        #     {
        #         "max_games": 10_000,
        #         "n_steps_collision_check": 0,
        #         "n_steps_proximity_check": 0,
        #         "convert_proximity_to_bool": True,
        #         "lr": 0.0001,
        #         "starting_epsilon": 800,
        #         "random_scale": 2_000
        #     },
        #     wandb_mode
        # ),
        # RunSettings(
        #     "debug model weights",
        #     "no-weight-init",
        #     "collision_check=0 ; n_steps_proximity_check=0 ; max_update_start_steps=0 ; max_update_end_steps=1 ; convert_proximity_to_bool=True",
        #     {
        #         "init_kaiming_normal": False,
        #         "max_games": 1200,
        #         "n_steps_collision_check": 0,
        #         "n_steps_proximity_check": 0,
        #         "max_update_start_steps": 0,
        #         "max_update_end_steps": 0,
        #         "convert_proximity_to_bool": True,
        #     },
        #     wandb_mode
        # ),
        # RunSettings(
        #     "debug model weights",
        #     "with-weight-init",
        #     "collision_check=0 ; n_steps_proximity_check=0 ; max_update_start_steps=0 ; max_update_end_steps=1 ; convert_proximity_to_bool=True",
        #     {
        #         "init_kaiming_normal": True,
        #         "max_games": 1200,
        #         "n_steps_collision_check": 0,
        #         "n_steps_proximity_check": 0,
        #         "max_update_start_steps": 0,
        #         "max_update_end_steps": 0,
        #         "convert_proximity_to_bool": True,
        #     },
        #     wandb_mode
        # )
    ]
    for rs in run_settings:
        train(rs)

    # params = {
    #     'max_games': {'values': [2000]},
    #     'epsilon': {'values': [0]},
    #     'gamma': {'values': [0.9]},
    #     'lr': {'values': [0.001]},
    #     'batch_size': {'values': [1_000]},
    #     'max_memory': {'values': [100_000]},
    #     'n_steps_collision_check': {'values': [0, 1, 2, 4, 8]},
    #     'max_update_steps': {'values': [0, 10, 30, 90, 270]},
    #     'collision_types': {'values': [[CollisionType.BOTH], [CollisionType.BODY, CollisionType.BORDER]]},
    #     'model_hidden_size_l1': {'values': [128, 256, 512, 1024]},
    # }
    #
    # method = "random"
    #
    # sweep_config = {
    #     'method': method,
    #     'metric': {
    #         'name': 'mean_score',
    #         'goal': 'maximize'
    #     },
    #     'parameters': params,
    #     'early_terminate':  {
    #         'type': 'hyperband',
    #         'min_iter': 1200,
    #         'eta': 100,
    #     }
    # }
    #
    # sweep_id = wandb.sweep(sweep_config)
    # wandb.agent(sweep_id, function=train)
