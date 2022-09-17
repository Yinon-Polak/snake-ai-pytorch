import torch
import random
import numpy as np
from collections import deque
from typing import Tuple, List, Optional, Deque

from src.collision_type import CollisionType
from src.game import SnakeGameAI, Direction, Point, BLOCK_SIZE, head_to_global_direction
from src.model import Linear_QNet, QTrainer

import wandb


def flatten(l: List):
    return [item for sublist in l for item in sublist]


class Agent:

    def __init__(self, **kwargs):
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
        self.n_games: int = 0
        self.n_features: int = kwargs.get("n_features")
        self.max_games: int = kwargs.get('max_games', 3500)
        self.epsilon: int = kwargs.get('epsilon', 0)
        self.gamma: float = kwargs.get('gamma', 0.9)
        self.lr: float = kwargs.get('lr', 0.001)
        self.batch_size: int = kwargs.get('batch_size', 1_000)
        self.max_memory: int = kwargs.get('max_memory', 100_000)
        self.n_steps_collision_check: int = kwargs.get('n_steps_collision_check', 0)
        self.max_update_steps: int = kwargs.get('max_update_steps', 0)
        self.collision_types: List[CollisionType] = kwargs.get('collision_types', [CollisionType.BOTH])
        # self.non_zero_memory: Deque[int] = kwargs.get('non_zero_memory', deque(maxlen=self.max_memory))

        self.memory = deque(maxlen=self.max_memory)  # popleft()
        self.model = Linear_QNet(self.n_features, 256, 3)
        self.trainer = QTrainer(self.model, lr=self.lr, gamma=self.gamma)

        self.last_scores = deque(maxlen=500)

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
            self, game: SnakeGameAI,
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

        # distance_l = head.x / game.w
        # distance_r = (game.w - head.x) / game.w
        # distance_u = (game.h - head.y) / game.h
        # distance_d = head.x / game.h
        #
        # # distance to body
        # distance_to_body_stright, distance_to_body_right, distance_to_body_left = self.get_distances_to_body(game)

        state = [

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
        if self.max_update_steps > 0:
            self._update_rewards(game, reward)

        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def _update_rewards(self, game: SnakeGameAI, last_reward: int):
        n_change = min(len(game.snake), self.max_update_steps)
        last_records = []
        for _ in range(n_change):
            (state, action, reward, next_state, done) = self.memory.pop()
            reward = last_reward  # reward
            last_records.insert(0, (state, action, reward, next_state, done))
            # self.remember_no_zero(state, action, reward, next_state, done)

        for record in last_records:
            self.memory.append(record)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    @staticmethod
    def get_distances_to_body(game: SnakeGameAI) -> Tuple[int, int, int]:
        """

        :param game:
        :param look_dierction:
        :return:
        """

        def is_snake_point_right(p):
            return p.y == game.head.y and p.x > game.head.x

        def is_snake_point_down(p):
            return p.x == game.head.x and p.y > game.head.y

        def is_snake_point_left(p):
            return p.y == game.head.y and p.x < game.head.x

        def is_snake_point_up(p):
            return p.x == game.head.x and p.y < game.head.y

        def get_same_lines_criteria(direction):
            if direction == Direction.RIGHT:
                return is_snake_point_right
            if direction == Direction.DOWN:
                return is_snake_point_down
            if direction == Direction.LEFT:
                return is_snake_point_left
            if direction == Direction.UP:
                return is_snake_point_up

        def y_dist(p):
            return abs(p.x - game.head.x)

        def x_dist(p):
            return abs(p.y - game.head.y)

        def get_distance_to_closest_point(direction):
            criteria = get_same_lines_criteria(game.direction)
            body_points = sorted(
                filter(criteria, game.snake),
                key=x_dist if direction in [Direction.RIGHT, Direction.LEFT] else y_dist,
            )
            closest_point = next(iter(body_points), None)
            if not closest_point:
                return 1

            dist_func = x_dist if direction in [Direction.RIGHT, Direction.LEFT] else y_dist
            return dist_func(closest_point) / game.w

        # dist to body stright
        distance_stright = get_distance_to_closest_point(game.direction)

        # distance to body right
        direction = head_to_global_direction(current_direction=game.direction, action=[0, 1, 0])
        distance_to_right = get_distance_to_closest_point(direction)

        # distance to left
        direction = head_to_global_direction(current_direction=game.direction, action=[0, 0, 1])
        distance_to_left = get_distance_to_closest_point(direction)

        return distance_stright, distance_to_right, distance_to_left


def train(
        group: str,
        run: int,
        note: str = None,
        wandb_mode: Optional[str] = None,
        wandb_setttings: wandb.Settings = None,
        agent_kwargs: dict = {},
):
    total_score = 0
    record = 0
    agent = Agent(**agent_kwargs)
    game = SnakeGameAI()

    wandb.init(
        reinit=True,
        project='sanke-ai-group-runs',
        group=group,
        name=str(run),
        notes=note,
        config={
            "architecture": "Linear_QNet",
            "learning_rate": agent.lr,
            "batch_size": agent.batch_size,
            "max_memory": agent.max_memory,
            "gamma": agent.gamma,
        },
        settings=wandb_setttings,
        mode=wandb_mode,
    )

    try:
        wandb.bwatch(agent.model)
    except Exception as e:
        print(e)
        wandb.watch(agent.model)

    while agent.n_games < agent.max_games:
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
            game.reset()
            agent.n_games += 1
            agent.train_long_memory(game, reward)

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            total_score += score
            agent.last_scores.append(score)
            ma = sum(agent.last_scores) / agent.n_games
            mean_score = total_score / agent.n_games


            # weights and baises logging
            wandb.log({
                'score': score,
                'mean_score': mean_score,
                'ma_500_score': ma,
            })


    wandb.finish()


if __name__ == '__main__':
    # wandb_mode = "disabled"
    wandb_mode = "online"

    parmas = [
        # ("base-line", "base line - as it came from repo", {"n_features": 11}),
        # ("batch-size-2000", "increase batch size to 2000", {"n_features": 11, "batch_size": 2000}),
        # ("batch-size-5000", "increase batch size to 5000", {"n_features": 11, "batch_size": 5000}),
        ("batch-size=10_000", "increase batch size to 10000", {"n_features": 11, "batch_size": 10_000}),
        ("split-collisions", "collision_types = [CollisionType.BODY, CollisionType.BORDER]", {"n_features": 14, "collision_types": [CollisionType.BODY, CollisionType.BORDER]}),
        ("max_update_steps=30", "set last 30 moves reward equal last reward", {"n_features": 11, "max_update_steps": 30}),
        ("n_steps_collision_check=1", "look ahead 1 steps and check collsions, collision_types = CollisionType.BOTH", {"n_features": 20, "n_steps_collision_check": 1}),
    ]

    for (j, (group, note, agent_kwargs)) in enumerate(parmas):
        for i in range(5):
            train(
                group=group,
                run=i,
                note=note,
                wandb_mode=wandb_mode,
                agent_kwargs=agent_kwargs,
            )

    # train('split-collision', 0, agent_kwargs={"n_features": 11, "max_games": 10}, note=None, wandb_mode="disabled",)

