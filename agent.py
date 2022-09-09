from typing import Tuple, List

import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE, head_to_global_direction
from model import Linear_QNet, QTrainer
from helper import plot

import wandb


def flatten(l: List):
    return [item for sublist in l for item in sublist]


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.lr = 0.001
        self.batch_size = 10_000
        self.max_memory = 100_000

        self.memory = deque(maxlen=self.max_memory)  # popleft()
        # self.non_zero_memory = deque(maxlen=self.max_memory)  # popleft()
        self.model = Linear_QNet(29, 256, 3)
        self.trainer = QTrainer(self.model, lr=self.lr, gamma=self.gamma)

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
            point_r: Point,
            point_d: Point,
            point_l: Point,
            point_u: Point,
    ):
        dir_l, dir_r, dir_u, dir_d = self.get_direction_bool_vector(direction)
        return [
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
        ]

    @staticmethod
    def get_direction_bool_vector(direction):
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
                                                     point_r=point_r1,
                                                     point_d=point_d1,
                                                     point_l=point_l1,
                                                     point_u=point_u1, )
            collisions_vectors_dist.append(collisions_vec_dist)

        return flatten(collisions_vectors_dist)

    def get_state(self, game):

        dir_l, dir_r, dir_u, dir_d = self.get_direction_bool_vector(game.direction)
        head = game.snake[0]
        point_l1, point_r1, point_u1, point_d1 = self.get_sorounding_points(head, c=1)
        # point_l2, point_r2, point_u2, point_d2 = self.get_sorounding_points(head, c=2)

        collisions_vec_dist_0 = self.is_collisions(game,
                                                   direction=game.direction,
                                                   point_r=point_r1,
                                                   point_d=point_d1,
                                                   point_l=point_l1,
                                                   point_u=point_u1, )

        collisions_vec_dist_s1 = self.get_is_collisions_wrapper(
            game,
            game.direction,
            point_l1, point_r1, point_u1, point_d1,
            2
        )

        collisions_vec_dist_r1 = self.get_is_collisions_wrapper(
            game,
            [0, 1, 0],
            point_l1, point_r1, point_u1, point_d1,
            2
        )

        collisions_vec_dist_l1 = self.get_is_collisions_wrapper(
            game,
            [0, 0, 1],
            point_l1, point_r1, point_u1, point_d1,
            2
        )

        # distance_l = head.x / game.w
        # distance_r = (game.w - head.x) / game.w
        # distance_u = (game.h - head.y) / game.h
        # distance_d = head.x / game.h
        #
        # # distance to body
        # distance_to_body_stright, distance_to_body_right, distance_to_body_left = self.get_distances_to_body(game)

        state = [
            *collisions_vec_dist_0,
            *collisions_vec_dist_s1,
            *collisions_vec_dist_r1,
            *collisions_vec_dist_l1,

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

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    # def remember_no_zero(self, state, action, reward, next_state, done):
    #     self.non_zero_memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
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

    def update_rewards(self, game: SnakeGameAI, last_reward):
        n_change = min(len(game.snake), 30)
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

    def get_distances_to_body(self, game: SnakeGameAI) -> Tuple[int, int, int]:
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


def train(name: str, run: int):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    wandb.init(
        # Set the project where this run will be logged
        project='sanke-ai',
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"{name}-{run}",
        # Track hyperparameters and run metadata
        config={
            "architecture": "Linear_QNet",
            "learning_rate": agent.lr,
            "batch_size": agent.batch_size,
            "max_memory": agent.max_memory,
            "gamma": agent.gamma,
            # "epochs": 10,
        },
        # mode="disabled",
    )
    wandb.watch(agent.model)

    while agent.n_games < 800:
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
            #
            #
            #
            # , plot result
            game.reset()
            agent.n_games += 1
            agent.update_rewards(game, reward)
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)

            # weights and baises logging
            wandb.log({
                'score': score,
                'mean_score': mean_score,
            })
    wandb.finish()


if __name__ == '__main__':
    # todo run batch_size [n*2, n*3]
    # todo change sampling scheme
    for i in range(3):
        train('ahead-2-is-collision ; update rewards - len 30', i)
