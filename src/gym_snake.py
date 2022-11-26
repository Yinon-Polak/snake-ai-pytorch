import random
from typing import List

import numpy as np
import wandb
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

from src.collision_type import CollisionType
from src.pygame_controller.pygame_controller import PygameController, DummyPygamController
from src.utils.agent_utils.collision_calculator import CollisionCalculator
from src.utils.agent_utils.proximity_calculator import ProximityCalculator
from src.wrappers import Direction, Point, CLOCK_WISE

import numpy as np
import gym
from gym import spaces
from gym.spaces import Dict, Discrete, Box
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import A2C

from torch import nn

BLOCK_SIZE = 20


class SnakeGameAIGym(gym.Env):

    def __init__(
            self,
            collision_types: List[CollisionType] = [CollisionType.BOTH],
            n_steps_collision_check: int = 0,
            n_steps_proximity_check: int = 0,
            convert_proximity_to_bool: bool = True,
            override_proximity_to_bool: bool = True,
            add_prox_preferred_turn_0: bool = False,
            add_prox_preferred_turn_1: bool = False,
            w: int = 640,
            h: int = 480,
            use_pygame: bool = True,
            positive_reward: int = 10,
            negative_reward: int = -10,
    ):
        super(SnakeGameAIGym, self).__init__()

        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.w = w
        self.h = h
        self.n_blocks = (self.w / 20) * (self.h / 20)
        self.pygame_controller = PygameController(self.w, self.h, BLOCK_SIZE) if use_pygame else DummyPygamController()

        #### vars that's are defined in reset()
        self.direction = None
        self.head = None
        self.snake = None
        self.trail = None
        self.last_trail = None
        self.score = None
        self.food = None
        self.frame_iteration = None
        self.n_games = 0
        ####

        self.action_space = spaces.Discrete(3)
        self.observation_space = Dict({
            "is_first_episode_step": Discrete(2),
            # "len_episode": Box(low=0, high=1, shape=()),
            # "len_snake": Box(low=0, high=1, shape=()),
            "prox_s": Discrete(2),
            "prox_r": Discrete(2),
            "prox_l": Discrete(2),
            "collision_s": Discrete(2),
            "collision_r": Discrete(2),
            "collision_l": Discrete(2),
            "dir_l": Discrete(2),
            "dir_r": Discrete(2),
            "dir_u": Discrete(2),
            "dir_d": Discrete(2),
            "food_left": Discrete(2),
            "food_right": Discrete(2),
            "food_up": Discrete(2),
            "food_down": Discrete(2),
        })
        # self.reset()
        self.proximity_calculator = ProximityCalculator()
        self.collision_calculator = CollisionCalculator(BLOCK_SIZE)
        self.collision_types = collision_types
        self.n_steps_collision_check = n_steps_collision_check
        self.n_steps_proximity_check = n_steps_proximity_check
        self.convert_proximity_to_bool = convert_proximity_to_bool
        self.override_proximity_to_bool = override_proximity_to_bool
        self.add_prox_preferred_turn_0 = add_prox_preferred_turn_0
        self.add_prox_preferred_turn_1 = add_prox_preferred_turn_1

    def _get_state(self):

        dir_l, dir_r, dir_u, dir_d = self.collision_calculator.get_direction_bool_vector(self.direction)
        head = self.snake[0]
        point_l1, point_r1, point_u1, point_d1 = self.collision_calculator.get_sorounding_points(head, c=1)

        collisions_vec = self.collision_calculator.clac_collision_vec_by_type(
            self,
            self.collision_types,
            point_l1,
            point_r1,
            point_u1,
            point_d1,
            self.n_steps_collision_check
        )

        # distance to body
        distance_to_body_vec = ProximityCalculator().calc_proximity(
            self,
            self.n_steps_proximity_check,
            self.convert_proximity_to_bool,
            self.override_proximity_to_bool,
            self.add_prox_preferred_turn_0,
            self.add_prox_preferred_turn_1,
            point_l1, point_r1, point_u1, point_d1,
        )

        snake_len = len(self.snake)

        state_0 = [
            len(self.last_trail) == 0,
            # len(self.last_trail) / self.n_blocks,
            # snake_len / self.n_blocks,

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
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y,  # food down
        ]

        self.sum_score = 0
        self.n_games = 0

        state_dict = {
            "is_first_episode_step": int(state_0[0]),
            # "len_episode": np.array(state_0[1], dtype=np.float32),
            # "len_snake": np.array(state_0[2], dtype=np.float32),
            "prox_s": int(state_0[1]),
            "prox_r": int(state_0[2]),
            "prox_l": int(state_0[3]),
            "collision_s": int(state_0[4]),
            "collision_r": int(state_0[5]),
            "collision_l": int(state_0[6]),
            "dir_l": int(state_0[7]),
            "dir_r": int(state_0[8]),
            "dir_u": int(state_0[9]),
            "dir_d": int(state_0[10]),
            "food_left": int(state_0[11]),
            "food_right": int(state_0[12]),
            "food_up": int(state_0[13]),
            "food_down": int(state_0[14]),
        }
        return state_dict

    def step(self, action):
        return self.play_step(action)

    def render(self, mode="human"):
        pass

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.trail = []
        self.last_trail = []

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

        return self._get_state()

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        self.pygame_controller.check_quit_event()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            self.n_games += 1
            reward = self.negative_reward

            self.sum_score += self.score
            self.n_games += 1
            wandb.log({
                'score': self.score,
                'mean_score': self.sum_score / self.n_games,
            })

            return self._get_state(), reward, game_over, {"score": self.score}

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = self.positive_reward
            self._place_food()
            # self.last_trail.clear()  # todo make sure this is being cleared
        else:
            crumb = self.snake.pop()
            self.trail.insert(0, crumb)
            self.last_trail.insert(0, crumb)

        # 5. update ui and clock
        self.pygame_controller.update_ui(self.food, self.score, self.snake)
        self.pygame_controller.clock_tick()
        # 6. return game over and score

        return self._get_state(), reward, game_over, {"score": self.score}

    def is_collision(self, collision_type: CollisionType = CollisionType.BOTH, pt: Point = None):
        if pt is None:
            pt = self.head
        # hits boundary
        if (collision_type == CollisionType.BORDER or collision_type == CollisionType.BOTH) and \
                (pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0):
            return True
        # hits itself
        if (collision_type == CollisionType.BODY or collision_type == CollisionType.BOTH) and \
                pt in self.snake[1:]:
            return True

        return False

    def _is_hit_itslef(self, pt=None) -> bool:
        if pt is None:
            pt = self.head

        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _move(self, action):
        if isinstance(action, np.int64):
            action_list = [0, 0, 0]
            action_list[action] = 1
            action = action_list
        elif not isinstance(action, list):
            raise RuntimeError(f"no convertor implemented for action type: {type(action)}, only np.int64 and list are supported")

        self.direction = head_to_global_direction(current_direction=self.direction, action=action)

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)


def head_to_global_direction(current_direction, action) -> Direction:
    # [straight, right, left]
    idx = CLOCK_WISE.index(current_direction)

    if np.array_equal(action, [1, 0, 0]):
        new_dir = CLOCK_WISE[idx]  # no change
    elif np.array_equal(action, [0, 1, 0]):
        next_idx = (idx + 1) % 4
        new_dir = CLOCK_WISE[next_idx]  # right turn r -> d -> l -> u
    else:  # [0, 0, 1]
        next_idx = (idx - 1) % 4
        new_dir = CLOCK_WISE[next_idx]  # left turn r -> u -> l -> d

    return new_dir


if __name__ == '__main__':
    ##
    ## check env
    ##
    # from stable_baselines3.common.env_checker import check_env
    # env = SnakeGameAIGym(
    #     collision_types = [CollisionType.BOTH],
    #     n_steps_collision_check=0,
    #     n_steps_proximity_check=0,
    #     convert_proximity_to_bool=True,
    #     override_proximity_to_bool=True,
    #     add_prox_preferred_turn_0=False,
    #     add_prox_preferred_turn_1=False,
    # )
    # check_env(env, warn=True)

    ##
    ## run
    ##
    env_name = "snake-ai-gym-v1"
    config = {
        "policy_type": "MultiInputPolicy",
        "total_timesteps": 25000,
        "env_name": env_name,
    }
    run = wandb.init(
        project="a2c",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )

    from gym.envs.registration import register


    # Example for the CartPole environment
    register(
        # unique identifier for the env `name-version`
        id=env_name,
        # path to the class for creating the env
        # Note: entry_point also accept a class as input (and not only a string)
        entry_point=SnakeGameAIGym,
        # Max number of steps per episode, using a `TimeLimitWrapper`
        # max_episode_steps=500_000,
    )

    model = A2C(
        config["policy_type"],
        config["env_name"],
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        # n_steps=300,
        # policy_kwargs={'activation_fn': nn.ReLU},
        # learning_rate=0.007,
        # gamma=0.9
    )
    model.learn(
        total_timesteps=config["total_timesteps"],
        # n_eval_episodes=3000,
        callback=WandbCallback(
            # model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    run.finish()