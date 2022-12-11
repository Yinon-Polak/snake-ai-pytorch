import copy
import random

import numpy as np
import wandb
from gym.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

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

from src.gym_ext.info_vec_env_wrapper import CostumeWandbEnvLogger, CostumeWandbVecEnvLogger

BLOCK_SIZE = 20

KWARGS = {
    "collision_types": [CollisionType.BOTH],
    "n_steps_collision_check": 0,
    "n_steps_proximity_check": 0,
    "convert_proximity_to_bool": True,
    "override_proximity_to_bool": True,
    "add_prox_preferred_turn_0": False,
    "add_prox_preferred_turn_1": False,
    "w": 640,
    "h": 480,
    "use_pygame": False,
    "positive_reward": 10,
    "negative_reward": -10,
}


class SnakeGameAIGym(gym.Env):

    def __init__(self, **kwargs):
        super(SnakeGameAIGym, self).__init__()

        _kwargs = copy.deepcopy(KWARGS)
        _kwargs.update(kwargs)

        print(f'kwargs:', _kwargs)

        self.positive_reward = _kwargs["positive_reward"]
        self.negative_reward = _kwargs["negative_reward"]
        self.w = _kwargs["w"]
        self.h = _kwargs["h"]
        self.use_pygame = _kwargs["use_pygame"]
        self.w_pixels = int(self.w / BLOCK_SIZE)
        self.h_pixels = int(self.h / BLOCK_SIZE)
        self.n_blocks = self.w_pixels * self.h_pixels
        self.screen_mat_shape = (self.h_pixels + 2, self.w_pixels + 2, 1)
        self.pygame_controller = PygameController(self.w, self.h,
                                                  BLOCK_SIZE) if self.use_pygame else DummyPygamController()

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

        self.pixel_color_border = 1
        self.pixel_color_background = 64
        self.pixel_color_body = 128
        self.pixel_color_snake_head = 192
        self.pixel_color_food = 255


        self.action_space = spaces.Discrete(3)
        self.observation_space = Box(low=0, high=255, shape=self.screen_mat_shape, dtype=np.uint8)

        self.reset()
        initial_state = self._get_features()
        gym_spaces_mapping = {}
        for k, v in initial_state.items():
            if isinstance(v, int):
                gym_spaces_mapping[k] = Discrete(2)
            elif isinstance(v, float):
                gym_spaces_mapping[k] = Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32)
            else:
                raise Exception("type not supported")

        self.observation_space = Dict(gym_spaces_mapping)

    def _get_features(self):
        dir_l, dir_r, dir_u, dir_d = self.collision_calculator.get_direction_bool_vector(self.direction)
        head = self.snake[0]
        point_l1, point_r1, point_u1, point_d1 = self.collision_calculator.get_sorounding_points(head, c=1)

        snake_len = len(self.snake)
        len_episode = len(self.last_trail) / self.n_blocks
        normalized_len_snake = snake_len / self.n_blocks
        is_first_episode_step = len(self.last_trail) == 0

        features = {
            "is_first_episode_step": int(is_first_episode_step),
            "dir_l": int(dir_l),
            "dir_r": int(dir_r),
            "dir_u": int(dir_u),
            "dir_d": int(dir_d),
            "food_left": int(self.food.x < self.head.x),  # food left
            "food_right": int(self.food.x > self.head.x),  # food right
            "food_up": int(self.food.y < self.head.y),  # food up
            "food_down": int(self.food.y > self.head.y),  # food down
            "len_episode": len_episode,
            "len_snake": normalized_len_snake,
        }

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

        c = 0
        for i in range(0, len(collisions_vec), 3):
            features[f"collision_s_{c}"] = int(collisions_vec[i])
            features[f"collision_r_{c}"] = int(collisions_vec[i+1])
            features[f"collision_l_{c}"] = int(collisions_vec[i+2])
            c += 1

        c = 0
        for i in range(0, len(distance_to_body_vec), 3):
            features[f"prox_s_{c}"] = int(distance_to_body_vec[i]) if self.convert_proximity_to_bool else distance_to_body_vec[i]
            features[f"prox_r_{c}"] = int(distance_to_body_vec[i+1]) if self.convert_proximity_to_bool else distance_to_body_vec[i+1]
            features[f"prox_l_{c}"] = int(distance_to_body_vec[i+2]) if self.convert_proximity_to_bool else distance_to_body_vec[i+2]
            c += 1

        return features

    def _get_state(self):
        # try:
            pixel_mat = np.zeros(self.screen_mat_shape)
            pixel_mat += self.pixel_color_background

            # border
            pixel_mat[0, :] = self.pixel_color_border
            pixel_mat[-1, :] = self.pixel_color_border
            pixel_mat[:, 0] = self.pixel_color_border
            pixel_mat[:, -1] = self.pixel_color_border

            pixel_mat[self.food.get_y_x_0_tuple()] = self.pixel_color_food
            for body_point in self.snake:
                pixel_mat[body_point.get_y_x_0_tuple()] = self.pixel_color_body

            pixel_mat[self.head.get_y_x_0_tuple()] = self.pixel_color_snake_head
            return pixel_mat.astype(np.uint8)

        # except Exception as e:
        #     print(e)

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
            raise RuntimeError(
                f"no convertor implemented for action type: {type(action)}, only np.int64 and list are supported")

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
    env_name = "snake-ai-gym-v1"
    config = {
        "policy_type": "MultiInputPolicy",
        "total_timesteps": 5_000_000,
        "env_name": env_name,
    }
    run = wandb.init(
        project="test",
        name="auto spaces dict init",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )

    vec_env = make_vec_env(vec_env_cls=DummyVecEnv, env_id=SnakeGameAIGym, wrapper_class=RecordEpisodeStatistics,
                           n_envs=15)
    # vec_env = RecordEpisodeStatistics(vec_env)
    vec_env = CostumeWandbVecEnvLogger(vec_env)

    model = A2C(
        config["policy_type"],
        vec_env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        # n_steps=300,
        policy_kwargs={'activation_fn': nn.ReLU},
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
