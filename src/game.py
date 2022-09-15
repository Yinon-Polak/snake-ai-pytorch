import random
from enum import Enum
from collections import namedtuple
import numpy as np

from src.collision_type import CollisionType
from src.pygame_controller import PygameController
from src.wrappers import Direction, Point, CLOCK_WISE

BLOCK_SIZE = 20


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.pygame_controller = PygameController(self.w, self.h, BLOCK_SIZE)

        #### vars that's are defined in reset()
        self.direction = None
        self.head = None
        self.snake = None
        self.trail = None
        self.last_trail = None
        self.is_looping = None
        self.count_in_tail = None
        self.score = None
        self.food = None
        self.frame_iteration = None
        ####

        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.trail = []
        self.last_trail = []
        self.is_looping = False
        self.count_in_tail = 0

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

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
        if self.is_collision(CollisionType.BOTH) or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            self.last_trail.clear()
        else:
            crumb = self.snake.pop()
            self.trail.insert(0, crumb)
            self.last_trail.insert(0, crumb)

        # 5. update ui and clock
        self.pygame_controller.update_ui(self.food, self.score, self.snake)
        self.pygame_controller.clock_tick()
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, collision_type: CollisionType, pt: Point = None):
        if pt is None:
            pt = self.head
        # hits boundary
        if (collision_type == CollisionType.BORDER or collision_type == CollisionType.BOTH) and\
                (pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0):
            return True
        # hits itself
        if (collision_type == CollisionType.BODY or collision_type == CollisionType.BOTH) and\
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
