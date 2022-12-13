from collections import namedtuple
from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


CLOCK_WISE = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]


@dataclass
class Point:
    x: int
    y: int
    block_size: int = 20

    def _pixel_index(self, a):
        return int(a // self.block_size) + 1  # add one for border, the range of x is (-20,w), for y its (-20,h)

    def get_y_x_tuple(self):
        return self._pixel_index(self.y), self._pixel_index(self.x)
