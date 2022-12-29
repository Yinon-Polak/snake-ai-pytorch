from collections import namedtuple
from enum import Enum



class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')


CLOCK_WISE_DIRECTIONS = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

SNAKE_TURNS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

HORIZONTAL_DIRECTIONS = [Direction.RIGHT, Direction.LEFT]
VERTICAL_DIRECTIONS = [Direction.DOWN, Direction.UP]
