from typing import List, Tuple

from src.collision_type import CollisionType
from src.game import SnakeGameAI, head_to_global_direction
from src.utils.utils import flatten
from src.wrappers import Point, Direction


class CollisionCalculator:

    def __init__(self, block_size: int):
        self.block_size = block_size

    def get_sorounding_points(self, point: Point, c: int = 1) -> Tuple[Point, Point, Point, Point]:
        point_l = Point(point.x - c * self.block_size, point.y)
        point_r = Point(point.x + c * self.block_size, point.y)
        point_u = Point(point.x, point.y - c * self.block_size)
        point_d = Point(point.x, point.y + c * self.block_size)
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
                n_steps
            ) for collision_type in collision_types
        ]

        return flatten(collisions_vec)