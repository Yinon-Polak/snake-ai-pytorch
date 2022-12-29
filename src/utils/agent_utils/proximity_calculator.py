from typing import List

import numpy as np

from src.game import SnakeGameAI, head_to_global_direction
from src.utils.utils import flatten
from src.wrappers import Point, Direction, HORIZONTAL_DIRECTIONS, SNAKE_TURNS


class DistanceCalculator:

    def __init__(self):
        pass

    def calc_distance(
            self,
            game: SnakeGameAI,
            n_steps: int,
            convert_proximity_to_bool: bool,
            override_proximity_to_bool: bool,
            add_prox_preferred_turn_0: bool,
            add_prox_preferred_turn_1: bool,
            calc_border: bool,
            point_l1: Point,
            point_r1: Point,
            point_u1: Point,
            point_d1: Point,
    ) -> List[float]:
        if n_steps == -1:
            return []

        result = []
        result.extend(self.get_distance_to_body(game.direction, game.head, game.snake, game.w, game.h))
        if calc_border:
            result.extend(self.get_distance_to_border_in_snake_view(game.head, game.direction, game.w, game.h))

        if n_steps > 0:
            proximity_vec_ahead = [
                self.get_distance_wrapper(
                    game=game,
                    turn=turn,
                    n_steps=n_steps,
                    calc_border=calc_border,
                    point_l1=point_l1,
                    point_r1=point_r1,
                    point_u1=point_u1,
                    point_d1=point_d1,
                ) for turn in SNAKE_TURNS
            ]

            result.extend(flatten(proximity_vec_ahead))

        return result

    @staticmethod
    def get_distance_to_border(point: Point, direction: Direction, w, h) -> float:
        if direction == Direction.RIGHT:
            return (w - point.x) / w
        if direction == Direction.DOWN:
            return (h - point.y) / h
        if direction == Direction.LEFT:
            return point.x / w
        if direction == Direction.UP:
            return point.y / h

    def get_distance_to_border_in_snake_view(self, point: Point, direction: Direction, w: int, h: int) -> List[float]:
        distances = []
        for snake_direction in SNAKE_TURNS:
            direction = head_to_global_direction(current_direction=direction, action=snake_direction)
            distance = self.get_distance_to_border(point, direction, w, h)
            distances.append(distance)

        return distances

    # todo change name to collision_danger_prob
    @staticmethod
    def get_distance_to_body(
            initial_direction: Direction,
            comparison_point: Point,
            snake: List[Point],
            w: int,
            h: int
    ) -> List[float]:
        """

        :param game:
        :param look_dierction:
        :return:
        """

        def is_snake_point_right(snake_point: Point):
            return snake_point.y == comparison_point.y and snake_point.x > comparison_point.x

        def is_snake_point_down(snake_point: Point):
            return snake_point.x == comparison_point.x and snake_point.y > comparison_point.y

        def is_snake_point_left(snake_point: Point):
            return snake_point.y == comparison_point.y and snake_point.x < comparison_point.x

        def is_snake_point_up(snake_point: Point):
            return snake_point.x == comparison_point.x and snake_point.y < comparison_point.y

        def get_same_lines_criteria(direction):
            if direction == Direction.RIGHT:
                return is_snake_point_right
            if direction == Direction.DOWN:
                return is_snake_point_down
            if direction == Direction.LEFT:
                return is_snake_point_left
            if direction == Direction.UP:
                return is_snake_point_up

        def y_distance(snake_point: Point):
            return abs(snake_point.y - comparison_point.y) - 20

        def x_distance(snake_point: Point):
            return abs(snake_point.x - comparison_point.x) - 20

        def calc_proximity_to_closest_body_point(direction) -> float:
            criteria = get_same_lines_criteria(direction)
            body_points = sorted(
                filter(criteria, snake),
                key=x_distance if direction in HORIZONTAL_DIRECTIONS else y_distance,
            )
            closest_point = next(iter(body_points), None)
            if not closest_point:
                return 1

            dist_func = x_distance if direction in HORIZONTAL_DIRECTIONS else y_distance
            denominator = w if direction in [Direction.RIGHT, Direction.LEFT] else h
            return 1 - (dist_func(closest_point) / denominator)

        distances = []
        for snake_direction in SNAKE_TURNS:
            direction = head_to_global_direction(current_direction=initial_direction, action=snake_direction)
            distance = calc_proximity_to_closest_body_point(direction)
            distances.append(distance)

        return distances

    def get_distance_wrapper(
            self,
            game: SnakeGameAI,
            turn: List[int],
            calc_border: bool,
            point_l1: Point,
            point_r1: Point,
            point_u1: Point,
            point_d1: Point,
            n_steps: int = 1,
    ) -> List[float]:
        some_direction = head_to_global_direction(game.direction, turn)

        collisions_distances_vectors = []
        for _ in range(n_steps):
            point_ahead = self.get_point_ahead(
                some_direction,
                point_l1, point_r1, point_u1, point_d1,
            )

            distance_to_body = self.get_distance_to_body(
                initial_direction=some_direction,
                comparison_point=point_ahead,
                snake=game.snake,
                w=game.w,
                h=game.h,
            )
            collisions_distances_vectors.append(distance_to_body)

            if calc_border:
                border_proximity_v = self.get_distance_to_border_in_snake_view(point_ahead, some_direction, game.w, game.h)
                # border_proximity_v = [min(border_proximity_i, distance_to_body[i]) for i, border_proximity_i in enumerate(border_proximity_v)] # take min between distance to body and distance to border
                collisions_distances_vectors.append(border_proximity_v)

        return flatten(collisions_distances_vectors)

    @staticmethod
    def get_point_ahead(direction: Direction, point_l2, point_r2, point_u2, point_d2) -> Point:
        if direction == Direction.LEFT: return point_l2
        if direction == Direction.RIGHT: return point_r2
        if direction == Direction.UP: return point_u2
        if direction == Direction.DOWN: return point_d2

    @staticmethod
    def calc_preferred_turns_0(proximity_vec) -> List[int]:
        preferred_turns = []
        for i in range(0, len(proximity_vec), 3):
            sub_vec = proximity_vec[i:i+3]
            if min(sub_vec) == 1:
                preferred_turns.extend([0, 0, 1])
                continue

            a = np.argmin(sub_vec)
            if a == 0:
                preferred_turns.extend([1, 0, 0])
            elif a == 1:
                preferred_turns.extend([0, 1, 0])
            else:
                preferred_turns.extend([0, 0, 0])

        return preferred_turns

    @staticmethod
    def calc_preferred_turns_1(proximity_vec) -> List[int]:
        preferred_turns = []
        for i in range(0, len(proximity_vec), 3):
            preferred_turns.append(int(proximity_vec[i] > proximity_vec[i+1]))
            preferred_turns.append(int(proximity_vec[i] > proximity_vec[i+2]))
            preferred_turns.append(int(proximity_vec[i+1] > proximity_vec[i+2]))
            preferred_turns.append(int(min(proximity_vec[i:i+3]) == 1))

        return preferred_turns
