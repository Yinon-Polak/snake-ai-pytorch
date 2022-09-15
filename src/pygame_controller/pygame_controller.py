import pygame
import abc
from pathlib import Path
from typing import List

import zope.interface

from src.wrappers import Point

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

SPEED = 40


class PygameControllerInterface(zope.interface.Interface):
    def check_quit_event():
        """"""

    def update_ui(food: Point, score: float, snake: List[Point]):
        """"""

    def clock_tick():
        """"""


@zope.interface.implementer(PygameControllerInterface)
class DummyPygamController:
    def __init__(self):
        pass

    @staticmethod
    def check_quit_event():
        pass

    def update_ui(self, food: Point, score: float, snake: List[Point]):
        pass

    def clock_tick(self):
        pass


@zope.interface.implementer(PygameControllerInterface)
class PygameController:

    def __init__(self, w, h, block_size):
        pygame.init()
        font_path = Path(__file__).parent.parent.parent / 'resources/arial.ttf'
        self.font = pygame.font.Font(font_path, 25)
        self.block_size = block_size

        # init display
        self.display = pygame.display.set_mode((w, h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

    @staticmethod
    def check_quit_event():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    def update_ui(self, food: Point, score: float, snake: List[Point]):
        self.display.fill(BLACK)

        for pt in snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(food.x, food.y, self.block_size, self.block_size))

        text = self.font.render("Score: " + str(score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def clock_tick(self):
        self.clock.tick(SPEED)

