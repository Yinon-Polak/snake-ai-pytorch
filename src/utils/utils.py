from typing import List


def flatten(l: List):
    return [item for sublist in l for item in sublist]
