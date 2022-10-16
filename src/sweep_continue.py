from typing import Dict, List

import wandb

from src.agent import train
from src.collision_type import CollisionType

if __name__ == '__main__':
    sweep_id = "8wos4ahr"
    wandb.agent(sweep_id, function=train, count=10)
