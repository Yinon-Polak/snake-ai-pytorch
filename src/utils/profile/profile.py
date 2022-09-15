import cProfile
import os
import re
from datetime import datetime

from src.agent import train


if __name__ == '__main__':
    agent_kwargs = {"max_games": 1000}
    timestamp = datetime.now()
    output_path = f'/Users/yinonpolak/Library/CloudStorage/GoogleDrive-yinonpolak@gmail.com/My Drive/projects/snake-ai-pytorch/src/outputs/restats-{timestamp}'
    cProfile.run('train(\'split-collision\', 0, agent_kwargs=agent_kwargs)', output_path)
